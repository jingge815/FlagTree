"""Mega decode-only linear + fused-add RMSNorm kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import cdiv, next_power_of_2, require_cuda_contiguous
from .norm import fused_add_rms_norm


def _decode_linear_tile_params(n: int, k: int) -> tuple[int, int, int]:
    if n == 5_120 and k == 8_192:
        return 4, 1_024, 3
    if n == 5_120 and k == 25_600:
        return 8, 1_024, 3
    if k <= 128:
        return 8, 128, 3
    if k <= 512:
        return 8, 512, 3
    return 8, 1_024, 3


@triton.jit
def _linear_fused_rmsnorm_pull_kernel(
    x,
    linear_weight,
    rms_weight,
    residual,
    scratch,
    out,
    linear_ready_flags,
    linear_ready_summary,
    N: tl.constexpr,
    K: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    WORKER_COUNT: tl.constexpr,
    EPS: tl.constexpr,
):
    worker = tl.program_id(0)
    for n_block in tl.range(worker, N_BLOCKS, WORKER_COUNT):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
        offs_k = tl.arange(0, BLOCK_K)[None, :]
        n_mask = offs_n < N
        weight_ptrs = linear_weight + offs_n * K + offs_k
        x_ptrs = x + offs_k
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for start_k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            k_mask = start_k + offs_k < K
            weight = tl.load(weight_ptrs, mask=n_mask & k_mask, other=0.0).to(tl.float32)
            x_value = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            acc += weight * x_value
            weight_ptrs += BLOCK_K
            x_ptrs += BLOCK_K

        linear = tl.sum(acc, axis=1)
        offs_out = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(scratch + offs_out, linear, mask=offs_out < N)
        tl.store(linear_ready_flags + n_block, 1)
        tl.atomic_add(linear_ready_summary, 1, sem="release", scope="gpu")

    if worker == 0:
        while tl.atomic_add(linear_ready_summary, 0, sem="acquire", scope="gpu") != N_BLOCKS:
            pass

        h = tl.arange(0, BLOCK_H)
        mask = h < N
        linear = tl.load(scratch + h, mask=mask, other=0.0)
        residual_value = tl.load(residual + h, mask=mask, other=0.0).to(tl.float32)
        combined = linear + residual_value
        tl.store(residual + h, combined, mask=mask)
        mean = tl.sum(combined * combined, axis=0) / N
        inv_rms = tl.rsqrt(mean + EPS)
        weight = tl.load(rms_weight + h, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + h, combined * inv_rms * weight, mask=mask)


def _validate_decode_inputs(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
) -> tuple[int, int]:
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("linear_weight", linear_weight)
    require_cuda_contiguous("residual", residual)
    require_cuda_contiguous("rms_weight", rms_weight)
    if bias is not None:
        raise ValueError("linear_fused_add_rms_norm_decode_mega currently supports bias-free projections only")
    if x.dim() != 2 or linear_weight.dim() != 2:
        raise ValueError(f"x and linear_weight must be 2D, got {tuple(x.shape)} and {tuple(linear_weight.shape)}")
    if x.shape[0] != 1:
        raise ValueError(f"linear_fused_add_rms_norm_decode_mega is decode-only and expects M=1, got {x.shape[0]}")
    n, k = linear_weight.shape
    if x.shape[1] != k:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(linear_weight.shape)}")
    if residual.shape != (1, n):
        raise ValueError(f"residual must have shape {(1, n)}, got {tuple(residual.shape)}")
    if rms_weight.shape != (n, ):
        raise ValueError(f"rms_weight must have shape {(n,)}, got {tuple(rms_weight.shape)}")
    if x.dtype != torch.bfloat16 or linear_weight.dtype != torch.bfloat16:
        raise ValueError("linear_fused_add_rms_norm_decode_mega requires bf16 x and linear_weight")
    if residual.dtype != torch.bfloat16 or rms_weight.dtype != torch.bfloat16:
        raise ValueError("linear_fused_add_rms_norm_decode_mega requires bf16 residual and rms_weight")
    if next_power_of_2(n) > 8192:
        raise ValueError("linear_fused_add_rms_norm_decode_mega currently supports output hidden <= 8192")
    total_instances = linear_fused_add_rms_norm_decode_task_instances(n, k)
    if total_instances > 4096:
        raise ValueError("linear_fused_add_rms_norm_decode_mega currently supports <=4096 task instances")
    return n, k


def linear_fused_add_rms_norm_decode_task_instances(n: int, k: int) -> int:
    block_n, _, _ = _decode_linear_tile_params(n, k)
    n_blocks = cdiv(n, block_n)
    return n_blocks + 1


def _decode_pull_worker_count(n: int, k: int, device: torch.device) -> int:
    block_n, _, _ = _decode_linear_tile_params(n, k)
    n_blocks = cdiv(n, block_n)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    ctas_per_sm = 8 if n == 5_120 and k == 8_192 else 1
    return max(1, min(num_sms * ctas_per_sm, n_blocks))


def linear_fused_add_rms_norm_decode_reference(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    linear = x.to(torch.float32) @ linear_weight.to(torch.float32).t()
    combined = linear + residual.to(torch.float32)
    inv_rms = torch.rsqrt(torch.mean(combined * combined, dim=-1, keepdim=True) + eps)
    out = (combined * inv_rms * rms_weight.to(torch.float32)).to(x.dtype)
    return out, combined.to(residual.dtype)


def linear_fused_add_rms_norm_decode_baseline(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .linear import linear

    linear_out = linear(x, linear_weight, bias)
    return fused_add_rms_norm(linear_out, residual, (linear_weight.shape[0], ), rms_weight, eps)


def linear_fused_add_rms_norm_decode_mega(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3 decode ``fused_add_rms_norm(linear(x), residual)``.

    The residual tensor is updated in place, matching ``fused_add_rms_norm``.
    """
    from triton.runtime import driver

    n, k = _validate_decode_inputs(x, linear_weight, bias, residual, rms_weight)
    target = driver.active.get_current_target()
    if target.backend != "cuda":
        raise RuntimeError(f"linear_fused_add_rms_norm_decode_mega requires CUDA target, got {target.backend!r}")

    out = torch.empty_like(residual)
    scratch = torch.empty((1, n), device=x.device, dtype=torch.float32)
    block_n, block_k, num_stages = _decode_linear_tile_params(n, k)
    n_blocks = cdiv(n, block_n)
    block_h = next_power_of_2(n)
    worker_count = _decode_pull_worker_count(n, k, x.device)
    linear_ready_flags = torch.zeros((n_blocks,), device=x.device, dtype=torch.int32)
    linear_ready_summary = torch.zeros((1,), device=x.device, dtype=torch.int32)
    _linear_fused_rmsnorm_pull_kernel[(worker_count, )](
        x,
        linear_weight,
        rms_weight,
        residual,
        scratch,
        out,
        linear_ready_flags,
        linear_ready_summary,
        N=n,
        K=k,
        N_BLOCKS=n_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
        NUM_STAGES=num_stages,
        WORKER_COUNT=worker_count,
        EPS=eps,
        num_warps=4,
        num_stages=num_stages,
    )
    return out, residual


def validate_linear_fused_add_rms_norm_decode_mega() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("validate_linear_fused_add_rms_norm_decode_mega requires CUDA")

    torch.manual_seed(0)
    for n, k in [(64, 128), (128, 256), (5120, 8192)]:
        x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        residual = torch.randn((1, n), device="cuda", dtype=torch.bfloat16)
        rms_weight = torch.randn((n, ), device="cuda", dtype=torch.bfloat16)
        residual_ref = residual.clone()
        residual_mega = residual.clone()
        ref, residual_ref = linear_fused_add_rms_norm_decode_reference(
            x,
            weight,
            residual_ref,
            rms_weight,
            eps=1.0e-5,
        )
        out, residual_out = linear_fused_add_rms_norm_decode_mega(
            x,
            weight,
            None,
            residual_mega,
            rms_weight,
            eps=1.0e-5,
        )
        torch.testing.assert_close(out, ref, rtol=2.0e-2, atol=2.0e-2)
        torch.testing.assert_close(residual_out, residual_ref, rtol=2.0e-2, atol=2.0e-2)
