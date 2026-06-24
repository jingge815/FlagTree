"""Linear + RMSNorm reference and non-mega Triton baseline."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import next_power_of_2, require_cuda_contiguous
from .norm import rms_norm


@triton.jit
def _linear_rmsnorm_baseline_linear_kernel(
    x,
    linear_weight,
    linear_out,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < HIDDEN
    x_value = tl.load(x + row)
    weight = tl.load(linear_weight + h, mask=mask, other=0.0)
    tl.store(linear_out + row * HIDDEN + h, x_value * weight, mask=mask)


def _validate_linear_rmsnorm_inputs(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    caller: str,
) -> tuple[int, int]:
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("linear_weight", linear_weight)
    require_cuda_contiguous("rms_weight", rms_weight)
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}")
    if linear_weight.dim() != 2 or linear_weight.shape[1] != x.shape[1]:
        raise ValueError("linear_weight must be [hidden, in_features]")
    if rms_weight.dim() != 1 or rms_weight.shape[0] != linear_weight.shape[0]:
        raise ValueError("rms_weight must be 1D with length matching linear_weight.shape[0]")
    if x.dtype != torch.float32 or linear_weight.dtype != torch.float32 or rms_weight.dtype != torch.float32:
        raise ValueError(f"{caller} currently requires float32 tensors")
    rows, in_features = x.shape
    hidden = linear_weight.shape[0]
    if in_features != 1:
        raise ValueError(f"{caller} currently requires in_features == 1")
    if rows <= 0 or hidden <= 0:
        raise ValueError("rows and hidden must be positive")
    return rows, hidden


def linear_rmsnorm_triton_baseline(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Compute ``rmsnorm(linear(x))`` with normal non-mega Triton kernels."""
    rows, hidden = _validate_linear_rmsnorm_inputs(
        x,
        linear_weight,
        rms_weight,
        caller="linear_rmsnorm_triton_baseline",
    )
    linear_out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    _linear_rmsnorm_baseline_linear_kernel[(rows,)](
        x,
        linear_weight,
        linear_out,
        hidden,
        next_power_of_2(hidden),
    )
    return rms_norm(linear_out, (hidden,), rms_weight, eps)


def linear_rmsnorm_reference(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Torch reference for the linear + RMSNorm tutorial baseline."""
    linear = (x.to(torch.float32) @ linear_weight.to(torch.float32).t()).to(x.dtype)
    linear_f32 = linear.to(torch.float32)
    inv_rms = torch.rsqrt(torch.mean(linear_f32 * linear_f32, dim=-1, keepdim=True) + eps)
    return (linear_f32 * inv_rms * rms_weight.to(torch.float32)).to(x.dtype)
