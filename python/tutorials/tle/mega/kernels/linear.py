"""BF16 linear projection kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import cdiv, default_block_m, require_cuda_contiguous, row_stride


def _linear_tilewise_tma_pre_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    nargs["x_desc"].block_shape = [block_m, block_k]
    nargs["weight_desc"].block_shape = [block_n, block_k]
    nargs["out_desc"].block_shape = [block_m, block_n]


def _linear_hopper_tma_pre_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    nargs["x_desc"].block_shape = [block_m, block_k]
    nargs["weight_desc"].block_shape = [block_n, block_k]
    nargs["out_desc"].block_shape = [block_m, block_n]


def _linear_tilewise_prune_configs(configs, named_args, **kwargs):
    m = int(kwargs["M"])
    n = int(kwargs["N"])
    k = int(kwargs["K"])
    pruned = []
    for config in configs:
        block_m = config.kwargs["BLOCK_SIZE_M"]
        block_n = config.kwargs["BLOCK_SIZE_N"]
        block_k = config.kwargs["BLOCK_SIZE_K"]
        if m < 64 and block_m > 32:
            continue
        if n <= 64 and block_n > 64:
            continue
        if k < 128 and block_k > 64:
            continue
        pruned.append(config)
    return pruned or configs[:1]


def _linear_hopper_tma_prune_configs(configs, named_args, **kwargs):
    m = int(kwargs["M"])
    n = int(kwargs["N"])
    k = int(kwargs["K"])
    pruned = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        if m < 64 and block_m > 16:
            continue
        if m < 256 and block_m > 128:
            continue
        if n <= 64 and block_n > 64:
            continue
        if k < 128 and block_k > 64:
            continue
        pruned.append(config)
    return pruned or configs[:1]


_LINEAR_TILEWISE_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3),
]


_LINEAR_TILEWISE_TMA_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
    triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_STAGES": 3}, num_warps=4,
                  num_stages=3, pre_hook=_linear_tilewise_tma_pre_hook),
]


_LINEAR_HOPPER_TMA_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3,
                  pre_hook=_linear_hopper_tma_pre_hook),
]


def _is_tma_compatible_2d(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 2 or tensor.stride(-1) != 1:
        return False
    elem_bytes = tensor.element_size()
    return all((stride * elem_bytes) % 16 == 0 for stride in tensor.stride()[:-1])


def _supports_host_tma(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> bool:
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] < 9:
        return False
    return _is_tma_compatible_2d(x) and _is_tma_compatible_2d(weight) and _is_tma_compatible_2d(out)


@triton.jit
def _streamk_prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.jit
def _streamk_swizzle_tile(
    tile_id,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@triton.jit(
    do_not_specialize=[
        "iters_per_pid",
        "iters_remaining",
        "iters_per_tile",
    ]
)
def _linear_streamk_first_wave_bf16(
    A,
    B,
    C,
    P,
    bias_ptr,
    M,
    N,
    K,
    locks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    iters_per_pid,
    iters_remaining,
    iters_per_tile,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # Adapted from FlagGems src/flag_gems/ops/mm_streamk.py.
    # SPDX-License-Identifier: Apache-2.0
    pid = tl.program_id(0)
    start_iter = pid * iters_per_pid + tl.minimum(pid, iters_remaining)
    last_iter = (pid + 1) * iters_per_pid + tl.minimum(pid + 1, iters_remaining)
    while start_iter < last_iter:
        iter_offset_in_tile = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - iter_offset_in_tile), last_iter)
        tile_id = start_iter // iters_per_tile
        pid_m, pid_n = _streamk_swizzle_tile(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

        A_base = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
        A_base += BLOCK_K * stride_ak * iter_offset_in_tile
        B_base = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn
        B_base += BLOCK_K * stride_bk * iter_offset_in_tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_base)
                b = tl.load(B_base)
            else:
                k_offset_in_tile = (current_iter % iters_per_tile) * BLOCK_K
                k_mask = (k_offset_in_tile + rk) < K
                a = tl.load(A_base, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_base, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            A_base += BLOCK_K * stride_ak
            B_base += BLOCK_K * stride_bk

        rm1 = tl.arange(0, BLOCK_M)
        rn1 = tl.arange(0, BLOCK_N)
        if start_iter % iters_per_tile != 0:
            P_ptr = P + pid * BLOCK_M * BLOCK_N + (rm1[:, None] * BLOCK_N + rn1[None, :])
            tl.store(P_ptr, acc, cache_modifier=".cg")
            tl.atomic_xchg(locks + pid, 1)
        else:
            next_pid = pid + 1
            stop_loading_iter = start_iter + iters_per_tile
            end = end_iter
            while end < stop_loading_iter:
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
                    pass
                P_ptr = P + next_pid * BLOCK_M * BLOCK_N + (rm1[:, None] * BLOCK_N + rn1[None, :])
                acc += tl.load(P_ptr, cache_modifier=".cg")
                end += iters_per_pid + (next_pid < iters_remaining)
                next_pid += 1
            if HAS_BIAS:
                bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
                acc += bias[None, :]
            C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_ptr, acc, mask=mask)
        start_iter = end_iter


@triton.jit
def _linear_streamk_classic_kernel(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    total_tiles_streamk,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Adapted from FlagGems src/flag_gems/ops/mm_streamk.py.
    # SPDX-License-Identifier: Apache-2.0
    tile_id = tl.program_id(0) + total_tiles_streamk
    pid_m, pid_n = _streamk_swizzle_tile(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    prev_multiple = _streamk_prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = start_k + tl.arange(0, BLOCK_K)
        a = tl.load(A + ram[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    rk = prev_multiple + tl.arange(0, BLOCK_K)
    mask_k = rk < K
    a = tl.load(A + ram[:, None] * stride_am + rk[None, :] * stride_ak, mask=mask_k[None, :], other=0.0)
    b = tl.load(B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn, mask=mask_k[:, None], other=0.0)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    C_ptr = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C_ptr, acc.to(C.dtype.element_ty), mask=mask)


@triton.jit
def _linear_hopper_tma_kernel(
    x_desc,
    weight_desc,
    out_desc,
    bias_ptr,
    M,
    N,
    K,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Adapted from FlagGems runtime/backend/_nvidia/hopper/ops/mm.py.
    # SPDX-License-Identifier: Apache-2.0
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = (pid_m * BLOCK_M).to(tl.int32)
    offs_n = (pid_n * BLOCK_N).to(tl.int32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = (k_iter * BLOCK_K).to(tl.int32)
        a = x_desc.load([offs_m, offs_k])
        b = weight_desc.load([offs_n, offs_k])
        accumulator = tl.dot(a, b.T, acc=accumulator, allow_tf32=False)

    if HAS_BIAS:
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]
    out_desc.store([offs_m, offs_n], accumulator.to(out_desc.dtype))


@triton.jit
def _linear_splitk_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Adapted from FlagGems runtime/backend/_nvidia/hopper/ops/mm.py.
    # SPDX-License-Identifier: Apache-2.0
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(k_start, k_end):
        k0 = k_iter * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=mask, sem="relaxed")


@triton.jit
def _linear_gemv_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Adapted from FlagGems src/flag_gems/ops/mv.py.
    # SPDX-License-Identifier: Apache-2.0
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offs_k = tl.arange(0, BLOCK_K)[None, :]
    n_mask = offs_n < N
    weight_ptrs = weight_ptr + offs_n * stride_wn + offs_k * stride_wk
    x_ptrs = x_ptr + offs_k * stride_xk
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for start_k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
        k_mask = start_k + offs_k < K
        weight = tl.load(weight_ptrs, mask=n_mask & k_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        acc += weight * x
        weight_ptrs += BLOCK_K * stride_wk
        x_ptrs += BLOCK_K * stride_xk

    out = tl.sum(acc, axis=1)
    offs_out = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_out, mask=offs_out < N, other=0.0).to(tl.float32)
        out += bias
    tl.store(out_ptr + offs_out * stride_on, out, mask=offs_out < N)


@triton.jit
def _linear_gemv_persistent_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n_base = tl.arange(0, BLOCK_N)[:, None]
    offs_k = tl.arange(0, BLOCK_K)[None, :]
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    for tile_n in tl.range(pid, num_n_tiles, NUM_CTAS):
        offs_n = tile_n * BLOCK_N + offs_n_base
        n_mask = offs_n < N
        weight_ptrs = weight_ptr + offs_n * stride_wn + offs_k * stride_wk
        x_ptrs = x_ptr + offs_k * stride_xk
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for start_k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            k_mask = start_k + offs_k < K
            weight = tl.load(weight_ptrs, mask=n_mask & k_mask, other=0.0).to(tl.float32)
            x = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            acc += weight * x
            weight_ptrs += BLOCK_K * stride_wk
            x_ptrs += BLOCK_K * stride_xk

        out = tl.sum(acc, axis=1)
        offs_out = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_out, mask=offs_out < N, other=0.0).to(tl.float32)
            out += bias
        tl.store(out_ptr + offs_out * stride_on, out, mask=offs_out < N)


@triton.jit
def _linear_tilewise_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Adapted from ByteDance-Seed/Triton-distributed mega_triton_kernel/kernels/linear.py.
    # SPDX-FileCopyrightText: Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
    # SPDX-License-Identifier: MIT
    tile_id = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_bn[:, None] * K + offs_k[None, :])
        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b.T, accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _linear_tilewise_tma_kernel(
    x_desc,
    weight_desc,
    out_desc,
    bias_ptr,
    M,
    N,
    K,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        k0 = ki * BLOCK_SIZE_K
        a = x_desc.load([offs_m, k0])
        b = weight_desc.load([offs_n, k0])
        accumulator = tl.dot(a, b.T, accumulator)

    if HAS_BIAS:
        offs = offs_n + tl.arange(0, BLOCK_SIZE_N)
        bias = tl.load(bias_ptr + offs, mask=offs < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]
    out_desc.store([offs_m, offs_n], accumulator.to(out_desc.dtype))


_linear_tilewise_kernel_autotuned = triton.autotune(
    configs=_LINEAR_TILEWISE_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "HAS_BIAS"],
    prune_configs_by={"early_config_prune": _linear_tilewise_prune_configs},
    cache_results=True,
)(_linear_tilewise_kernel)


_linear_tilewise_tma_kernel_autotuned = triton.autotune(
    configs=_LINEAR_TILEWISE_TMA_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "HAS_BIAS"],
    prune_configs_by={"early_config_prune": _linear_tilewise_prune_configs},
    cache_results=True,
)(_linear_tilewise_tma_kernel)


_linear_hopper_tma_kernel_autotuned = triton.autotune(
    configs=_LINEAR_HOPPER_TMA_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "HAS_BIAS"],
    prune_configs_by={"early_config_prune": _linear_hopper_tma_prune_configs},
    cache_results=True,
)(_linear_hopper_tma_kernel)


def _streamk_block_m(m: int) -> int:
    if m <= 16:
        return 16
    return 128


def _use_streamk_shape(m: int, n: int, k: int) -> bool:
    if m == 1 and n == 5_120 and k in (8_192, 25_600):
        return True
    if m <= 128 and n == 5_120 and k == 25_600:
        return True
    return m >= 1_024 and n == 10_240 and k == 5_120


def _use_gemv_shape(m: int, n: int, k: int) -> bool:
    if m != 1:
        return False
    return (n, k) in {
        (10_240, 5_120),
        (5_120, 8_192),
        (5_120, 25_600),
        (51_200, 5_120),
        (151_936, 5_120),
    }


def _use_gemv_persistent_shape(m: int, n: int, k: int) -> bool:
    return m == 1 and n == 5_120 and k == 8_192


def _use_splitk_shape(m: int, n: int, k: int) -> bool:
    if n == 10_240 and k == 5_120 and m <= 128:
        return True
    if n == 5_120 and k == 8_192 and m <= 128:
        return True
    return n == 5_120 and k == 25_600 and m == 1


def _use_hopper_tma_shape(m: int, n: int, k: int) -> bool:
    if n == 10_240 and k == 5_120 and m >= 512:
        return True
    if n == 5_120 and k in (8_192, 25_600) and m >= 128:
        return True
    return n == 51_200 and k == 5_120 and (m == 1 or m >= 512)


def linear_backend_name(m: int, n: int, k: int) -> str:
    if _use_gemv_persistent_shape(m, n, k):
        return "gemv_persistent"
    if _use_gemv_shape(m, n, k):
        return "gemv"
    if _use_splitk_shape(m, n, k):
        return "splitk"
    if _use_hopper_tma_shape(m, n, k):
        return "hopper_tma"
    return "streamk" if _use_streamk_shape(m, n, k) else "tilewise"


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` with stream-K or tilewise based on GEMM shape."""
    if block_m is None and block_n is None and block_k is None:
        if x.dim() == 2 and weight.dim() == 2:
            m = int(x.shape[0])
            n = int(weight.shape[0])
            k = int(x.shape[1])
            if _use_gemv_persistent_shape(m, n, k):
                return linear_gemv_persistent(x, weight, bias)
            if _use_gemv_shape(m, n, k):
                return linear_gemv(x, weight, bias)
            if bias is None and _use_splitk_shape(m, n, k):
                return linear_splitk(x, weight, bias)
            if _use_hopper_tma_shape(m, n, k):
                return linear_hopper_tma(x, weight, bias)
            if _use_streamk_shape(m, n, k):
                return linear_streamk(x, weight, bias)
    return linear_tilewise(x, weight, bias, block_m=block_m, block_n=block_n, block_k=block_k)


def linear_streamk(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int | None = None,
    block_n: int = 128,
    block_k: int = 128,
    group_m: int = 8,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Stream-K BF16 matmul adapted from FlagGems ``mm_streamk.py``."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_streamk currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != n:
            raise ValueError(f"bias size {bias.numel()} does not match output size {n}")

    bm = _streamk_block_m(m) if block_m is None else block_m
    bn = block_n
    bk = block_k
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    blocks_m = cdiv(m, bm)
    blocks_n = cdiv(n, bn)
    total_tiles = blocks_m * blocks_n
    iters_per_tile = cdiv(k, bk)
    tiles_per_wave = int(sm_count or torch.cuda.get_device_properties(x.device).multi_processor_count)
    number_cooperative_tiles = total_tiles % tiles_per_wave
    number_other_tiles = total_tiles - number_cooperative_tiles
    if number_other_tiles > 0 and number_cooperative_tiles < tiles_per_wave * 0.5:
        number_cooperative_tiles += tiles_per_wave
    elif number_other_tiles > 0 and number_cooperative_tiles > tiles_per_wave * 0.8:
        number_cooperative_tiles = 0

    num_warps = 4 if bm <= 64 else 8
    if number_cooperative_tiles > 0:
        total_iters_streamk = number_cooperative_tiles * iters_per_tile
        iters_per_pid = total_iters_streamk // tiles_per_wave
        iters_remaining = total_iters_streamk % tiles_per_wave
        locks = torch.zeros((tiles_per_wave,), device=x.device, dtype=torch.int32)
        partials = torch.empty((tiles_per_wave, bm, bn), device=x.device, dtype=torch.float32)
        _linear_streamk_first_wave_bf16[(tiles_per_wave, )](
            x,
            weight,
            out,
            partials,
            bias if bias is not None else x,
            m,
            n,
            k,
            locks,
            x.stride(0),
            x.stride(1),
            weight.stride(1),
            weight.stride(0),
            out.stride(0),
            out.stride(1),
            iters_per_pid=iters_per_pid,
            iters_remaining=iters_remaining,
            iters_per_tile=iters_per_tile,
            HAS_BIAS=bias is not None,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=group_m,
            EVEN_K=(k % bk) == 0,
            num_stages=3,
            num_warps=num_warps,
        )

    remaining_tiles = total_tiles - number_cooperative_tiles
    if remaining_tiles > 0:
        _linear_streamk_classic_kernel[(remaining_tiles, )](
            x,
            weight,
            out,
            bias if bias is not None else x,
            m,
            n,
            k,
            x.stride(0),
            x.stride(1),
            weight.stride(1),
            weight.stride(0),
            out.stride(0),
            out.stride(1),
            number_cooperative_tiles,
            HAS_BIAS=bias is not None,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=group_m,
            num_stages=3,
            num_warps=num_warps,
        )
    return out


def linear_hopper_tma(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hopper host-TMA GEMM adapted from FlagGems Hopper mm.py."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_hopper_tma currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != n:
            raise ValueError(f"bias size {bias.numel()} does not match output size {n}")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if not _supports_host_tma(x, weight, out):
        raise RuntimeError("linear_hopper_tma requires Hopper-compatible contiguous 2D tensors for host TMA")

    from triton.tools.tensor_descriptor import TensorDescriptor

    dummy_block = [1, 1]
    x_desc = TensorDescriptor.from_tensor(x, dummy_block)
    weight_desc = TensorDescriptor.from_tensor(weight, dummy_block)
    out_desc = TensorDescriptor.from_tensor(out, dummy_block)
    grid = lambda meta: (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), )
    _linear_hopper_tma_kernel_autotuned[grid](
        x_desc,
        weight_desc,
        out_desc,
        bias if bias is not None else x,
        M=m,
        N=n,
        K=k,
        HAS_BIAS=bias is not None,
    )
    return out


def _splitk_params(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    block_m = 16 if m <= 16 else 128
    block_n = 128
    block_k = 128
    if m <= 16 and k >= 16_384:
        split_k = 8
    elif m <= 128 and k >= 16_384:
        split_k = 4
    elif k >= 8_192:
        split_k = 2
    else:
        split_k = 1
    num_warps = 4 if block_m <= 64 else 8
    return block_m, block_n, block_k, split_k, num_warps


def linear_splitk(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Split-K BF16 matmul adapted from FlagGems Hopper mm.py."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if bias is not None:
        raise ValueError("linear_splitk currently supports no-bias projections only")
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_splitk currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    out.zero_()
    block_m, block_n, block_k, split_k, num_warps = _splitk_params(m, n, k)
    grid = (cdiv(m, block_m) * cdiv(n, block_n), split_k)
    _linear_splitk_kernel[grid](
        x,
        weight,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_stages=3,
        num_warps=num_warps,
    )
    return out


def _gemv_params(n: int, k: int) -> tuple[int, int, int]:
    if n == 10_240 and k == 5_120:
        return 16, 128, 8
    if n == 5_120 and k == 8_192:
        return 8, 512, 8
    if n == 5_120 and k == 25_600:
        return 8, 1_024, 8
    if n == 51_200 and k == 5_120:
        return 8, 512, 8
    if n == 151_936 and k == 5_120:
        return 8, 1_024, 8
    return 16, 256, 8


def _gemv_persistent_params(n: int, k: int) -> tuple[int, int, int, int]:
    if n == 5_120 and k == 8_192:
        return 4, 1_024, 4, 8
    block_n, block_k, num_warps = _gemv_params(n, k)
    return block_n, block_k, num_warps, 1


def linear_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-only BF16 matrix-vector projection adapted from FlagGems ``mv.py``."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_gemv currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if m != 1:
        raise ValueError(f"linear_gemv is decode-only and expects M=1, got M={m}")
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != n:
            raise ValueError(f"bias size {bias.numel()} does not match output size {n}")

    block_n, block_k, num_warps = _gemv_params(n, k)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _linear_gemv_kernel[(cdiv(n, block_n), )](
        x,
        weight,
        bias if bias is not None else x,
        out,
        N=n,
        K=k,
        stride_xk=x.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_on=out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_STAGES=3,
        num_stages=3,
        num_warps=num_warps,
    )
    return out


def linear_gemv_persistent(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 3,
    cta_per_sm: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Decode-only persistent GEMV with a tunable number of resident CTAs per SM."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_gemv_persistent currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if m != 1:
        raise ValueError(f"linear_gemv_persistent expects M=1, got M={m}")
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != n:
            raise ValueError(f"bias size {bias.numel()} does not match output size {n}")

    default_block_n, default_block_k, default_num_warps, default_cta_per_sm = _gemv_persistent_params(n, k)
    block_n = int(block_n or default_block_n)
    block_k = int(block_k or default_block_k)
    num_warps = int(num_warps or default_num_warps)
    cta_per_sm = int(cta_per_sm or default_cta_per_sm)
    if cta_per_sm < 1:
        raise ValueError(f"cta_per_sm must be positive, got {cta_per_sm}")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    num_n_tiles = cdiv(n, block_n)
    num_sms = int(sm_count or torch.cuda.get_device_properties(x.device).multi_processor_count)
    num_ctas = min(num_sms * cta_per_sm, num_n_tiles)
    grid = (num_ctas, )
    _linear_gemv_persistent_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        N=n,
        K=k,
        stride_xk=x.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_on=out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_STAGES=num_stages,
        NUM_CTAS=num_ctas,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out


def linear_tilewise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_stages: int = 3,
) -> torch.Tensor:
    """Standalone wrapper around ByteDance Triton-distributed tile-wise matmul."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_tilewise currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != n:
            raise ValueError(f"bias size {bias.numel()} does not match output size {n}")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if block_m is not None or block_n is not None or block_k is not None:
        bm = default_block_m(m) if block_m is None else block_m
        bn = 64 if block_n is None else block_n
        bk = 64 if block_k is None else block_k
        grid = (cdiv(m, bm) * cdiv(n, bn), )
        _linear_tilewise_kernel[grid](
            x,
            weight,
            bias if bias is not None else x,
            out,
            m,
            n,
            k,
            bias is not None,
            bm,
            bn,
            bk,
            num_stages,
            num_warps=4,
            num_stages=num_stages,
        )
    else:
        grid = lambda meta: (cdiv(m, meta["BLOCK_SIZE_M"]) * cdiv(n, meta["BLOCK_SIZE_N"]), )
        _linear_tilewise_kernel_autotuned[grid](
            x,
            weight,
            bias if bias is not None else x,
            out,
            M=m,
            N=n,
            K=k,
            HAS_BIAS=bias is not None,
        )
    return out


def linear_tilewise_tma(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_stages: int = 3,
) -> torch.Tensor:
    """Tile-wise matmul using host TensorDescriptor/TMA loads."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear_tilewise_tma currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != n:
            raise ValueError(f"bias size {bias.numel()} does not match output size {n}")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if not _supports_host_tma(x, weight, out):
        raise RuntimeError("linear_tilewise_tma requires Hopper-compatible contiguous 2D tensors for host TMA")

    from triton.tools.tensor_descriptor import TensorDescriptor

    if block_m is not None or block_n is not None or block_k is not None:
        bm = default_block_m(m) if block_m is None else block_m
        bn = 64 if block_n is None else block_n
        bk = 64 if block_k is None else block_k
        x_desc = TensorDescriptor.from_tensor(x, [bm, bk])
        weight_desc = TensorDescriptor.from_tensor(weight, [bn, bk])
        out_desc = TensorDescriptor.from_tensor(out, [bm, bn])
        grid = (cdiv(m, bm) * cdiv(n, bn), )
        _linear_tilewise_tma_kernel[grid](
            x_desc,
            weight_desc,
            out_desc,
            bias if bias is not None else x,
            m,
            n,
            k,
            bias is not None,
            bm,
            bn,
            bk,
            num_stages,
            num_warps=4,
            num_stages=num_stages,
        )
    else:
        dummy_block = [1, 1]
        x_desc = TensorDescriptor.from_tensor(x, dummy_block)
        weight_desc = TensorDescriptor.from_tensor(weight, dummy_block)
        out_desc = TensorDescriptor.from_tensor(out, dummy_block)
        grid = lambda meta: (cdiv(m, meta["BLOCK_SIZE_M"]) * cdiv(n, meta["BLOCK_SIZE_N"]), )
        _linear_tilewise_tma_kernel_autotuned[grid](
            x_desc,
            weight_desc,
            out_desc,
            bias if bias is not None else x,
            M=m,
            N=n,
            K=k,
            HAS_BIAS=bias is not None,
        )
    return out


def lm_head(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute lm_head projection, using GEMV for single-token decode."""
    if (
        x.dim() == 2
        and weight.dim() == 2
        and _use_gemv_shape(int(x.shape[0]), int(weight.shape[0]), int(x.shape[1]))
    ):
        return linear_gemv(x, weight, bias)
    return linear_tilewise_tma(x, weight, bias)


def qkv_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    q_dim: int,
    kv_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute packed QKV projection and return split Q, K, V views."""
    qkv = linear(x, weight, bias)
    q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
    return q, k, v


@triton.jit
def _silu_and_mul_packed_kernel(
    packed_ptr,
    out_ptr,
    inner: tl.constexpr,
    packed_stride_r,
    packed_stride_c,
    out_stride_r,
    out_stride_c,
    tile_size: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * tile_size + tl.arange(0, tile_size)
    mask = col < inner
    gate = tl.load(
        packed_ptr + row * packed_stride_r + col * packed_stride_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        packed_ptr + row * packed_stride_r + (col + inner) * packed_stride_c,
        mask=mask,
        other=0.0,
    )
    silu = tl.fdiv(gate, (1.0 + tl.exp(-gate)))
    tl.store(
        out_ptr + row * out_stride_r + col * out_stride_c,
        (silu * up).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _silu_and_mul_packed_launch(inner: int) -> tuple[int, int]:
    tile_size = min(1024, triton.next_power_of_2(inner))
    return tile_size, 1


def silu_and_mul(packed: torch.Tensor) -> torch.Tensor:
    return silu_and_mul_packed(packed)


def silu_and_mul_packed(packed: torch.Tensor) -> torch.Tensor:
    if packed.shape[-1] % 2 != 0:
        raise ValueError(f"silu_and_mul packed last dimension must be even, got {packed.shape}")
    out = torch.empty((*packed.shape[:-1], packed.shape[-1] // 2), device=packed.device, dtype=packed.dtype)
    return silu_and_mul_packed_out(packed, out)


def silu_and_mul_packed_out(packed: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if packed.shape[-1] != out.shape[-1] * 2 or packed.shape[:-1] != out.shape[:-1]:
        raise ValueError(f"silu_and_mul packed input/output shape mismatch: packed={packed.shape}, out={out.shape}")
    inner = out.shape[-1]
    rows = out.numel() // inner
    tile_size, num_warps = _silu_and_mul_packed_launch(inner)
    _silu_and_mul_packed_kernel[(rows, cdiv(inner, tile_size))](
        packed,
        out,
        inner,
        row_stride(packed),
        packed.stride(-1),
        row_stride(out),
        out.stride(-1),
        tile_size,
        num_warps=num_warps,
    )
    return out
