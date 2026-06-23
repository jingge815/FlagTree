"""Shared helpers for tutorial kernels."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def default_block_m(rows: int) -> int:
    # Keep the M dimension tensor-core friendly even for decode where rows=1.
    return 16


def cuda_events_time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def cdiv(a: int, b: int) -> int:
    return triton.cdiv(a, b)


NORM_LOOP_CONFIGS = [
    triton.Config({"TILE_N": tile_n}, num_warps=warps)
    for tile_n in (1024, 2048, 4096, 8192)
    for warps in (4, 8, 16)
]


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


def as_normalized_shape_tuple(normalized_shape: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(normalized_shape, int):
        return (normalized_shape, )
    return tuple(int(dim) for dim in normalized_shape)


def norm_2d_shape(x: torch.Tensor, normalized_shape: int | tuple[int, ...] | list[int]) -> tuple[int, int]:
    shape = as_normalized_shape_tuple(normalized_shape)
    if len(shape) != 1:
        raise NotImplementedError("RMSNorm kernels currently support one normalized dimension")
    if x.shape[-1] != shape[0]:
        raise ValueError(f"normalized_shape {shape} does not match input last dimension {x.shape[-1]}")
    return math.prod(x.shape[:-1]), shape[0]


def row_stride(tensor: torch.Tensor) -> int:
    return tensor.stride(-2) if tensor.ndim >= 2 else 0
