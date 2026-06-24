# flagtree tle
"""Executable tests for the TLE pull-based mega scheduler prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from triton.runtime import driver


_MEGA_TUTORIAL_ROOT = Path(__file__).resolve().parents[3] / "tutorials" / "tle" / "mega"
sys.path.insert(0, str(_MEGA_TUTORIAL_ROOT))
from kernels.linear_fused_rmsnorm import (  # noqa: E402
    linear_fused_add_rms_norm_decode_mega,
    linear_fused_add_rms_norm_decode_reference,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA to execute the generated cooperative scheduler kernel",
)


def test_linear_fused_add_rmsnorm_decode_pull_scheduler_matches_torch(with_allocator):
    target = driver.active.get_current_target()
    if target.backend != "cuda":
        pytest.skip(f"Requires CUDA target, got {target.backend!r}")

    n, k = 64, 128
    torch.manual_seed(512)
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
    linear_weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((1, n), device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn((n,), device="cuda", dtype=torch.bfloat16)

    ref, residual_ref = linear_fused_add_rms_norm_decode_reference(
        x,
        linear_weight,
        residual.clone(),
        rms_weight,
        eps=1.0e-5,
    )
    out, residual_out = linear_fused_add_rms_norm_decode_mega(
        x,
        linear_weight,
        None,
        residual.clone(),
        rms_weight,
        eps=1.0e-5,
    )
    torch.testing.assert_close(out, ref, rtol=2.0e-2, atol=2.0e-2)
    torch.testing.assert_close(residual_out, residual_ref, rtol=2.0e-2, atol=2.0e-2)
