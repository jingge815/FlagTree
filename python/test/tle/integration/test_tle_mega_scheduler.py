# flagtree tle
"""Executable tests for the TLE mega task scheduler MVP."""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir, tle
from triton.compiler import make_backend
from triton.runtime import driver


_MEGA_TUTORIAL_ROOT = Path(__file__).resolve().parents[3] / "tutorials" / "tle" / "mega"
sys.path.insert(0, str(_MEGA_TUTORIAL_ROOT))
from kernels.linear_rmsnorm import (  # noqa: E402
    _materialize_linear_rmsnorm_scheduler,
    linear_rmsnorm_mega_scheduler,
    linear_rmsnorm_reference,
    linear_rmsnorm_triton_baseline,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA to execute the generated cooperative scheduler kernel",
)


@triton.jit
def _producer_ref_kernel(mid, n: tl.constexpr):
    tile = tl.program_id(0)
    if tile < n:
        tl.store(mid + tile, 7)


@triton.jit
def _consumer_ref_kernel(mid, out, n: tl.constexpr):
    tile = tl.program_id(0)
    if tile < n:
        value = tl.load(mid + tile)
        tl.store(out + tile, value + 5)


def _task_scheduler_ttir() -> str:
    return textwrap.dedent(
        r"""
        module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
          tt.func private @producer_body(%tile: i32, %mid: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
            %c7 = arith.constant 7 : i32
            %ptr = tt.addptr %mid, %tile : !tt.ptr<i32>, i32
            tt.store %ptr, %c7 : !tt.ptr<i32>
            tt.return
          }

          tt.func private @consumer_body(%tile: i32, %mid: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
            %mptr = tt.addptr %mid, %tile : !tt.ptr<i32>, i32
            %v = tt.load %mptr : !tt.ptr<i32>
            %c5 = arith.constant 5 : i32
            %sum = arith.addi %v, %c5 : i32
            %optr = tt.addptr %out, %tile : !tt.ptr<i32>, i32
            tt.store %optr, %sum : !tt.ptr<i32>
            tt.return
          }

          tt.func @task_scheduler_exec(%mid: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
            tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 2>} : !tt.ptr<i32>
            tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 2>} : !tt.ptr<i32>
            tle.task.declare {callee = @producer_body, domain_shape = array<i64: 2>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
            tle.task.declare {callee = @consumer_body, domain_shape = array<i64: 2>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
            tt.return
          }
        }
        """
    ).strip()


def _materialize_scheduler_ttgir(tmp_path, target):
    backend = make_backend(target)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    ttir_path = tmp_path / "task_scheduler_exec.ttir"
    ttir_path.write_text(_task_scheduler_ttir())
    module = ir.parse_mlir_module(str(ttir_path), context)
    module.context = context

    pm = ir.pass_manager(context)
    pm.enable_debug()
    tle.passes.add_verify_task_graph(pm)
    tle.passes.add_analyze_task_graph(pm)
    tle.passes.add_materialize_task_scheduler(pm)
    tle.passes.add_materialize_task_runtime_state(pm)
    pm.run(module, "materialize_tle_mega_scheduler")

    ttgir_path = tmp_path / "task_scheduler_exec.ttgir"
    ttgir_path.write_text(str(module))
    return ttgir_path


def test_generated_task_scheduler_runs_two_task_graph(tmp_path, with_allocator):
    target = driver.active.get_current_target()
    if target.backend != "cuda":
        pytest.skip(f"Requires CUDA target, got {target.backend!r}")

    n_tiles = 2
    mid_ref = torch.full((n_tiles,), -1, device="cuda", dtype=torch.int32)
    out_ref = torch.full((n_tiles,), -1, device="cuda", dtype=torch.int32)
    _producer_ref_kernel[(n_tiles, )](mid_ref, n_tiles)
    _consumer_ref_kernel[(n_tiles, )](mid_ref, out_ref, n_tiles)

    ttgir_path = _materialize_scheduler_ttgir(tmp_path, target)
    scheduler = triton.compile(str(ttgir_path), target=target)
    assert scheduler.metadata.launch_cooperative_grid
    assert re.search(r"call .*@producer_body", scheduler.asm["llir"])
    assert re.search(r"call .*@consumer_body", scheduler.asm["llir"])

    mid = torch.full((n_tiles,), -1, device="cuda", dtype=torch.int32)
    out = torch.full((n_tiles,), -1, device="cuda", dtype=torch.int32)
    scheduler[(n_tiles, 1, 1)](mid, out)

    torch.testing.assert_close(mid, mid_ref)
    torch.testing.assert_close(out, out_ref)


def test_linear_rmsnorm_scheduler_graph_ir_and_publish_fence(tmp_path):
    target = driver.active.get_current_target()
    if target.backend != "cuda":
        pytest.skip(f"Requires CUDA target, got {target.backend!r}")

    ttgir_path, materialized = _materialize_linear_rmsnorm_scheduler(2, 4, 1.0e-5, tmp_path, target)
    assert 'task = "linear_tile"' in materialized
    assert 'task = "rms_reduce"' in materialized
    assert 'task = "rms_apply"' in materialized
    assert 'producer = "linear_tile[0,0]"' in materialized
    assert 'consumer = "rms_reduce[0]"' in materialized
    assert 'consumer = "rms_apply[1,3]"' in materialized
    assert 'tile = "linear_to_rms[1,3]"' in materialized
    assert 'tile = "rms_stat[1]"' in materialized

    scheduler = triton.compile(str(ttgir_path), target=target)
    assert scheduler.metadata.launch_cooperative_grid
    assert re.search(r"call .*@linear_tile_body", scheduler.asm["llir"])
    assert re.search(r"call .*@rms_reduce_body", scheduler.asm["llir"])
    assert re.search(r"call .*@rms_apply_body", scheduler.asm["llir"])
    assert "membar.gl;" in scheduler.asm["ptx"]


@pytest.mark.parametrize("rows, hidden", [(1, 2), (2, 4), (3, 5)])
def test_linear_rmsnorm_generated_scheduler_matches_torch(rows, hidden):
    target = driver.active.get_current_target()
    if target.backend != "cuda":
        pytest.skip(f"Requires CUDA target, got {target.backend!r}")

    torch.manual_seed(rows * 100 + hidden)
    x = torch.randn((rows, 1), device="cuda", dtype=torch.float32)
    linear_weight = torch.randn((hidden, 1), device="cuda", dtype=torch.float32)
    rms_weight = torch.randn((hidden, ), device="cuda", dtype=torch.float32)

    out = linear_rmsnorm_mega_scheduler(x, linear_weight, rms_weight, eps=1.0e-5)
    baseline = linear_rmsnorm_triton_baseline(x, linear_weight, rms_weight, eps=1.0e-5)
    ref = linear_rmsnorm_reference(x, linear_weight, rms_weight, eps=1.0e-5)
    torch.testing.assert_close(out, ref, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(baseline, ref, rtol=1.0e-5, atol=1.0e-5)
