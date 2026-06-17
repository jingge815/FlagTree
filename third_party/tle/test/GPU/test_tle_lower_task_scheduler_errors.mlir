// RUN: triton-opt %s --triton-tle-analyze-task-graph --triton-tle-materialize-task-scheduler --triton-tle-lower-task-scheduler -split-input-file -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @producer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func private @consumer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func @reject_parallel_initial_ready(%mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    // expected-error @+1 {{restricted task scheduler lowering requires exactly one initial ready task}}
    tle.task.declare {callee = @producer_body, domain_shape = array<i64: 2>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {callee = @consumer_body, domain_shape = array<i64: 2>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}
