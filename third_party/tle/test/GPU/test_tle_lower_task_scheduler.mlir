// RUN: triton-opt %s --triton-tle-analyze-task-graph --triton-tle-materialize-task-scheduler --triton-tle-lower-task-scheduler -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @producer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func private @consumer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_two_task_scheduler
  // CHECK-NOT: tle.task_graph.scheduler
  // CHECK-NOT: tle.task.declare
  // CHECK-NOT: tle.task_grid.create
  // CHECK: arith.constant 0 : i32
  // CHECK: tt.call @producer_body(%{{.*}}, %arg0, %arg1) : (i32, !tt.ptr<f16>, !tt.ptr<f16>) -> ()
  // CHECK: arith.constant 0 : i32
  // CHECK: tt.call @consumer_body(%{{.*}}, %arg0, %arg1) : (i32, !tt.ptr<f16>, !tt.ptr<f16>) -> ()
  tt.func @lower_two_task_scheduler(%mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task.declare {callee = @producer_body, domain_shape = array<i64: 1>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {callee = @consumer_body, domain_shape = array<i64: 1>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}
