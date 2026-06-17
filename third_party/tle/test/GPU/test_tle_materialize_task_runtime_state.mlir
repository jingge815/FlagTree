// RUN: triton-opt %s --triton-tle-analyze-task-graph --triton-tle-materialize-task-scheduler --triton-tle-materialize-task-runtime-state -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: tle.requires_cooperative_grid = 1 : i32
  tt.func private @producer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func private @consumer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  // CHECK-LABEL: tt.func @materialize_two_task_runtime_state
  // CHECK: tle.task_graph.scheduler
  // CHECK: tle.task_graph.runtime_state
  // CHECK-SAME: alignment_bytes = 4 : i64
  // CHECK-SAME: completed_count_offset_bytes = 16 : i64
  // CHECK-SAME: counter_bytes = 4 : i64
  // CHECK-SAME: counter_type = "i32"
  // CHECK-SAME: dep_counters_offset_bytes = 20 : i64
  // CHECK-SAME: init_flag_offset_bytes = 0 : i64
  // CHECK-SAME: num_instances = 4 : i64
  // CHECK-SAME: queue_capacity = 4 : i64
  // CHECK-SAME: queue_element_bytes = 4 : i64
  // CHECK-SAME: queue_head_offset_bytes = 8 : i64
  // CHECK-SAME: queue_lock_offset_bytes = 4 : i64
  // CHECK-SAME: queue_storage_offset_bytes = 36 : i64
  // CHECK-SAME: queue_tail_offset_bytes = 12 : i64
  // CHECK-SAME: state_size_bytes = 52 : i64
  tt.func @materialize_two_task_runtime_state(%mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    tle.task.declare {callee = @producer_body, domain_shape = array<i64: 2>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {callee = @consumer_body, domain_shape = array<i64: 2>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}
