// RUN: triton-opt %s --triton-tle-analyze-task-graph --triton-tle-materialize-task-scheduler -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @producer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func private @consumer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  // CHECK-LABEL: tt.func @materialize_two_task_scheduler
  // CHECK-NOT: tle.task_graph.analysis
  // CHECK: tle.task_graph.scheduler
  // CHECK-SAME: counter_type = "i32"
  // CHECK-SAME: callee = @producer_body
  // CHECK-SAME: task = "producer"
  // CHECK-SAME: task_id = 0 : i64
  // CHECK-SAME: callee = @consumer_body
  // CHECK-SAME: task = "consumer"
  // CHECK-SAME: task_id = 1 : i64
  // CHECK-SAME: edge_consumer_ids = array<i32: 2, 3>
  // CHECK-SAME: consumer = "consumer[1]"
  // CHECK-SAME: producer = "producer[1]"
  // CHECK-SAME: tile = "mid[1]"
  // CHECK-SAME: initial_ready = ["producer[0]", "producer[1]"]
  // CHECK-SAME: initial_ready_ids = array<i32: 0, 1>
  // CHECK-SAME: instance_coord_offsets = array<i32: 0, 1, 2, 3, 4>
  // CHECK-SAME: instance_coords = array<i32: 0, 1, 0, 1>
  // CHECK-SAME: instance_dep_counts = array<i32: 0, 0, 1, 1>
  // CHECK-SAME: instance_task_ids = array<i32: 0, 0, 1, 1>
  // CHECK-SAME: dep_count = 1 : i64
  // CHECK-SAME: instance = "consumer[1]"
  // CHECK-SAME: num_edges = 2 : i64
  // CHECK-SAME: num_instances = 4 : i64
  // CHECK-SAME: num_tasks = 2 : i64
  // CHECK-SAME: producer_edge_offsets = array<i32: 0, 1, 2, 2, 2>
  // CHECK-SAME: queue_capacity = 4 : i64
  // CHECK-SAME: task_domain_ranks = array<i32: 1, 1>
  // CHECK-SAME: task_names = ["producer", "consumer"]
  tt.func @materialize_two_task_scheduler(%mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    tle.task.declare {callee = @producer_body, domain_shape = array<i64: 2>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {callee = @consumer_body, domain_shape = array<i64: 2>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}
