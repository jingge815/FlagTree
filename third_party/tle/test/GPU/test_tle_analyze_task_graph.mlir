// RUN: triton-opt %s --triton-tle-analyze-task-graph -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @analyze_linear_rms_apply_graph
  // CHECK-SAME: tle.task_graph.analysis
  // CHECK-SAME: consumer = "rms_reduce[0]"
  // CHECK-SAME: producer = "linear_tile[0,0]"
  // CHECK-SAME: tile = "linear_to_rms[0,0]"
  // CHECK-SAME: consumer = "rms_apply[1,1]"
  // CHECK-SAME: producer = "rms_reduce[1]"
  // CHECK-SAME: tile = "rms_stat[1]"
  // CHECK-SAME: initial_ready = ["linear_tile[0,0]", "linear_tile[0,1]", "linear_tile[1,0]", "linear_tile[1,1]"]
  // CHECK-SAME: dep_count = 2 : i64
  // CHECK-SAME: instance = "rms_apply[1,1]"
  // CHECK-SAME: num_edges = 12 : i64
  // CHECK-SAME: num_instances = 10 : i64
  tt.func @analyze_linear_rms_apply_graph(%scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>, %stat: !tt.ptr<f32>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %scratch, %partial {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 2, 2>} : !tt.ptr<f16>, !tt.ptr<f32>
    tle.task_grid.create %stat {field_names = ["stat"], grid_name = "rms_stat", scope = "device", shape = array<i64: 2>} : !tt.ptr<f32>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "rms_out", scope = "device", shape = array<i64: 2, 2>} : !tt.ptr<f16>
    tle.task.declare {domain_shape = array<i64: 2, 2>, reads = [], task_name = "linear_tile", writes = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0, d1)>}]}
    tle.task.declare {domain_shape = array<i64: 2>, reads = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>, wildcard_dims = array<i64: 1>}], task_name = "rms_reduce", writes = [{grid = "rms_stat", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {domain_shape = array<i64: 2, 2>, reads = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0, d1)>}, {grid = "rms_stat", map = affine_map<(d0, d1) -> (d0)>}], task_name = "rms_apply", writes = [{grid = "rms_out", map = affine_map<(d0, d1) -> (d0, d1)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_duplicate_producer(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "g", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task.declare {domain_shape = array<i64: 1>, reads = [], task_name = "producer_a", writes = [{grid = "g", map = affine_map<(d0) -> (d0)>}]}
    // expected-error @+1 {{task graph has multiple producers for g[0]}}
    tle.task.declare {domain_shape = array<i64: 1>, reads = [], task_name = "producer_b", writes = [{grid = "g", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_unproduced_read(%scratch: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "input_grid", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out_grid", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    // expected-error @+1 {{task graph read requires unproduced tile input_grid[0]}}
    tle.task.declare {domain_shape = array<i64: 1>, reads = [{grid = "input_grid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out_grid", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // expected-error @+1 {{task graph contains a dependency cycle}}
  tt.func @reject_cycle(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>) {
    tle.task_grid.create %a {field_names = ["a"], grid_name = "ga", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task_grid.create %b {field_names = ["b"], grid_name = "gb", scope = "device", shape = array<i64: 1>} : !tt.ptr<f16>
    tle.task.declare {domain_shape = array<i64: 1>, reads = [{grid = "gb", map = affine_map<(d0) -> (d0)>}], task_name = "task_a", writes = [{grid = "ga", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {domain_shape = array<i64: 1>, reads = [{grid = "ga", map = affine_map<(d0) -> (d0)>}], task_name = "task_b", writes = [{grid = "gb", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_instance_overflow(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "g", scope = "device", shape = array<i64: 50000, 50000>} : !tt.ptr<f16>
    // expected-error @+1 {{task instance cardinality exceeds signed 32-bit range}}
    tle.task.declare {domain_shape = array<i64: 50000, 50000>, reads = [], task_name = "too_many", writes = [{grid = "g", map = affine_map<(d0, d1) -> (d0, d1)>}]}
    tt.return
  }
}
