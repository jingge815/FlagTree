// RUN: triton-opt %s --triton-tle-verify-task-graph -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @linear_tile_body(%row: i32, %h_tile: i32, %scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>, %stat: !tt.ptr<f32>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func private @rms_reduce_body(%row: i32, %scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>, %stat: !tt.ptr<f32>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func private @rms_apply_body(%row: i32, %h_tile: i32, %scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>, %stat: !tt.ptr<f32>, %out: !tt.ptr<f16>) {
    tt.return
  }

  // CHECK-LABEL: tt.func @valid_linear_rms_apply_graph
  tt.func @valid_linear_rms_apply_graph(%scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>, %stat: !tt.ptr<f32>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %scratch, %partial {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>, !tt.ptr<f32>
    tle.task_grid.create %stat {field_names = ["stat"], grid_name = "rms_stat", scope = "device", shape = array<i64: 16>} : !tt.ptr<f32>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "rms_out", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>

    // CHECK: tle.task.declare
    // CHECK-SAME: callee = @linear_tile_body
    // CHECK-SAME: task_name = "linear_tile"
    tle.task.declare {callee = @linear_tile_body, domain_shape = array<i64: 16, 2>, reads = [], task_name = "linear_tile", writes = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0, d1)>}]}

    // CHECK: tle.task.declare
    // CHECK-SAME: callee = @rms_reduce_body
    // CHECK-SAME: task_name = "rms_reduce"
    tle.task.declare {callee = @rms_reduce_body, domain_shape = array<i64: 16>, reads = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>, wildcard_dims = array<i64: 1>}], task_name = "rms_reduce", writes = [{grid = "rms_stat", map = affine_map<(d0) -> (d0)>}]}

    // CHECK: tle.task.declare
    // CHECK-SAME: callee = @rms_apply_body
    // CHECK-SAME: task_name = "rms_apply"
    tle.task.declare {callee = @rms_apply_body, domain_shape = array<i64: 16, 2>, reads = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0, d1)>}, {grid = "rms_stat", map = affine_map<(d0, d1) -> (d0)>}], task_name = "rms_apply", writes = [{grid = "rms_out", map = affine_map<(d0, d1) -> (d0, d1)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_unknown_grid(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    // expected-error @+1 {{task write map references unknown task_grid missing_grid}}
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "missing_grid", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_rank_mismatch(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>
    // expected-error @+1 {{task write map target rank does not match referenced task_grid rank}}
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_wildcard_out_of_rank(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>
    // expected-error @+1 {{task read wildcard dimension is outside referenced grid rank}}
    tle.task.declare {domain_shape = array<i64: 16, 2>, reads = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0)>, wildcard_dims = array<i64: 2>}], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0, d1)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_unshaped_grid(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    // expected-error @+1 {{references task_grid "linear_to_rms" without static shape}}
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_duplicate_task_name(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "dup_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    // expected-error @+1 {{duplicates task name in task graph metadata}}
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "dup_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_duplicate_grid_name(%scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    // expected-error @+1 {{duplicates task_grid name in task graph metadata}}
    tle.task_grid.create %partial {field_names = ["partial"], grid_name = "linear_to_rms", scope = "cta", shape = array<i64: 16>} : !tt.ptr<f32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_unknown_callee(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    // expected-error @+1 {{task callee references unknown tt.func @missing_body}}
    tle.task.declare {callee = @missing_body, domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_self_callee(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    // expected-error @+1 {{task callee must not reference the enclosing graph function}}
    tle.task.declare {callee = @reject_self_callee, domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @public_body(%row: i32, %scratch: !tt.ptr<f16>) {
    tt.return
  }

  tt.func @reject_public_callee(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    // expected-error @+1 {{task callee @public_body must be a private device function}}
    tle.task.declare {callee = @public_body, domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @bad_body(%row: i32) {
    tt.return
  }

  tt.func @reject_bad_callee_abi(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    // expected-error @+1 {{input count must be task domain rank plus enclosing graph function inputs}}
    tle.task.declare {callee = @bad_body, domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}
