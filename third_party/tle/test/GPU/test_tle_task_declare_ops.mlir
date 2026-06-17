// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @linear_tile_body(%row: i32, %h_tile: i32) {
    tt.return
  }

  // CHECK-LABEL: tt.func @valid_task_declare
  tt.func @valid_task_declare() {
    // CHECK: tle.task.declare
    // CHECK-SAME: callee = @linear_tile_body
    // CHECK-SAME: task_name = "linear_tile"
    tle.task.declare {callee = @linear_tile_body, domain_shape = array<i64: 16, 2>, reads = [], task_name = "linear_tile", writes = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0, d1)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_empty_domain() {
    // expected-error @+1 {{expects non-empty task domain_shape}}
    tle.task.declare {domain_shape = array<i64>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_symbolic_map() {
    // expected-error @+1 {{expects task write map to have no symbols}}
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0)[s0] -> (d0)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_non_projection_map() {
    // expected-error @+1 {{expects task write map results to be dimension projections}}
    tle.task.declare {domain_shape = array<i64: 16, 2>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0, d1) -> (d0 + d1)>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_wildcard_write() {
    // expected-error @+1 {{task writes must not use wildcard_dims}}
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = [{grid = "linear_to_rms", map = affine_map<(d0) -> (d0)>, wildcard_dims = array<i64: 1>}]}
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_empty_writes() {
    // expected-error @+1 {{expects task writes to contain at least one map}}
    tle.task.declare {domain_shape = array<i64: 16>, reads = [], task_name = "bad_task", writes = []}
    tt.return
  }
}
