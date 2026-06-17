// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @valid_task_grid_ops
  tt.func @valid_task_grid_ops(%scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>) {
    // CHECK: tle.task_grid.create
    // CHECK-SAME: grid_name = "linear_to_rms"
    // CHECK-SAME: shape = array<i64: 16, 2>
    tle.task_grid.create %scratch, %partial {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>, !tt.ptr<f32>
    // CHECK: "tle.task_grid.tile_id"
    // CHECK-SAME: grid_name = "linear_to_rms"
    %row, %h_tile = "tle.task_grid.tile_id"(%scratch, %partial) {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : (!tt.ptr<f16>, !tt.ptr<f32>) -> (i32, i32)
    // CHECK: tle.task_grid.commit
    tle.task_grid.commit %scratch, %partial {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>, !tt.ptr<f32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_invalid_scope(%scratch: !tt.ptr<f16>) {
    // expected-error @+1 {{supports only scope = "device" or "cta"}}
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "cluster"} : !tt.ptr<f16>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_duplicate_field_name(%scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>) {
    // expected-error @+1 {{expects unique task_grid field names}}
    tle.task_grid.create %scratch, %partial {field_names = ["scratch", "scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>, !tt.ptr<f32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_tile_id_result_count(%scratch: !tt.ptr<f16>) {
    // expected-error @+1 {{expects tile_id result count to match shape rank}}
    %row = "tle.task_grid.tile_id"(%scratch) {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : (!tt.ptr<f16>) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_commit_field_name_count(%scratch: !tt.ptr<f16>) {
    // expected-error @+1 {{expects field_names size to match field operands}}
    tle.task_grid.commit %scratch {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_reserved_field_name(%scratch: !tt.ptr<f16>) {
    // expected-error @+1 {{expects valid public task_grid field names}}
    tle.task_grid.create %scratch {field_names = ["commit"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    tt.return
  }
}
