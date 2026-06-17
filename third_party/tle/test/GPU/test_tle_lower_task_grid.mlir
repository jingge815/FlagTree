// RUN: triton-opt %s --triton-tle-lower-task-grid -split-input-file -verify-diagnostics | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @lower_task_grid
  tt.func @lower_task_grid(%scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>) {
    tle.task_grid.create %scratch, %partial {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>, !tt.ptr<f32>
    %row, %h_tile = "tle.task_grid.tile_id"(%scratch, %partial) {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : (!tt.ptr<f16>, !tt.ptr<f32>) -> (i32, i32)
    tle.task_grid.commit %scratch, %partial {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>, !tt.ptr<f32>
    // CHECK-NOT: tle.task_grid
    tt.return
  }

  // CHECK-LABEL: tt.func @lower_task_grid_same_name_in_second_func
  tt.func @lower_task_grid_same_name_in_second_func(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : !tt.ptr<f16>
    %tile = "tle.task_grid.tile_id"(%scratch) {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16>} : (!tt.ptr<f16>) -> i32
    tle.task_grid.commit %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    // CHECK-NOT: tle.task_grid
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_missing_create(%scratch: !tt.ptr<f16>) {
    // expected-error @+1 {{requires a preceding matching task_grid.create}}
    tle.task_grid.commit %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_duplicate_create(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    // expected-error @+1 {{duplicates an existing task_grid.create}}
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_field_name_mismatch(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    // expected-error @+1 {{field_names do not match the preceding task_grid.create}}
    tle.task_grid.commit %scratch {field_names = ["partial"], grid_name = "linear_to_rms", scope = "device"} : !tt.ptr<f16>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_shape_mismatch(%scratch: !tt.ptr<f16>, %partial: !tt.ptr<f32>) {
    tle.task_grid.create %scratch, %partial {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>, !tt.ptr<f32>
    // expected-error @+1 {{shape does not match the preceding task_grid.create}}
    %row, %h_tile = "tle.task_grid.tile_id"(%scratch, %partial) {field_names = ["scratch", "partial"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 3>} : (!tt.ptr<f16>, !tt.ptr<f32>) -> (i32, i32)
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_used_tile_id(%scratch: !tt.ptr<f16>) {
    tle.task_grid.create %scratch {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : !tt.ptr<f16>
    // expected-error @+1 {{result use requires scheduler codegen}}
    %row, %h_tile = "tle.task_grid.tile_id"(%scratch) {field_names = ["scratch"], grid_name = "linear_to_rms", scope = "device", shape = array<i64: 16, 2>} : (!tt.ptr<f16>) -> (i32, i32)
    %sum = arith.addi %row, %h_tile : i32
    tt.return
  }
}
