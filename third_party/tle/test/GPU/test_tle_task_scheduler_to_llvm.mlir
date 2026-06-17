// RUN: triton-opt %s --triton-tle-analyze-task-graph --triton-tle-materialize-task-scheduler --triton-tle-materialize-task-runtime-state --convert-triton-gpu-to-llvm=compute-capability=90 -reconcile-unrealized-casts | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @producer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  tt.func private @consumer_body(%tile: i32, %mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tt.return
  }

  // CHECK: tle.requires_cooperative_grid = 1 : i32
  // CHECK: tle.task_scheduler_scratch_offset = 0 : i32
  // CHECK: tle.task_scheduler_shared_offset = 0 : i32
  // CHECK: ttg.global_scratch_memory_size = 52 : i32
  // CHECK: ttg.shared = 4 : i32
  // CHECK-LABEL: llvm.func internal @producer_body
  // CHECK-LABEL: llvm.func internal @consumer_body
  // CHECK-LABEL: llvm.func @task_scheduler_to_llvm
  // CHECK: llvm.atomicrmw xchg
  // CHECK-SAME: syncscope("device") release
  // CHECK: llvm.atomicrmw xchg
  // CHECK-SAME: syncscope("device") acq_rel
  // CHECK: nvvm.barrier0
  // CHECK: llvm.call @producer_body
  // CHECK-NEXT: llvm.inline_asm
  // CHECK-SAME: "membar.gl;"
  // CHECK-NOT: llvm.fence
  // CHECK: llvm.call @consumer_body
  // CHECK-NEXT: llvm.inline_asm
  // CHECK-SAME: "membar.gl;"
  // CHECK-NOT: llvm.fence
  // CHECK: llvm.atomicrmw add
  // CHECK-SAME: syncscope("device") acq_rel
  // CHECK-NOT: tle.task_graph.scheduler
  // CHECK-NOT: tle.task_graph.runtime_state
  tt.func @task_scheduler_to_llvm(%mid: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    tle.task_grid.create %mid {field_names = ["mid"], grid_name = "mid", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    tle.task_grid.create %out {field_names = ["out"], grid_name = "out", scope = "device", shape = array<i64: 2>} : !tt.ptr<f16>
    tle.task.declare {callee = @producer_body, domain_shape = array<i64: 2>, reads = [], task_name = "producer", writes = [{grid = "mid", map = affine_map<(d0) -> (d0)>}]}
    tle.task.declare {callee = @consumer_body, domain_shape = array<i64: 2>, reads = [{grid = "mid", map = affine_map<(d0) -> (d0)>}], task_name = "consumer", writes = [{grid = "out", map = affine_map<(d0) -> (d0)>}]}
    tt.return
  }
}
