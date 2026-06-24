// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 -reconcile-unrealized-casts | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier_sm90_yields
  tt.func @wait_barrier_sm90_yields(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %phase: i32) {
    // CHECK: mbarrier.try_wait.parity.shared.b64 complete, [$0], $1, 0x989680;
    // CHECK-SAME: @!complete bra.uni waitLoop;
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: predicated_wait_barrier_sm90_yields
  tt.func @predicated_wait_barrier_sm90_yields(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %phase: i32, %pred: i1) {
    // CHECK: @!$2 bra.uni skipWait;
    // CHECK: mbarrier.try_wait.parity.shared.b64 complete, [$0], $1, 0x989680;
    // CHECK-SAME: @!complete bra.uni waitLoop;
    // CHECK-SAME: skipWait:
    ttng.wait_barrier %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}
