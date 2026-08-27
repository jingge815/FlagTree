// RUN: not triton-opt %s -pim-lower-to-emitc 2>&1 | FileCheck %s

// num-tasklets < 1 must fail outright rather than silently producing wrong
// code (unlike num-tasklets == 1, the degenerate single-block case covered
// in lower_to_emitc.mlir, which is not an error).

// CHECK: error: pim-lower-to-emitc requires pim.num-tasklets >= 1, got 0
module attributes {"pim.num-tasklets" = 0 : i32} {
  tt.func public @noop_kernel(%o_ptr: !tt.ptr<f32>) {
    tt.return
  }
}
