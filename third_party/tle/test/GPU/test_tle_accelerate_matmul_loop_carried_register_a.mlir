// RUN: triton-opt %s --tritongpu-accelerate-matmul | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @loop_carried_computed_a_stays_register_operand
  tt.func public @loop_carried_computed_a_stays_register_operand(
      %v: tensor<128x128xbf16, #blocked>) -> tensor<64x128xf32, #blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %one = arith.constant dense<1.000000e+00> : tensor<64x128xf32, #blocked>
    %p_init = arith.truncf %one : tensor<64x128xf32, #blocked> to tensor<64x128xbf16, #blocked>

    %res:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%p_arg = %p_init, %acc_arg = %zero)
        -> (tensor<64x128xbf16, #blocked>, tensor<64x128xf32, #blocked>) {
      // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<64x128xbf16
      // CHECK: ttng.warp_group_dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x128xbf16, #ttg.dot_op
      %a = ttg.convert_layout %p_arg : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #dotOp0>
      %b = ttg.convert_layout %v : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #dotOp1>
      %out = tt.dot %a, %b, %acc_arg : tensor<64x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<64x128xf32, #blocked>
      scf.yield %p_init, %out : tensor<64x128xbf16, #blocked>, tensor<64x128xf32, #blocked>
    }
    tt.return %res#1 : tensor<64x128xf32, #blocked>
  }
}
