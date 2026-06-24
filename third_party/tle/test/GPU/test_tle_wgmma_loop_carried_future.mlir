// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @loop_carried_future_materializes_after_younger_dot
  tt.func @loop_carried_future_materializes_after_younger_dot(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%score = %zero, %acc = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[YOUNGER:.+]] = ttng.warp_group_dot
      %younger = ttng.warp_group_dot %a0, %b0, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]] = ttng.warp_group_dot_wait
      // CHECK-SAME: {pendings = 1 : i32}
      // CHECK-NEXT: %[[READY:.+]] = arith.addf %[[WAIT]], %[[WAIT]]
      %score_ready = arith.addf %score, %score : tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[NEXT:.+]] = ttng.warp_group_dot {{.*}}, %[[READY]]
      %next = ttng.warp_group_dot %a1, %b1, %score_ready {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[DEPTH:.+]]:{{.*}} = ttng.warp_group_dot_wait
      // CHECK-SAME: {pendings = 2 : i32}
      // CHECK-NEXT: scf.yield %[[DEPTH]]#{{[0-9]+}}, %[[DEPTH]]#{{[0-9]+}}
      scf.yield %next, %younger : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @loop_carried_accumulator_chain_stays_unmaterialized
  tt.func @loop_carried_accumulator_chain_stays_unmaterialized(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[RES:.+]] = scf.for
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[DEPTH:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
      // CHECK-SAME: {pendings = 1 : i32}
      // CHECK-NEXT: scf.yield %[[DEPTH]]#0
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    // CHECK-NEXT: }
    // CHECK-NEXT: %[[FINAL:.+]] = ttng.warp_group_dot_wait %[[RES]]
    // CHECK-SAME: {pendings = 0 : i32}
    tt.store %out, %res : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
