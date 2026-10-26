// RUN: triton-opt %s -pim-dce | FileCheck %s

// barrier 没有结果也没有用户，但它是任务线程间的同步点，不能当死代码删。
// CHECK-LABEL: @keep_barrier
// CHECK: pim.barrier
module {
  tt.func @keep_barrier(%t: tensor<4xf16>) -> tensor<4xf16> {
    %buf = pim.wram_alloc : !pim.memdesc<4xf16, #pim.wram>
    pim.wram_store %t, %buf : tensor<4xf16> -> !pim.memdesc<4xf16, #pim.wram>
    pim.barrier
    %r = pim.wram_load %buf : !pim.memdesc<4xf16, #pim.wram> -> tensor<4xf16>
    tt.return %r : tensor<4xf16>
  }
}
