// RUN: triton-opt %s -pim-tile-to-budget | FileCheck %s

// i32 循环不读归纳变量、循环体无副作用，走只改步长的分支。
// 整段 64 字节超预算 32，单轮 4 字节，最大 tile 为 8；新步长必须仍是 i32。
// CHECK-LABEL: @i32_unused
// CHECK: %[[S:.*]] = arith.constant 8 : i32
// CHECK: scf.for {{.*}} step %[[S]] {{.*}} : i32
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @i32_unused(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %c1 = arith.constant 1 : i32
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> : i32 {
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}
