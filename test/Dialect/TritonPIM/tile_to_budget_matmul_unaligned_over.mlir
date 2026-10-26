// RUN: triton-opt %s -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG --allow-empty

// 1x5 乘 5x1 的 f16 矩阵乘整块 28 字节超预算 16。B 路没有 DMA，
// tile 不受对齐约束：[1,1,1] 工作集 2+2+8=12 字节，整除各维，必须选中。
// DIAG-NOT: untiled
// CHECK: "pim.tile-k" = 1
// CHECK-SAME: "pim.tile-m" = 1
// CHECK-SAME: "pim.tile-n" = 1
// CHECK-LABEL: @over
// CHECK: tile = array<i64: 1, 1, 1>
module attributes {"pim.wram-bytes" = 16 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @over(%a: tensor<1x5xf16>, %w: tensor<5x1xf16>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<1x5xf16>, tensor<5x1xf16> -> tensor<1x1xf16>
    tt.return
  }
}
