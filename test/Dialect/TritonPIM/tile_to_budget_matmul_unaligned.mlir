// RUN: triton-opt %s -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG --allow-empty

// 1x3 乘 3x1 的 f16 矩阵乘只有几十字节，远在预算内，只是不满足 DMA 对齐。
// 预算内的矩阵乘不切，也不给「超预算切不了」的诊断。
// DIAG-NOT: untiled
// CHECK-NOT: pim.tile-
// CHECK-LABEL: @tiny
// CHECK-NOT: tile = array
module attributes {"pim.wram-bytes" = 65536 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @tiny(%a: tensor<1x3xf16>, %w: tensor<3x1xf16>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<1x3xf16>, tensor<3x1xf16> -> tensor<1x1xf16>
    tt.return
  }
}

