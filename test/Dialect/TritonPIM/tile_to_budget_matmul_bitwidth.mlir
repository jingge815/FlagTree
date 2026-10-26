// RUN: triton-opt %s -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// i1 没有 C 实现，所以不和 EMITC 用例放在同一个文件里。
// i1 矩阵乘的元素宽度不是整字节，评估模型算不了工作集，保持原样；同模块的
// @sliceable 照常切成 2x2x8。模块里有一个矩阵乘没切，模块头不写，并给出诊断。
// DIAG: left this matmul untiled: element width is not a whole number of bytes
// DIAG: pim-tile-to-budget wrote no module tile attributes: not every matmul in the module uses the same tile
// CHECK-NOT: pim.tile-m
// CHECK-LABEL: @bit_matmul
// CHECK-NOT: tile = array
// CHECK-LABEL: @sliceable
// CHECK: tile = array<i64: 2, 2, 8>
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @bit_matmul(%a: tensor<4x8xi1>, %w: tensor<8x4xi1>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xi1>, tensor<8x4xi1> -> tensor<4x4xi1>
    tt.return
  }
  tt.func @sliceable(%a: tensor<4x8xi8>, %w: tensor<8x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}
