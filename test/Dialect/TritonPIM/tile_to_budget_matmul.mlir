// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG
// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -pim-lower-to-emitc | FileCheck %s --check-prefix=EMITC

// B 路整算子矩阵乘没有循环，按形状选 tile 写在算子上。16x128 乘 128x8、
// 组宽 64、预算 1024：元素最多且整除各维与组宽的是 M=4、N=8、K=64，
// 工作集 4*64 + 8*64 + 4*8*8 = 1024 字节。
// CHECK: "pim.tile-k" = 64 : i64, "pim.tile-m" = 4 : i64, "pim.tile-n" = 8 : i64
// CHECK-LABEL: @oversized
// CHECK: pim.matmul {{.*}}tile = array<i64: 4, 8, 64>
// 降到 C 时先分配两个 4x8 的 f32 累加块（各 128 字节），再按 tile 生成块循环：
// M 方向 16/4 = 4 块。
// EMITC-LABEL: @oversized
// EMITC: value = 128 : i64
// EMITC: "malloc"
// EMITC: value = 128 : i64
// EMITC: "malloc"
// EMITC: value = 4 : i32
// EMITC: emitc.for
module attributes {"pim.wram-bytes" = 1024 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @oversized(%a: tensor<16x128xi8>, %w: tensor<128x8xi8>, %s: tensor<2x8xf16>) {
    %y = pim.matmul %a, %w scales %s
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, groupDequantAccum = true, groupSize = 64>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4, groupSize = 64>,
        stationarity = #pim.stationarity<weight>}
       : tensor<16x128xi8>, tensor<128x8xi8> scales tensor<2x8xf16> -> tensor<16x8xi8>
    tt.return
  }
}

// -----

// 整算子已在预算内时不写 tile。
// CHECK-LABEL: @within_budget
// CHECK-NOT: tile = array
module attributes {"pim.wram-bytes" = 65536 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @within_budget(%a: tensor<4x8xf16>, %w: tensor<8x4xf16>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xf16>, tensor<8x4xf16> -> tensor<4x4xf16>
    tt.return
  }
}

// -----

// M=7、N=7 是质数，整除它们的 tile 只有 1 或整维；最小的 1x1x1 也要
// 2+2+8=12 字节，预算 8 放不下，切不了时保持原样并给出诊断。
// DIAG: left this matmul untiled: no tile divides M, N and the group within the WRAM budget
// CHECK-LABEL: @untileable
// CHECK-NOT: tile = array
module attributes {"pim.wram-bytes" = 8 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @untileable(%a: tensor<7x8xf16>, %w: tensor<8x7xf16>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<7x8xf16>, tensor<8x7xf16> -> tensor<7x7xf16>
    tt.return
  }
}

// -----

// 同一模块两个矩阵乘选出的 tile 不同：m1 是 2x2x8，m2 是 1x2x16。两个都
// 超预算，各自写 tile 照常切开；模块头只有一组属性，描述不了两个，不写。
// DIAG: pim-tile-to-budget wrote no module tile attributes: not every matmul in the module uses the same tile
// CHECK-NOT: pim.tile-m
// CHECK-LABEL: @m1
// CHECK: tile = array<i64: 2, 2, 8>
// CHECK-LABEL: @m2
// CHECK: tile = array<i64: 1, 2, 16>
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @m1(%a: tensor<4x8xi8>, %w: tensor<8x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
  tt.func @m2(%a: tensor<4x16xi8>, %w: tensor<16x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x16xi8>, tensor<16x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}
