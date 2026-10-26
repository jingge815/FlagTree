// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// CHECK-DAG: "pim.tile-m" = 2
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// 真实 kernel 的寻址是「行号 * 行距 + 列偏移」：splat(归纳变量) 之后
// 还要经过 muli、expand_dims、broadcast、addi 才到 addptr。
// 一次迭代加载 128 字节再存 128 字节，合计 256，跨度 16、预算 512。
// 认得出这种形态就必须把步长改大，而不是保持 %c1 静默跳过。
// CHECK-LABEL: @row_major_addressing
// CHECK: step %c2
module attributes {"pim.wram-bytes" = 512 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @row_major_addressing(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %stride = arith.constant dense<32> : tensor<1x1xi32>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %scaled = arith.muli %row2, %stride : tensor<1x1xi32>
      %wide = tt.broadcast %scaled : tensor<1x1xi32> -> tensor<1x32xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x32xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x32x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x32x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x32x!tt.ptr<f32>>, tensor<1x32xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x32x!tt.ptr<f32>>, tensor<1x32xi32>
      %v = tt.load %ia : tensor<1x32x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x32x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 跨度 24 不是 2 的幂，寻址形态符合重建要求。一次迭代加载加存储共 64 字节、
// 预算 128，只容得下两轮，所以步长取 2，尾块用不满的 tile 收尾。
// 非 2 的幂不能成为静默跳过的理由。
// CHECK-LABEL: @odd_extent_row_major
// CHECK: step %c2
module attributes {"pim.wram-bytes" = 128 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @odd_extent_row_major(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c1 = arith.constant 1 : index
    %stride = arith.constant dense<8> : tensor<1x1xi32>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    scf.for %i = %c0 to %c24 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %scaled = arith.muli %row2, %stride : tensor<1x1xi32>
      %wide = tt.broadcast %scaled : tensor<1x1xi32> -> tensor<1x8xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x8xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %v = tt.load %ia : tensor<1x8x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 寻址既不是 splat 直连 addptr，也不是行主序：splat 的结果先取了余数。
// 一次迭代加载加存储共 8 字节、跨度 16、预算 4，超预算。认不出时必须给出
// 诊断，否则产物与预算内不切的循环无法区分。
// DIAG: left this loop untiled: addressing is not splat or row-major
// CHECK-LABEL: @unrecognized_addressing_is_reported
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 16 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @unrecognized_addressing_is_reported(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %two = arith.constant dense<2> : tensor<1xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %offs = arith.remui %row, %two : tensor<1xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      %v = tt.load %ia : tensor<1x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x!tt.ptr<f32>>
    }
    tt.return
  }
}
