// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// 跨度 7 不被任何大于 1 的 2 的幂整除。一次迭代加载加存储共 64 字节、
// 预算 256，本可以取 tile 4，但最后一轮会从第 4 行读到第 7 行，
// 合法行只有 0 到 6。回退到 2 也不整除时保持原循环并给出诊断。
// DIAG: left this loop untiled: no searched tile divides the loop extent
// CHECK-LABEL: @extent_not_divisible
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 256 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @extent_not_divisible(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c1 = arith.constant 1 : index
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    scf.for %i = %c0 to %c7 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %wide = tt.broadcast %row2 : tensor<1x1xi32> -> tensor<1x8xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x8xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %v = tt.load %ia : tensor<1x8x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      tt.store %oa, %v : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 循环带掩码，掩码长度与指针相同。重建把指针拉长到 tile 而掩码保持原长时，
// tt.load 会因两侧形状不同而验证失败。认不出可随之放大的掩码时保持原循环。
// 跨度取 8，能被 tile 整除，这样拦住它的只有掩码，不是整除检查。
// DIAG: left this loop untiled: masked load or store
// CHECK-LABEL: @masked_loop
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 256 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @masked_loop(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>,
                       %mask: tensor<1x8xi1>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    scf.for %i = %c0 to %c8 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %wide = tt.broadcast %row2 : tensor<1x1xi32> -> tensor<1x8xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x8xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %v = tt.load %ia, %mask : tensor<1x8x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      tt.store %oa, %v, %mask : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 原步长是 2，访问第 0、2、4、6 行。重建把范围拉成连续的 tile 个位置后，
// 访问的是第 0、1 行和第 4、5 行，原迭代集合被换掉。步长不是 1 时不重建。
// DIAG: left this loop untiled: loop step is not 1
// CHECK-LABEL: @step_two
// CHECK: step %c2
module attributes {"pim.wram-bytes" = 128 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @step_two(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    scf.for %i = %c0 to %c8 step %c2 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %wide = tt.broadcast %row2 : tensor<1x1xi32> -> tensor<1x8xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x8xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %v = tt.load %ia : tensor<1x8x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      tt.store %oa, %v : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 跨度 30 不被预算内最大的 tile 4 整除，但 tile 2 既整除又在预算内。
// 一次迭代加载加存储共 64 字节、预算 256。要向下回退到能整除的 tile，
// 不能直接放弃。
// CHECK: "pim.tile-m" = 2
// CHECK-LABEL: @fallback_divisible_tile
// CHECK: step %c2
// CHECK: tt.load %{{.*}} : tensor<2x8x!tt.ptr<f32>>
module attributes {"pim.wram-bytes" = 256 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @fallback_divisible_tile(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c30 = arith.constant 30 : index
    %c1 = arith.constant 1 : index
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    scf.for %i = %c0 to %c30 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %wide = tt.broadcast %row2 : tensor<1x1xi32> -> tensor<1x8xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x8xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %v = tt.load %ia : tensor<1x8x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      tt.store %oa, %v : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}
