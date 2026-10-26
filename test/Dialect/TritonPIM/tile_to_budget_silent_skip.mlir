// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// 超预算却切不了的循环必须给诊断，否则与预算内不切的产物无法区分。

// 只跑一轮：步长等于跨度，tensor<16xf32> 共 64 字节，预算 32。
// DIAG: remark: pim-tile-to-budget left this loop untiled: loop runs a single iteration
// CHECK-LABEL: @single_iteration
// CHECK: step %c16
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @single_iteration(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %r = scf.for %i = %c0 to %c16 step %c16 iter_args(%acc = %x) -> tensor<16xf32> {
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}

// -----

// 跨度 7 不是 2 的幂且不读归纳变量：只改步长会对不齐尾块。
// tensor<8xf32> 32 字节摊到 7 轮，合计 28 字节，预算 8。
// DIAG: remark: pim-tile-to-budget left this loop untiled: loop extent is not a power of two
// CHECK-LABEL: @odd_extent_unused_iv
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 8 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @odd_extent_unused_iv(%x: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c7 step %c1 iter_args(%acc = %x) -> tensor<8xf32> {
      scf.yield %acc : tensor<8xf32>
    }
    tt.return %r : tensor<8xf32>
  }
}

// -----

// 有副作用却不读归纳变量：每轮写同一块，只改步长会少写几轮。
// 每轮存储 32 字节、16 轮，预算 64。
// DIAG: remark: pim-tile-to-budget left this loop untiled: loop has side effects but does not read its induction variable
// CHECK-LABEL: @effect_unused_iv
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @effect_unused_iv(%out: !tt.ptr<f32>, %v: tensor<8xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %optr = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %oa = tt.addptr %optr, %offs : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      tt.store %oa, %v : tensor<8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 预算内的循环保持静默，不发诊断。
// DIAG-NOT: @within_budget
// CHECK-LABEL: @within_budget
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 4096 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @within_budget(%x: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c7 step %c1 iter_args(%acc = %x) -> tensor<8xf32> {
      scf.yield %acc : tensor<8xf32>
    }
    tt.return %r : tensor<8xf32>
  }
}

// -----

// 每轮加载 f32 16 字节、f16 8 字节，存储 16 字节，共 40 字节，跨度 16，整段 640 超过预算 64，但单轮放得下；
// 两轮一块要 80 字节，没有比 1 更大的 tile。诊断要说清是没有更大的 tile，
// 不能说单轮超预算。
// DIAG: left this loop untiled: no tile larger than one iteration fits the WRAM budget
// DIAG-NOT: one iteration exceeds the WRAM budget
// CHECK-LABEL: @no_larger_tile
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @no_larger_tile(%in: !tt.ptr<f32>, %h: !tt.ptr<f16>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x4xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %hptr = tt.splat %h : !tt.ptr<f16> -> tensor<1x4x!tt.ptr<f16>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %ha = tt.addptr %hptr, %offs : tensor<1x4x!tt.ptr<f16>>, tensor<1x4xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %v = tt.load %ia : tensor<1x4x!tt.ptr<f32>>
      %u = tt.load %ha : tensor<1x4x!tt.ptr<f16>>
      %w = arith.extf %u : tensor<1x4xf16> to tensor<1x4xf32>
      %d = arith.addf %v, %w : tensor<1x4xf32>
      tt.store %oa, %d : tensor<1x4x!tt.ptr<f32>>
    }
    tt.return
  }
}
