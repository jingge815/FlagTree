// RUN: triton-opt %s -split-input-file -pim-tile-to-budget | FileCheck %s

// 没有 tt.dot 的循环，工作集超预算时步长被改写。
// 每步 4 字节、跨度 16、预算 32，步长从 1 改到 8。
// CHECK: "pim.tile-m" = 8
// CHECK-LABEL: @rewrite_oversized
// CHECK: step %c8
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @rewrite_oversized(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}

// -----

// iter_args 每轮都在累积时不改步长：结果依赖迭代次数，只改步长会把
// 16 次累加变成 2 次。
// CHECK-LABEL: @keep_accumulating_iter_args
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @keep_accumulating_iter_args(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      %s = arith.addf %acc, %x : tensor<16xf32>
      scf.yield %s : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}

// -----

// 步长已经大于等于跨度的零次迭代循环不改写，也不能崩。
// CHECK-LABEL: @keep_zero_trip
// CHECK: step %c32
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @keep_zero_trip(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %r = scf.for %i = %c0 to %c16 step %c32 iter_args(%acc = %x) -> tensor<16xf32> {
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}

// -----

// 循环体按归纳变量取址时，步长改大的同时循环体按新步长重建：
// 每轮覆盖的元素数从 1 变成 tile，结果与逐元素处理相同。
// CHECK-LABEL: @rebuild_body_with_step
// CHECK: step %c8
// CHECK: tensor<8x!tt.ptr<f32>>
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @rebuild_body_with_step(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      %v = tt.load %ia : tensor<1x!tt.ptr<f32>>
      %d = arith.addf %v, %v : tensor<1xf32>
      tt.store %oa, %d : tensor<1x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 预算内的循环不改。
// CHECK-LABEL: @keep_within_budget
// CHECK: step %c1
// CHECK-NOT: pim.tile-m
module attributes {"pim.wram-bytes" = 65536 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @keep_within_budget(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}

// -----

// 循环体用了归纳变量时不改步长，否则取值位置错位。
// CHECK-LABEL: @keep_induction_var
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @keep_induction_var(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      %idx = arith.index_cast %i : index to i32
      %v = tt.splat %idx : i32 -> tensor<16xi32>
      %f = arith.sitofp %v : tensor<16xi32> to tensor<16xf32>
      %s = arith.addf %acc, %f : tensor<16xf32>
      scf.yield %s : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}

// -----

// 偏移是二维张量时同样按新步长重建：沿第一维扩成一段范围，其余维度
// 保持不动。基址是 splat 出的单一值，每一行所有列读同一个地址，范围
// 不乘行距，覆盖的地址集合与原循环相同。每步加载加存储共 256 字节、
// 跨度 16、预算 512，步长从 1 改到 2。
// CHECK-LABEL: @rebuild_body_rank2
// CHECK: step %c2
// CHECK: tensor<2x32x!tt.ptr<f32>>
module attributes {"pim.wram-bytes" = 512 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @rebuild_body_rank2(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x32xi32>
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

// 二维偏移的原语义是 splat：每一行的所有列都等于该轮的基址。
// 重建后沿第一维扩成一段范围，但范围不乘行距，基址也不乘行距，
// 这样第 0 轮两行覆盖的地址集合仍是 0 到 1，与原循环相同。
// 乘了行距会让第 0 轮读到 0 到 7，原循环根本不访问那些地址。
// CHECK-LABEL: @rebuild_body_rank2_offsets
// CHECK-NOT: arith.muli
// CHECK: tt.make_range {end = 2 : i32, start = 0 : i32}
// CHECK-NOT: arith.muli
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @rebuild_body_rank2_offsets(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x4xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %v = tt.load %ia : tensor<1x4x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 一次迭代加载两块、存回一块，三块都要进 WRAM。每块 32 字节，合计 96，
// 预算 96 只容得下一次迭代，步长保持 1。只按最大的一块算会当成 32 字节，
// 把两轮并成一步，重建后一次搬 64 字节的三块，合计超出预算。
// CHECK-LABEL: @keep_when_transfers_sum_over_budget
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 96 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @keep_when_transfers_sum_over_budget(%in: !tt.ptr<f32>, %w: !tt.ptr<f32>,
                                               %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x8xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %wptr = tt.splat %w : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %wa = tt.addptr %wptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %v = tt.load %ia : tensor<1x8x!tt.ptr<f32>>
      %u = tt.load %wa : tensor<1x8x!tt.ptr<f32>>
      %d = arith.addf %v, %u : tensor<1x8xf32>
      tt.store %oa, %d : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// iter_args 原样 yield 但循环体用了归纳变量。结果不随步长变化，
// 但重建出的循环不带 iter_args，用它替换原循环的结果会触发断言。
// 这种循环不重建，步长保持不变。
// CHECK-LABEL: @keep_iter_args_when_induction_var_is_used
// CHECK: step %c1
// CHECK: iter_args
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @keep_iter_args_when_induction_var_is_used(%x: tensor<16xf32>,
                                                     %in: !tt.ptr<f32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      %v = tt.load %ia : tensor<1x!tt.ptr<f32>>
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}

// -----

// 循环的「原样」是整段循环：16 轮共 512 字节，超过预算 64，所以要切；
// 选 tile 时按单块工作集，每轮 32 字节，两轮一块 64 字节正好在预算内，
// 块数从 16 降到 8。这与 tt.dot 路径同一个评估模型：整算子装得下不改，
// 装不下就选预算内块数最少的切法。
// CHECK-LABEL: @per_iteration_fits_whole_loop_does_not
// CHECK: step %c2
// CHECK: tensor<2x4x!tt.ptr<f32>>
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @per_iteration_fits_whole_loop_does_not(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x4xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %v = tt.load %ia : tensor<1x4x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x4x!tt.ptr<f32>>
    }
    tt.return
  }
}
