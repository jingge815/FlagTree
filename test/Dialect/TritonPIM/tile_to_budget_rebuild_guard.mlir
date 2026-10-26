// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// 循环体对加载结果做归约。拉长首维会把「每轮归约一行」改成「每轮归约多行」，
// 语义改变，必须保持原循环并给诊断。
// DIAG: left this loop untiled: loop body has an op the rebuild cannot widen
// CHECK-NOT: pim.tile-m
// CHECK-LABEL: @reduce_in_body
// CHECK: step %c1
// CHECK-NOT: step %c2
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @reduce_in_body(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %stride = arith.constant dense<4> : tensor<1x1xi32>
    %cols = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %optr = tt.splat %out : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %oa = tt.addptr %optr, %cols : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    scf.for %i = %c0 to %c8 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %scaled = arith.muli %row2, %stride : tensor<1x1xi32>
      %wide = tt.broadcast %scaled : tensor<1x1xi32> -> tensor<1x4xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x4xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %v = tt.load %ia : tensor<1x4x!tt.ptr<f32>>
      %s = "tt.reduce"(%v) ({
      ^bb0(%a: f32, %b: f32):
        %t = arith.addf %a, %b : f32
        tt.reduce.return %t : f32
      }) {axis = 0 : i32} : (tensor<1x4xf32>) -> tensor<4xf32>
      tt.store %oa, %s : tensor<4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 两条加载链行距不同：x 行距 32、w 行距 64。重建只有一个行距，套用到两条链
// 会把 x 的地址改错，必须保持原循环并给诊断。
// DIAG: left this loop untiled: address chains use different row strides
// CHECK-LABEL: @two_strides
// CHECK: step %c1
// CHECK-NOT: step %c2
module attributes {"pim.wram-bytes" = 512 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @two_strides(%x: !tt.ptr<f32>, %w: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %sx = arith.constant dense<32> : tensor<1x1xi32>
    %sw = arith.constant dense<64> : tensor<1x1xi32>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %rx = arith.muli %row2, %sx : tensor<1x1xi32>
      %rw = arith.muli %row2, %sw : tensor<1x1xi32>
      %bx = tt.broadcast %rx : tensor<1x1xi32> -> tensor<1x8xi32>
      %bw = tt.broadcast %rw : tensor<1x1xi32> -> tensor<1x8xi32>
      %ox = arith.addi %bx, %cols2 : tensor<1x8xi32>
      %ow = arith.addi %bw, %cols2 : tensor<1x8xi32>
      %px = tt.splat %x : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %pw = tt.splat %w : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %po = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ax = tt.addptr %px, %ox : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %aw = tt.addptr %pw, %ow : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %ao = tt.addptr %po, %ox : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %vx = tt.load %ax : tensor<1x8x!tt.ptr<f32>>
      %vw = tt.load %aw : tensor<1x8x!tt.ptr<f32>>
      %s = arith.addf %vx, %vw : tensor<1x8xf32>
      tt.store %ao, %s : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 同一模块两条循环，按预算分别要切成 2 和 4。模块只有一个 pim.tile-m，
// 第二条与第一条 tile 不同就保持原样并给诊断，属性与第一条的步长一致。
// DIAG: left this loop untiled: module already tiles another loop with a different tile
// CHECK: "pim.tile-m" = 2
// CHECK-LABEL: @two_loops
// CHECK: step %c2
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 128 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @two_loops(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %s8 = arith.constant dense<8> : tensor<1x1xi32>
    %s4 = arith.constant dense<4> : tensor<1x1xi32>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %c4r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c4r2 = tt.expand_dims %c4r {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %scaled = arith.muli %row2, %s8 : tensor<1x1xi32>
      %wide = tt.broadcast %scaled : tensor<1x1xi32> -> tensor<1x8xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x8xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %v = tt.load %ia : tensor<1x8x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x8x!tt.ptr<f32>>
    }
    scf.for %j = %c0 to %c16 step %c1 {
      %base = arith.index_cast %j : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %scaled = arith.muli %row2, %s4 : tensor<1x1xi32>
      %wide = tt.broadcast %scaled : tensor<1x1xi32> -> tensor<1x4xi32>
      %offs = arith.addi %wide, %c4r2 : tensor<1x4xi32>
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

// 两条链行距乘积都是 32，但结构不同：x 是「行号乘 32 再加列偏移」，
// w 是「行号加列偏移后再乘 32」。重建只认先乘后加，w 会被多乘一次 32，
// 必须保持原循环并给诊断。
// DIAG: left this loop untiled: row stride is applied after the column offset
// CHECK-LABEL: @mul_after_add
// CHECK: step %c1
// CHECK-NOT: step %c2
module attributes {"pim.wram-bytes" = 512 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @mul_after_add(%x: !tt.ptr<f32>, %w: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %s1 = arith.constant dense<32> : tensor<1x1xi32>
    %s8 = arith.constant dense<32> : tensor<1x8xi32>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %rx = arith.muli %row2, %s1 : tensor<1x1xi32>
      %bx = tt.broadcast %rx : tensor<1x1xi32> -> tensor<1x8xi32>
      %ox = arith.addi %bx, %cols2 : tensor<1x8xi32>
      %bw = tt.broadcast %row2 : tensor<1x1xi32> -> tensor<1x8xi32>
      %sw = arith.addi %bw, %cols2 : tensor<1x8xi32>
      %ow = arith.muli %sw, %s8 : tensor<1x8xi32>
      %px = tt.splat %x : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %pw = tt.splat %w : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %po = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %ax = tt.addptr %px, %ox : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %aw = tt.addptr %pw, %ow : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %ao = tt.addptr %po, %ox : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %vx = tt.load %ax : tensor<1x8x!tt.ptr<f32>>
      %vw = tt.load %aw : tensor<1x8x!tt.ptr<f32>>
      %s = arith.addf %vx, %vw : tensor<1x8xf32>
      tt.store %ao, %s : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 循环外的不变 bias 参与逐元素运算。bias 不在归纳变量链上，不能把它的类型
// 改成 tile 长，要在新循环里广播过去，重建出的 IR 必须合法。
// 每轮加载与存储共 32 字节、跨度 16、预算 64，步长从 1 改到 2。
// CHECK-LABEL: @invariant_bias
// CHECK: step %c2
// CHECK: tt.broadcast %{{.*}} : tensor<1x4xf32> -> tensor<2x4xf32>
// CHECK: arith.addf %{{.*}} : tensor<2x4xf32>
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @invariant_bias(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %bias = arith.constant dense<1.0> : tensor<1x4xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x4xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %v = tt.load %ia : tensor<1x4x!tt.ptr<f32>>
      %d = arith.addf %v, %bias : tensor<1x4xf32>
      tt.store %oa, %d : tensor<1x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 循环内还有一条不走归纳变量的加载。它的地址每轮不变，但可能与循环内的
// 写有别名，重建不了，保持原循环并给诊断，triton-opt 不能因验证失败退出。
// DIAG: left this loop untiled: loop body has a tensor value outside the induction-variable chain
// CHECK-LABEL: @invariant_load
// CHECK: step %c1
// CHECK-NOT: step %c2
module attributes {"pim.wram-bytes" = 128 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @invariant_load(%in: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x4xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %bptr = tt.splat %b : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
      %v = tt.load %ia : tensor<1x4x!tt.ptr<f32>>
      %bv = tt.load %bptr : tensor<1x4x!tt.ptr<f32>>
      %d = arith.addf %v, %bv : tensor<1x4xf32>
      tt.store %oa, %d : tensor<1x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 矩阵乘先写了 pim.tile-m = 2，同模块的循环每轮 16 字节、跨度 16、预算 64，
// 自己会选 tile 4。tile-m 只有一个，循环保持原样并给诊断，不覆盖矩阵乘的值。
// DIAG: left this loop untiled: module already tiles another loop with a different tile
// CHECK: "pim.tile-m" = 2 : i64
// CHECK-LABEL: @mm_then_loop
// CHECK: tile = array<i64: 2, 2, 8>
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @mm_then_loop(%a: tensor<4x8xi8>, %w: tensor<8x4xi8>,
                        %in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x2xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi32>
      %v = tt.load %ia : tensor<1x2x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 偏移先是 1x4，再广播成 2x4：一轮里两行都指向第 i 行，逐行 bias 不同，
// 原循环第 i 行最后留下 bias 第二行。重建把首维改成 tile 会让两行指向
// 第 i 和 i+1 行，数值改变。每轮 64 字节、预算 128，tile 2 本可选中。首维不是 1 的偏移不重建，保持原循环并给诊断。
// DIAG: left this loop untiled: offset leading extent is not 1
// CHECK-LABEL: @broadcast_rows
// CHECK: step %c1
// CHECK-NOT: step %c2
module attributes {"pim.wram-bytes" = 128 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @broadcast_rows(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %bias = arith.constant dense<[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]> : tensor<2x4xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1x4xi32>
      %offs = tt.broadcast %row : tensor<1x4xi32> -> tensor<2x4xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %oa = tt.addptr %optr, %offs : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      %v = tt.load %ia : tensor<2x4x!tt.ptr<f32>>
      %d = arith.addf %v, %bias : tensor<2x4xf32>
      tt.store %oa, %d : tensor<2x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 同一模块两个矩阵乘选出同一个 tile 4x8x8，但 m1 是 i8、m2 是 f16，
// 工作集分别是 352 与 448 字节。模块头只有一个 pim.tile-wram-bytes，
// 描述不了两个，不写模块头；两个算子各自照常切开。
// DIAG: pim-tile-to-budget wrote no module tile attributes: not every matmul in the module uses the same tile
// CHECK-NOT: pim.tile-
// CHECK-LABEL: @same_tile_i8
// CHECK: tile = array<i64: 4, 8, 8>
// CHECK-LABEL: @same_tile_f16
// CHECK: tile = array<i64: 4, 8, 8>
module attributes {"pim.wram-bytes" = 512 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @same_tile_i8(%a: tensor<8x8xi8>, %w: tensor<8x8xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<8x8xi8>, tensor<8x8xi8> -> tensor<8x8xi8>
    tt.return
  }
  tt.func @same_tile_f16(%a: tensor<8x8xf16>, %w: tensor<8x8xf16>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<8x8xf16>, tensor<8x8xf16> -> tensor<8x8xf16>
    tt.return
  }
}

// -----

// 前一个矩阵乘 1x8 乘 8x1 共 24 字节，在预算内不切；后一个超预算切成 2x2x8。模块头一组属性描述
// 不了两个算子，不写模块头并给诊断；后一个算子上的 tile 照写，降级照用。
// DIAG: pim-tile-to-budget wrote no module tile attributes: not every matmul in the module uses the same tile
// DIAG-NOT: left this matmul untiled
// CHECK-NOT: pim.tile-m
// CHECK-LABEL: @small_first
// CHECK-NOT: tile = array
// CHECK-LABEL: @big_second
// CHECK: tile = array<i64: 2, 2, 8>
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @small_first(%a: tensor<1x8xi8>, %w: tensor<8x1xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<1x8xi8>, tensor<8x1xi8> -> tensor<1x1xi8>
    tt.return
  }
  tt.func @big_second(%a: tensor<4x8xi8>, %w: tensor<8x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// 两个矩阵乘 tile 不同（2x2x8 与 1x2x16），模块头不写。同模块还有一个循环，
// 跨度 4、每轮 8 字节，整段 32 字节在预算 64 内。循环保持原样，模块头也
// 不能因为循环多出一个 pim.tile-m。
// CHECK-NOT: pim.tile-
// CHECK-LABEL: @mixed_mm1
// CHECK-LABEL: @mixed_loop
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @mixed_mm1(%a: tensor<4x8xi8>, %w: tensor<8x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
  tt.func @mixed_mm2(%a: tensor<4x16xi8>, %w: tensor<16x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x16xi8>, tensor<16x4xi8> -> tensor<4x4xi8>
    tt.return
  }
  tt.func @mixed_loop(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x1xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
      %v = tt.load %ia : tensor<1x1x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x1x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// 同上两个 tile 不同的矩阵乘，循环跨度 16、每轮 16 字节、预算 64，自己会选
// tile 4。各矩阵乘按算子 tile 切开却没写模块头，循环也不切，诊断写明原因，
// 不能说成另一个循环。
// DIAG: left this loop untiled: module matmuls carry their own tiles
// CHECK-NOT: pim.tile-
// CHECK-LABEL: @blocked_loop
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 64 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @blocked_mm1(%a: tensor<4x8xi8>, %w: tensor<8x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
  tt.func @blocked_mm2(%a: tensor<4x16xi8>, %w: tensor<16x4xi8>) {
    %y = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x16xi8>, tensor<16x4xi8> -> tensor<4x4xi8>
    tt.return
  }
  tt.func @blocked_loop(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %base = arith.index_cast %i : index to i32
      %offs = tt.splat %base : i32 -> tensor<1x2xi32>
      %iptr = tt.splat %in : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
      %ia = tt.addptr %iptr, %offs : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi32>
      %v = tt.load %ia : tensor<1x2x!tt.ptr<f32>>
      tt.store %oa, %v : tensor<1x2x!tt.ptr<f32>>
    }
    tt.return
  }
}
