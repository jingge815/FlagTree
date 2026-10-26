// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// 外层循环超预算且读归纳变量，循环体里还有一层 scf.for。重建只改顶层算子的
// 结果类型，内层的加载仍是原长度，产物验证失败；外层删掉时内层也随之悬垂。
// 含嵌套循环的外层保持原样并给出诊断，产物要能通过 verifier。
// DIAG: left this loop untiled: loop contains a nested scf.for
// CHECK-LABEL: @nested_rows
// CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1
// CHECK: scf.for %{{.*}} = %c0 to %c2 step %c1
// CHECK: tt.load %{{.*}} : tensor<1x8x!tt.ptr<f32>>
module attributes {"pim.wram-bytes" = 128 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @nested_rows(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
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
      %optr = tt.splat %out : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
      %oa = tt.addptr %optr, %offs : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      scf.for %j = %c0 to %c2 step %c1 {
        %v = tt.load %ia : tensor<1x8x!tt.ptr<f32>>
        tt.store %oa, %v : tensor<1x8x!tt.ptr<f32>>
      }
    }
    tt.return
  }
}
