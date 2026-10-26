// RUN: triton-opt %s -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// 行主序寻址，一次迭代加载 1x128 的 f16 再存回，合计 512 字节。
// 预算只有 256，单轮已经超预算，步长再小也压不进去。
// 这种循环切不了，必须给出诊断，否则产物与预算内不切的循环无法区分。
// DIAG: left this loop untiled: one iteration exceeds the WRAM budget
// CHECK-LABEL: @row_exceeds_budget
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 256 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @row_exceeds_budget(%in: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %stride = arith.constant dense<128> : tensor<1x1xi32>
    %cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    scf.for %i = %c0 to %c64 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %scaled = arith.muli %row2, %stride : tensor<1x1xi32>
      %wide = tt.broadcast %scaled : tensor<1x1xi32> -> tensor<1x128xi32>
      %offs = arith.addi %wide, %cols2 : tensor<1x128xi32>
      %iptr = tt.splat %in : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
      %optr = tt.splat %out : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
      %ia = tt.addptr %iptr, %offs : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
      %oa = tt.addptr %optr, %offs : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
      %v = tt.load %ia : tensor<1x128x!tt.ptr<f16>>
      tt.store %oa, %v : tensor<1x128x!tt.ptr<f16>>
    }
    tt.return
  }
}
