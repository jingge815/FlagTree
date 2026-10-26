// RUN: triton-opt %s -split-input-file -pim-tile-to-budget -o %t.out 2> %t.err
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.err --check-prefix=DIAG

// 读 program id 的 grid 内核每个 program 只算一块，启动网格不变时重建循环
// 会改掉语义，与 tt.dot 路径同一条约定：不改写，超预算时给诊断。
// 同形的非 grid 循环（tile_to_budget_tail.mlir 的跨度 30 用例）会被改成步长 2。
// DIAG: left this loop untiled: grid-partitioned kernel
// CHECK-NOT: "pim.tile-m"
// CHECK-LABEL: @grid_loop
// CHECK: step %c1
module attributes {"pim.wram-bytes" = 256 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @grid_loop(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : index
    %c30 = arith.constant 30 : index
    %c1 = arith.constant 1 : index
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %pidv = tt.splat %pid : i32 -> tensor<1x8xi32>
    %base0 = arith.addi %cols2, %pidv : tensor<1x8xi32>
    scf.for %i = %c0 to %c30 step %c1 {
      %base = arith.index_cast %i : index to i32
      %row = tt.splat %base : i32 -> tensor<1xi32>
      %row2 = tt.expand_dims %row {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
      %wide = tt.broadcast %row2 : tensor<1x1xi32> -> tensor<1x8xi32>
      %offs = arith.addi %wide, %base0 : tensor<1x8xi32>
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
