// RUN: triton-opt %s -split-input-file -pim-dce | FileCheck %s

// 整算子级 IR 里，结果没人用的张量就是内核的输出：`pim-lower-to-emitc`
// 把最后一个这样的结果当作输出参数。死代码消除不能把它删掉。
// CHECK-LABEL: @keep_operator_output
// CHECK: pim.eltwise
module {
  tt.func @keep_operator_output(%x: tensor<4xf32>, %y: tensor<4xf32>) {
    %y0 = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.return
  }
}

// -----

// 带外读取的中间值同样没人用，但也不是死的：动态量化的相 1 算出的缩放
// 就是这样被下游节点读取的。更早的无用户张量结果也要保留。
// CHECK-LABEL: @keep_outofband_intermediate
// CHECK-COUNT-2: pim.eltwise
module {
  tt.func @keep_outofband_intermediate(%x: tensor<4xf32>, %y: tensor<4xf32>) {
    %scale = pim.eltwise %x, %y
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %out = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.return
  }
}

// -----

// 结果被显式存储的算子保留。
// CHECK-LABEL: @keep_stored
// CHECK: pim.eltwise
module {
  tt.func @keep_stored(%x: tensor<4xf32>, %y: tensor<4xf32>,
                       %o: tensor<4x!tt.ptr<f32>>) {
    %a = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %a : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 没人用、又不产出张量的纯算子是死的，删掉。
// CHECK-LABEL: @drop_dead_scalar
// CHECK-NOT: arith.addi
module {
  tt.func @drop_dead_scalar(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c = arith.constant 1 : i32
    %d = arith.addi %c, %c : i32
    tt.return %x : tensor<4xf32>
  }
}

// -----

// A 路的输出走 tt.store，其它方言没人用的张量运算是死的，多轮删除。
// CHECK-LABEL: @drop_dead_tensor_chain
// CHECK-NOT: arith.muli
// CHECK-NOT: tt.addptr
// CHECK: arith.subi
// CHECK: tt.store
module {
  tt.func @drop_dead_tensor_chain(%x: tensor<4xi32>, %p: tensor<4x!tt.ptr<i32>>) {
    %a = arith.muli %x, %x : tensor<4xi32>
    %q = tt.addptr %p, %a : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %s = arith.subi %x, %x : tensor<4xi32>
    tt.store %p, %s : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// 标了 Pure 的 pim.* 也是纯算子：没人用的 tasklet_id 与它派生的标量都删。
// CHECK-LABEL: @drop_dead_pure_pim
// CHECK-NOT: pim.tasklet_id
// CHECK-NOT: pim.dpu_id
// CHECK: tt.return
module {
  tt.func @drop_dead_pure_pim(%x: tensor<4xf32>) -> tensor<4xf32> {
    %tid = pim.tasklet_id : i32
    %did = pim.dpu_id : i32
    %d = arith.addi %tid, %did : i32
    tt.return %x : tensor<4xf32>
  }
}

// -----

// A 路函数第一个参数是指针，输出走 tt.store，没人用的 pim 布局算子是死的。
// CHECK-LABEL: @drop_dead_pim_on_a_path
// CHECK-NOT: pim.convert_layout
// CHECK-NOT: pim.transpose
// CHECK: tt.store
module {
  tt.func @drop_dead_pim_on_a_path(%base: !tt.ptr<f32>, %p: tensor<4x4x!tt.ptr<f32>>, %x: tensor<4x4xf32>,
                                   %y: tensor<4x4xf32, #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [1, 4], order = [1, 0]}>>) {
    %c = pim.convert_layout %y : tensor<4x4xf32, #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [1, 4], order = [1, 0]}>> -> tensor<4x4xf32, #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [4, 1], order = [0, 1]}>>
    %t = pim.transpose %x {axes = array<i64: 1, 0>} : tensor<4x4xf32> -> tensor<4x4xf32>
    tt.store %p, %x : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 整算子函数的内核输出按 `pim-lower-to-emitc` 的口径认定：最后一批没人用的
// 张量结果。产出它的算子即使不是 pim.* 也不能当死代码删，否则输出会退回成
// 更早的结果，dtype 与数值都变。
// CHECK-LABEL: @keep_trailing_arith_output
// CHECK: pim.lut
// CHECK: arith.truncf
module {
  tt.func @keep_trailing_arith_output(%x: tensor<4xf32>) {
    %y = pim.lut %x {kind = #pim.activation<relu>} : tensor<4xf32> -> tensor<4xf32>
    %z = arith.truncf %y : tensor<4xf32> to tensor<4xf16>
    tt.return
  }
}

// -----

// A 路内核可以用指针张量作参数：元素类型是 !tt.ptr 的张量参数不是整块张量，
// 这种函数里没人用的 pim.* 照删。
// CHECK-LABEL: @drop_dead_pim_with_pointer_tensor_args
// CHECK-NOT: pim.lut
// CHECK: tt.store
module {
  tt.func public @drop_dead_pim_with_pointer_tensor_args(
      %p: tensor<4x!tt.ptr<f32>>, %o: tensor<4x!tt.ptr<f32>>) {
    %x = tt.load %p : tensor<4x!tt.ptr<f32>>
    %y = pim.lut %x {kind = #pim.activation<relu>} : tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %x : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 整算子函数可以有 memdesc 参数（KV 写回就是这样）。第一个参数不是张量不代表
// 这是 A 路函数，没人用的张量结果仍是内核输出。
// CHECK-LABEL: @keep_tensor_output_behind_a_memdesc
// CHECK: pim.lut
module {
  tt.func @keep_tensor_output_behind_a_memdesc(
      %c: !pim.memdesc<64xi8, #pim.mram>, %x: tensor<4xf32>) {
    %y = pim.lut %x {kind = #pim.activation<relu>} : tensor<4xf32> -> tensor<4xf32>
    tt.return
  }
}

// -----

// 没有结果、但有内存副作用的算子不能当成无用户的纯算子删掉：写操作本身
// 就是内核的全部效果。
// CHECK-LABEL: @keep_memory_effects
// CHECK: pim.wram_alloc
// CHECK: pim.wram_store
// CHECK: pim.barrier
// CHECK: pim.dma_store
// CHECK: pim.kv_cache
module {
  tt.func @keep_memory_effects(%x: tensor<4xf16>, %p: tensor<4x!tt.ptr<f16>>,
                               %c: !pim.memdesc<64xi8, #pim.mram>, %pos: i32,
                               %v: tensor<4xi8>) {
    %buf = pim.wram_alloc : !pim.memdesc<4xf16, #pim.wram>
    pim.wram_store %x, %buf : tensor<4xf16> -> !pim.memdesc<4xf16, #pim.wram>
    pim.barrier
    pim.dma_store %buf -> %p : !pim.memdesc<4xf16, #pim.wram> -> tensor<4x!tt.ptr<f16>>
    pim.kv_cache %v, %c, %pos
        {layer = 0 : i64, isKey, mode = #pim.kv_mode<range_write>,
         inputBufferPolicy = "L2A_ignore"}
       : tensor<4xi8>, !pim.memdesc<64xi8, #pim.mram>, i32
    tt.return
  }
}
