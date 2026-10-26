// RUN: triton-opt %s -split-input-file -pim-cse | FileCheck %s

// 同一基本块内结构相同的两个 pim.eltwise 合并成一个。
// CHECK-LABEL: @merge_identical
// CHECK-COUNT-1: pim.eltwise
module {
  tt.func @merge_identical(%x: tensor<4xf32>, %y: tensor<4xf32>,
                           %o1: tensor<4x!tt.ptr<f32>>,
                           %o2: tensor<4x!tt.ptr<f32>>) {
    %a = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %b = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o1, %a : tensor<4x!tt.ptr<f32>>
    tt.store %o2, %b : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 属性不同（kind 不同）的两个不合并。
// CHECK-LABEL: @keep_different_kind
// CHECK-COUNT-2: pim.eltwise
module {
  tt.func @keep_different_kind(%x: tensor<4xf32>, %y: tensor<4xf32>,
                               %o1: tensor<4x!tt.ptr<f32>>,
                               %o2: tensor<4x!tt.ptr<f32>>) {
    %a = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %b = pim.eltwise %x, %y
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o1, %a : tensor<4x!tt.ptr<f32>>
    tt.store %o2, %b : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 操作数不同的两个不合并，即便属性相同。
// CHECK-LABEL: @keep_different_operands
// CHECK-COUNT-2: pim.eltwise
module {
  tt.func @keep_different_operands(%x: tensor<4xf32>, %y: tensor<4xf32>,
                                   %z: tensor<4xf32>,
                                   %o: tensor<4x!tt.ptr<f32>>) {
    %a = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %b = pim.eltwise %a, %z
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %b : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 两级链式重复一遍合并完，只剩一组。
// CHECK-LABEL: @merge_chain
// CHECK-COUNT-2: = pim.eltwise %
// CHECK-NOT: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @merge_chain(%x: tensor<4xf32>, %y: tensor<4xf32>,
                       %o1: tensor<4x!tt.ptr<f32>>,
                       %o2: tensor<4x!tt.ptr<f32>>) {
    %a1 = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %b1 = pim.eltwise %a1, %x
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %a2 = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %b2 = pim.eltwise %a2, %x
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o1, %b1 : tensor<4x!tt.ptr<f32>>
    tt.store %o2, %b2 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 整算子函数里与中间值结构相同的内核输出不合并。
// CHECK-LABEL: @keep_operator_output
// CHECK-COUNT-3: = pim.eltwise %
// CHECK: tt.return
module {
  tt.func @keep_operator_output(%x: tensor<4xf32>, %y: tensor<4xf32>) {
    %a = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %w = pim.eltwise %a, %x
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %z = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.return
  }
}

// -----

// 操作数与属性相同、但 combine 区域不同的两个 tt.reduce 不合并：一个求和一个求最大。
// CHECK-LABEL: @keep_different_reduce_region
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: arith.addf %{{.*}}, %{{.*}} : f32
module {
  tt.func @keep_different_reduce_region(%x: tensor<4xf32>, %o: !tt.ptr<f32>) {
    %a = "tt.reduce"(%x) ({
    ^bb0(%l: f32, %r: f32):
      %s = arith.addf %l, %r : f32
      tt.reduce.return %s : f32
    }) {axis = 0 : i32} : (tensor<4xf32>) -> f32
    %b = "tt.reduce"(%x) ({
    ^bb0(%l: f32, %r: f32):
      %m = arith.maximumf %l, %r : f32
      tt.reduce.return %m : f32
    }) {axis = 0 : i32} : (tensor<4xf32>) -> f32
    %c = arith.addf %a, %b : f32
    tt.store %o, %c : !tt.ptr<f32>
    tt.return
  }
}

// -----

// 分支内容不同的两个 scf.if 不合并。
// CHECK-LABEL: @keep_different_if_region
// CHECK-COUNT-2: scf.if
module {
  tt.func @keep_different_if_region(%c: i1, %x: i32, %o: !tt.ptr<i32>) {
    %a = scf.if %c -> i32 {
      %p = arith.addi %x, %x : i32
      scf.yield %p : i32
    } else {
      scf.yield %x : i32
    }
    %b = scf.if %c -> i32 {
      %q = arith.muli %x, %x : i32
      scf.yield %q : i32
    } else {
      scf.yield %x : i32
    }
    %s = arith.addi %a, %b : i32
    tt.store %o, %s : !tt.ptr<i32>
    tt.return
  }
}

// -----

// 多结果的纯算子也合并：两条相同的 pim.split 只留一条。
// CHECK-LABEL: @merge_multi_result
// CHECK-COUNT-1: pim.split
// CHECK-NOT: pim.split
// CHECK: tt.return
module {
  tt.func @merge_multi_result(%x: tensor<4x8xf16>,
                              %o: tensor<4x4x!tt.ptr<f16>>) {
    %a:2 = pim.split %x {axis = 1 : i64} : tensor<4x8xf16> -> tensor<4x4xf16>, tensor<4x4xf16>
    %b:2 = pim.split %x {axis = 1 : i64} : tensor<4x8xf16> -> tensor<4x4xf16>, tensor<4x4xf16>
    tt.store %o, %a#0 : tensor<4x4x!tt.ptr<f16>>
    tt.store %o, %a#1 : tensor<4x4x!tt.ptr<f16>>
    tt.store %o, %b#0 : tensor<4x4x!tt.ptr<f16>>
    tt.store %o, %b#1 : tensor<4x4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 整算子函数里多结果算子有一个结果没人用，它可能是内核输出，不合并。
// CHECK-LABEL: @keep_multi_result_output
// CHECK-COUNT-2: pim.split
module {
  tt.func @keep_multi_result_output(%x: tensor<4x8xf16>) {
    %a:2 = pim.split %x {axis = 1 : i64} : tensor<4x8xf16> -> tensor<4x4xf16>, tensor<4x4xf16>
    %b:2 = pim.split %x {axis = 1 : i64} : tensor<4x8xf16> -> tensor<4x4xf16>, tensor<4x4xf16>
    %c = pim.eltwise %a#0, %b#0
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4x4xf16>, tensor<4x4xf16> -> tensor<4x4xf16>
    tt.return
  }
}

