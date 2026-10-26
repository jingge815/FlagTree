// RUN: triton-opt %s -split-input-file -pim-fuse-activation | FileCheck %s

// 带 activation 属性位的算子后接唯一用户 pim.lut 就折，判定看属性位不看名字。
// eltwise 在改造前就在融合名单里，这里验证的是按属性位判定的路径，
// 不是一个名单之外的新算子。
// CHECK-LABEL: @eltwise_by_property
module {
  tt.func @eltwise_by_property(%x: tensor<4xf32>, %y: tensor<4xf32>,
                               %o: tensor<4x!tt.ptr<f32>>) {
    // CHECK: pim.eltwise{{.*}}activation = #pim.act_spec<kind = relu>
    // CHECK-NOT: pim.lut
    %e = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %a = pim.lut %e {kind = #pim.activation<relu>}
       : tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %a : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// pim.lut 有第二个用户时不折，否则那个用户失去定义。
// CHECK-LABEL: @second_user_blocks
module {
  tt.func @second_user_blocks(%x: tensor<4xf32>, %y: tensor<4xf32>,
                              %o1: tensor<4x!tt.ptr<f32>>,
                              %o2: tensor<4x!tt.ptr<f32>>) {
    // CHECK: pim.eltwise
    // CHECK-NOT: activation
    // CHECK: pim.lut
    %e = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %a = pim.lut %e {kind = #pim.activation<relu>}
       : tensor<4xf32> -> tensor<4xf32>
    tt.store %o1, %a : tensor<4x!tt.ptr<f32>>
    tt.store %o2, %e : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 没有 activation 属性位的算子后接 pim.lut 不折。
// CHECK-LABEL: @no_activation_slot
module {
  tt.func @no_activation_slot(%x: tensor<4xf32>,
                              %o: tensor<1x!tt.ptr<f32>>) {
    // CHECK: pim.reduce_axis
    // CHECK: pim.lut
    %r = pim.reduce_axis %x {kind = #pim.eltwise<add>, axis = 0 : i64}
       : tensor<4xf32> -> tensor<1xf32>
    %a = pim.lut %r {kind = #pim.activation<relu>}
       : tensor<1xf32> -> tensor<1xf32>
    tt.store %o, %a : tensor<1x!tt.ptr<f32>>
    tt.return
  }
}
