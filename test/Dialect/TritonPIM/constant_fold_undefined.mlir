// RUN: triton-opt %s -split-input-file -pim-constant-fold | FileCheck %s

// 除零没有与参考语义对齐的结果，不折，留给运行期。
// CHECK-LABEL: @skip_div_by_zero
// CHECK: pim.eltwise
module {
  tt.func @skip_div_by_zero(%o: tensor<4x!tt.ptr<f32>>) {
    %a = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %b = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<div>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %c : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 负数开方同样不折。
// CHECK-LABEL: @skip_sqrt_negative
// CHECK: pim.lut
module {
  tt.func @skip_sqrt_negative(%o: tensor<4x!tt.ptr<f32>>) {
    %a = arith.constant dense<-1.000000e+00> : tensor<4xf32>
    %y = pim.lut %a {kind = #pim.activation<sqrt>}
       : tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %y : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
