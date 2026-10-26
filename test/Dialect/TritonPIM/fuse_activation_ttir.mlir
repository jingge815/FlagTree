// RUN: triton-opt %s -split-input-file -pim-fuse-activation | FileCheck %s

// A 路形态：tt.dot 后接 Triton 对 `acc * sigmoid(acc)` 展开出的五连
// （0-x → exp → 1+e → 1/(1+e) → x*s）。convert-triton-to-pim 不碰这些
// 浮点算子（实测），融合要在 arith/math 形态上识别，折进 dot 的
// activation 属性。正例覆盖两种链根：dot 结果直接出循环（K 不分块时
// 没有 K 循环），以及经单携带值循环拿到的循环结果。

// CHECK-LABEL: @direct
module {
  tt.func @direct(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x128xf32>
    // CHECK: tt.dot {{.*}}activation = #pim.act_spec<kind = silu>
    %acc = tt.dot %x, %w, %c0 : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
    %neg = arith.subf %c0, %acc : tensor<4x128xf32>
    %e = math.exp %neg : tensor<4x128xf32>
    %one_plus = arith.addf %e, %c1 : tensor<4x128xf32>
    %s = arith.divf %c1, %one_plus : tensor<4x128xf32>
    %silu = arith.mulf %acc, %s : tensor<4x128xf32>
    tt.return
  }
}

// -----

// CHECK-LABEL: @loop_carried
module {
  tt.func @loop_carried(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>,
                        %init: tensor<4x128xf32>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: tt.dot {{.*}}activation = #pim.act_spec<kind = silu>
    %carried = scf.for %k = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %init) -> (tensor<4x128xf32>) : i32 {
      %d = tt.dot %x, %w, %acc : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
      scf.yield %d : tensor<4x128xf32>
    }
    %neg = arith.subf %c0, %carried : tensor<4x128xf32>
    %e = math.exp %neg : tensor<4x128xf32>
    %one_plus = arith.addf %e, %c1 : tensor<4x128xf32>
    %s = arith.divf %c1, %one_plus : tensor<4x128xf32>
    %silu = arith.mulf %carried, %s : tensor<4x128xf32>
    tt.return
  }
}

// -----

// 反例：链被多出来的用户打断（sigmoid 商另有读者），融合不得触发。
// CHECK-LABEL: @broken
module {
  tt.func @broken(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>,
                  %other: tensor<4x128xf32>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x128xf32>
    // CHECK: tt.dot {{%[0-9a-z_]+}}, {{%[0-9a-z_]+}}, {{%[0-9a-z_]+}} : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
    // CHECK-NOT: activation
    %acc = tt.dot %x, %w, %c0 : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
    %neg = arith.subf %c0, %acc : tensor<4x128xf32>
    %e = math.exp %neg : tensor<4x128xf32>
    %one_plus = arith.addf %e, %c1 : tensor<4x128xf32>
    %s = arith.divf %c1, %one_plus : tensor<4x128xf32>
    %silu = arith.mulf %acc, %s : tensor<4x128xf32>
    %extra = arith.addf %s, %other : tensor<4x128xf32>
    tt.return
  }
}

// -----

// relu：Triton 把 `tl.maximum(acc, 0.0)` 展开成 maxnumf 加 0 的 splat。
// CHECK-LABEL: @relu
// CHECK: tt.dot {{.*}}activation = #pim.act_spec<kind = relu>
// CHECK-NOT: arith.maxnumf
module {
  tt.func @relu(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %acc = tt.dot %x, %w, %c0 : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
    %r = arith.maxnumf %acc, %c0 : tensor<4x128xf32>
    %t = arith.truncf %r : tensor<4x128xf32> to tensor<4x128xf16>
    tt.return
  }
}

// -----

// sigmoid：与 silu 同一段四连，只是链根只有 subf 一个用户、没有末尾的 mulf。
// CHECK-LABEL: @sigmoid
// CHECK: tt.dot {{.*}}activation = #pim.act_spec<kind = sigmoid>
// CHECK-NOT: math.exp
module {
  tt.func @sigmoid(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x128xf32>
    %acc = tt.dot %x, %w, %c0 : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
    %neg = arith.subf %c0, %acc : tensor<4x128xf32>
    %e = math.exp %neg : tensor<4x128xf32>
    %one_plus = arith.addf %e, %c1 : tensor<4x128xf32>
    %s = arith.divf %c1, %one_plus : tensor<4x128xf32>
    %t = arith.truncf %s : tensor<4x128xf32> to tensor<4x128xf16>
    tt.return
  }
}

// -----

// 反例：maxnumf 的另一侧不是 0，不是 relu，不折。
// CHECK-LABEL: @clamp_not_relu
// CHECK-NOT: activation
// CHECK: arith.maxnumf
module {
  tt.func @clamp_not_relu(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x128xf32>
    %acc = tt.dot %x, %w, %c0 : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
    %r = arith.maxnumf %acc, %c1 : tensor<4x128xf32>
    %t = arith.truncf %r : tensor<4x128xf32> to tensor<4x128xf16>
    tt.return
  }
}

// -----

// 反例：激活链在携带累加器的 K 循环里面，每轮累加后都激活。降到 C 时整段
// K 归约完只激活一次，折进 dot 会改数值，所以不折。
// CHECK-LABEL: @chain_inside_k_loop
// CHECK-NOT: activation
// CHECK: math.exp
module {
  tt.func @chain_inside_k_loop(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>,
                               %init: tensor<4x128xf32>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %carried = scf.for %k = %c0_i32 to %c64_i32 step %c32_i32
        iter_args(%acc = %init) -> (tensor<4x128xf32>) : i32 {
      %d = tt.dot %x, %w, %acc : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
      %neg = arith.subf %c0, %d : tensor<4x128xf32>
      %e = math.exp %neg : tensor<4x128xf32>
      %one_plus = arith.addf %e, %c1 : tensor<4x128xf32>
      %s = arith.divf %c1, %one_plus : tensor<4x128xf32>
      %silu = arith.mulf %d, %s : tensor<4x128xf32>
      scf.yield %silu : tensor<4x128xf32>
    }
    tt.return
  }
}

// -----

// 反例：链在 scf.if 里，只在一侧分支执行；折进 dot 会让激活无条件生效。
// CHECK-LABEL: @chain_inside_if
// CHECK-NOT: activation
// CHECK: math.exp
module {
  tt.func @chain_inside_if(%x: tensor<4x32xf16>, %w: tensor<32x128xf16>,
                           %p: i1) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<4x128xf32>
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x128xf32>
    %acc = tt.dot %x, %w, %c0 : tensor<4x32xf16> * tensor<32x128xf16> -> tensor<4x128xf32>
    %r = scf.if %p -> (tensor<4x128xf32>) {
      %neg = arith.subf %c0, %acc : tensor<4x128xf32>
      %e = math.exp %neg : tensor<4x128xf32>
      %one_plus = arith.addf %e, %c1 : tensor<4x128xf32>
      %s = arith.divf %c1, %one_plus : tensor<4x128xf32>
      %silu = arith.mulf %acc, %s : tensor<4x128xf32>
      scf.yield %silu : tensor<4x128xf32>
    } else {
      scf.yield %c0 : tensor<4x128xf32>
    }
    tt.return
  }
}
