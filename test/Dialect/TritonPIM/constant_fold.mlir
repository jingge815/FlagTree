// RUN: triton-opt %s -split-input-file -pim-constant-fold | FileCheck %s

// silu 是 B 路真实算子里的激活，常量输入也要折。silu(0) = 0。
// CHECK-LABEL: @fold_silu_lut
// CHECK: arith.constant dense<0.000000e+00>
// CHECK-NOT: pim.lut
module {
  tt.func @fold_silu_lut(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<0.000000e+00> : tensor<4xf16>
    %y = pim.lut %a {kind = #pim.activation<silu>}
       : tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %y : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// max 是逐元素运算的一种，两侧都是常量时折成较大的那个。
// CHECK-LABEL: @fold_max_eltwise
// CHECK: arith.constant dense<3.000000e+00>
// CHECK-NOT: pim.eltwise
module {
  tt.func @fold_max_eltwise(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<1.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<max>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %c : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----
// 操作数全是常量的 pim.eltwise 折成一个常量。
// CHECK-LABEL: @fold_constant_eltwise
module {
  tt.func @fold_constant_eltwise(%o: tensor<4x!tt.ptr<f32>>) {
    // CHECK-NOT: pim.eltwise
    // CHECK: arith.constant dense<5.000000e+00>
    %a = arith.constant dense<2.000000e+00> : tensor<4xf32>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf32>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %c : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// phases 只记录相位序号和缓冲大小，不改变运算。softmax 展开出的 sub 与
// mul 相位都带它，操作数是常量时也要折。
// CHECK-LABEL: @fold_phased_eltwise
// CHECK-NOT: pim.eltwise
// CHECK: arith.constant dense<5.000000e+00>
module {
  tt.func @fold_phased_eltwise(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        combineMode = #pim.combine_mode<straightforward>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        phases = [#pim.phase_spec<index = 4, bytes = 8, unit = combiner>],
        isPhased, unit = #pim.unit<combiner>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %c : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// softmax 展开出的 exp 与 reciprocal 相位带查表窗口、表选择和默认的
// 浮点 fpsu。这些都不改变 kind 的数值，操作数是常量时也要折。
// CHECK-LABEL: @fold_phased_lut
// CHECK-NOT: pim.lut
// CHECK: arith.constant dense<1.000000e+00>
module {
  tt.func @fold_phased_lut(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<0.000000e+00> : tensor<4xf16>
    %y = pim.lut %a
       {kind = #pim.activation<exp>,
        activationMode = 0 : i64, specialOperators = 0 : i64,
        flpMinExp = 9 : i64, flpMaxExp = 16 : i64, flpMantisa = 3 : i64,
        fpsu = #pim.fpsu_spec<mode = floating_point>,
        phases = [#pim.phase_spec<index = 1, bytes = 8, unit = activation>],
        isPhased, unit = #pim.unit<activation>}
       : tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %y : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 常量 pim.lut 同样折掉。
// CHECK-LABEL: @fold_constant_lut
module {
  tt.func @fold_constant_lut(%o: tensor<4x!tt.ptr<f32>>) {
    // CHECK-NOT: pim.lut
    // CHECK: arith.constant
    %a = arith.constant dense<-2.000000e+00> : tensor<4xf32>
    %y = pim.lut %a {kind = #pim.activation<relu>}
       : tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %y : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// 操作数来自加载时不折，哪怕另一个是常量。
// CHECK-LABEL: @skip_loaded_operand
module {
  tt.func @skip_loaded_operand(%x: tensor<4xf32>, %o: tensor<4x!tt.ptr<f32>>) {
    // CHECK: pim.eltwise
    %b = arith.constant dense<3.000000e+00> : tensor<4xf32>
    %c = pim.eltwise %x, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    tt.store %o, %c : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// A 路：循环边界里的 arith 常量运算被折掉。
// CHECK-LABEL: @fold_arith_boundary
module {
  tt.func @fold_arith_boundary(%o: tensor<1x!tt.ptr<i32>>) {
    // CHECK-NOT: arith.addi
    // CHECK: arith.constant 6
    %a = arith.constant 2 : i32
    %b = arith.constant 4 : i32
    %c = arith.addi %a, %b : i32
    %z = tt.splat %c : i32 -> tensor<1xi32>
    tt.store %o, %z : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// 减法与乘法同样折掉，不只是加法。
// CHECK-LABEL: @fold_arith_sub_mul
module {
  tt.func @fold_arith_sub_mul(%o: tensor<1x!tt.ptr<i32>>) {
    // CHECK-NOT: arith.subi
    // CHECK-NOT: arith.muli
    // CHECK: arith.constant -6
    %a = arith.constant 2 : i32
    %b = arith.constant 4 : i32
    %d = arith.subi %a, %b : i32
    %e = arith.constant 3 : i32
    %c = arith.muli %d, %e : i32
    %z = tt.splat %c : i32 -> tensor<1xi32>
    tt.store %o, %z : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// 折入的激活会改变结果。带 relu 的 add 不能只按 kind 求值，
// 否则 [-2] + [0] 被折成 -2，而 relu 之后应为 0。
// CHECK-LABEL: @skip_eltwise_with_activation
// CHECK: pim.eltwise
// CHECK-NOT: arith.constant dense<0.000000e+00>
module {
  tt.func @skip_eltwise_with_activation(%o: tensor<1x!tt.ptr<f32>>) {
    %a = arith.constant dense<-2.000000e+00> : tensor<1xf32>
    %b = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        activation = #pim.act_spec<kind = relu>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    tt.store %o, %c : tensor<1x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// LUT 之后的缩放是数据通路的一部分。带 fpsuScale 的 exp(0) 不能折成 1，
// 缩放 2 之后结果是 2。
// CHECK-LABEL: @skip_lut_with_fpsu_scale
// CHECK: pim.lut
// CHECK-NOT: arith.constant dense<1.000000e+00>
module {
  tt.func @skip_lut_with_fpsu_scale(%o: tensor<1x!tt.ptr<f32>>) {
    %a = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %y = pim.lut %a {kind = #pim.activation<exp>, fpsuScale = 2.000000e+00 : f64}
       : tensor<1xf32> -> tensor<1xf32>
    tt.store %o, %y : tensor<1x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// B 路整算子级 IR 的张量是 f16。常量 f16 的逐元素运算也要折，
// 只认 f32 会让这条路上的常量链一次都不触发。
// CHECK-LABEL: @fold_f16_eltwise
// CHECK-NOT: pim.eltwise
// CHECK: arith.constant dense<5.000000e+00>
module {
  tt.func @fold_f16_eltwise(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %c : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// skip_connection 会把结果折进另一条链，求值只看 kind 看不到那条链，
// 所以不折。放宽 straightforward 不能连这种一起放。
// CHECK-LABEL: @keep_eltwise_with_skip_connection
// CHECK: pim.eltwise
// CHECK-NOT: arith.constant dense<5.000000e+00>
module {
  tt.func @keep_eltwise_with_skip_connection(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        combineMode = #pim.combine_mode<skip_connection>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %c : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// f16 的 LUT 同样折。
// CHECK-LABEL: @fold_f16_lut
// CHECK-NOT: pim.lut
// CHECK: arith.constant dense<0.000000e+00>
module {
  tt.func @fold_f16_lut(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<-2.000000e+00> : tensor<4xf16>
    %y = pim.lut %a {kind = #pim.activation<relu>}
       : tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %y : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// combineMode = straightforward 是相位展开给每条相位写的默认值，
// 它只说明结果原样进入下一相，不改变运算本身。带它的常量相位也要折，
// 否则 B 路展开出的相位一个都折不了。
// CHECK-LABEL: @fold_eltwise_with_straightforward_combine
// CHECK-NOT: pim.eltwise
// CHECK: arith.constant dense<5.000000e+00>
module {
  tt.func @fold_eltwise_with_straightforward_combine(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        combineMode = #pim.combine_mode<straightforward>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %c : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// softmax 展开出的 mul 相位在 phase_spec 里记 reads = [1, 3]，
// 说明它消费哪几条更早的相位。这是拓扑，不改变乘法本身。
// CHECK-LABEL: @fold_eltwise_with_phase_reads
// CHECK-NOT: pim.eltwise
// CHECK: arith.constant dense<6.000000e+00>
module {
  tt.func @fold_eltwise_with_phase_reads(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<mul>,
        combineMode = #pim.combine_mode<straightforward>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        phases = [#pim.phase_spec<index = 4, bytes = 8, unit = combiner, reads = [1, 3]>],
        isPhased, unit = #pim.unit<combiner>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %c : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// softmax 展开出的 reciprocal 相位用 specialOperators = 4 选倒数表，
// fpsu 记成 32 位累加。kind 已经是 reciprocal，两者都不改变求值结果。
// CHECK-LABEL: @fold_reciprocal_lut_with_table_select
// CHECK-NOT: pim.lut
// CHECK: arith.constant dense<5.000000e-01>
module {
  tt.func @fold_reciprocal_lut_with_table_select(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %y = pim.lut %a
       {kind = #pim.activation<reciprocal>,
        activationMode = 0 : i64, specialOperators = 4 : i64,
        flpMinExp = 15 : i64, flpMaxExp = 15 : i64, flpMantisa = 0 : i64,
        fpsu = #pim.fpsu_spec<mode = floating_point_32>,
        phases = [#pim.phase_spec<index = 3, bytes = 8, unit = activation>],
        isPhased, unit = #pim.unit<activation>}
       : tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %y : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// specialOperators = 4 选的是倒数表。kind 不是 reciprocal 时两者说的
// 不是同一个函数，按 kind 求值会对不上，所以不折。
// CHECK-LABEL: @keep_exp_lut_with_reciprocal_table
// CHECK: pim.lut
// CHECK-NOT: arith.constant dense<1.000000e+00>
module {
  tt.func @keep_exp_lut_with_reciprocal_table(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<0.000000e+00> : tensor<4xf16>
    %y = pim.lut %a
       {kind = #pim.activation<exp>, specialOperators = 4 : i64}
       : tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %y : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 表体是数据，不是寻址配置。全 0 的表配 kind = exp，按 kind 求值会得到 1，
// 而按表查值是 0，所以带 table 操作数时不折。
// CHECK-LABEL: @keep_lut_with_table
// CHECK: pim.lut
// CHECK-NOT: arith.constant dense<1.000000e+00>
module {
  tt.func @keep_lut_with_table(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<0.000000e+00> : tensor<4xf16>
    %t = arith.constant dense<0.000000e+00> : tensor<144xf16>
    %y = pim.lut %a, %t {kind = #pim.activation<exp>}
       : tensor<4xf16>, tensor<144xf16> -> tensor<4xf16>
    tt.store %o, %y : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// mode 不是 regular 时按偶数或奇数寻址，按 kind 直接算会对不上，不折。
// CHECK-LABEL: @keep_lut_with_even_mode
// CHECK: pim.lut
// CHECK-NOT: arith.constant dense<1.000000e+00>
module {
  tt.func @keep_lut_with_even_mode(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<0.000000e+00> : tensor<4xf16>
    %y = pim.lut %a {kind = #pim.activation<exp>, mode = #pim.lut_mode<even>}
       : tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %y : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// 结果类型与操作数不同时，按操作数类型折出会让下游类型对不上，不折。
// CHECK-LABEL: @keep_lut_with_different_result_type
// CHECK: pim.lut
module {
  tt.func @keep_lut_with_different_result_type(%o: tensor<4x!tt.ptr<f32>>) {
    %a = arith.constant dense<0.000000e+00> : tensor<4xf16>
    %y = pim.lut %a {kind = #pim.activation<exp>}
       : tensor<4xf16> -> tensor<4xf32>
    tt.store %o, %y : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// eltwise 同样：结果类型与操作数不同时不折，折了会生成类型对不上的 IR。
// CHECK-LABEL: @keep_eltwise_with_different_result_type
// CHECK: pim.eltwise
module {
  tt.func @keep_eltwise_with_different_result_type(%o: tensor<4x!tt.ptr<f32>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf32>
    tt.store %o, %c : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// datapath 带逐通道缩放和 kantor 块时，按 kind 求值会丢掉这份定标，不折。
// CHECK-LABEL: @keep_eltwise_with_scaled_datapath
// CHECK: pim.eltwise
// CHECK-NOT: arith.constant dense<5.000000e+00>
module {
  tt.func @keep_eltwise_with_scaled_datapath(%o: tensor<4x!tt.ptr<f16>>) {
    %a = arith.constant dense<2.000000e+00> : tensor<4xf16>
    %b = arith.constant dense<3.000000e+00> : tensor<4xf16>
    %c = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                 scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1>,
                                 kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>]>}
       : tensor<4xf16>, tensor<4xf16> -> tensor<4xf16>
    tt.store %o, %c : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// A 路真实 kernel 转 PIM 后，地址偏移的常量全是 dense i32 张量，
// 不是标量。dense<64> + dense<1> 折成 dense<65>。
// CHECK-LABEL: @fold_dense_i32_add
// CHECK-NOT: arith.addi
// CHECK: arith.constant dense<65>
module {
  tt.func @fold_dense_i32_add(%o: tensor<4x!tt.ptr<i32>>) {
    %a = arith.constant dense<64> : tensor<4xi32>
    %b = arith.constant dense<1> : tensor<4xi32>
    %c = arith.addi %a, %b : tensor<4xi32>
    tt.store %o, %c : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
