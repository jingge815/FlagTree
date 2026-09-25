// RUN: triton-opt %s -split-input-file -pim-expand-phases | FileCheck %s

// Dynamic quantize expands to 4 phases. p1 (identity scale) and p2
// (reciprocal) both read p0 -- a fan-out, not a chain.

// CHECK-LABEL: @dq_four_phases
module {
  tt.func @dq_four_phases(%x: tensor<1x4096xf16>, %s: tensor<32xf16>,
                          %o: tensor<1x4096x!tt.ptr<i8>>) {
    // Phase 0 is one grouped absmax on the pooling unit -- no reshape around it
    // any more, because a grouped reduction is what that unit does.
    // CHECK: pim.global_pool {{.*}}kind = #pim.pool_kind<absmax>{{.*}}#pim.phase_spec<index = 0, bytes = 64, unit = pooling>
    // Phase 1 is the identity table, not `relu`: the ÷256 lives in the FPSU
    // scale, so this phase only passes the value through the table.
    // CHECK: pim.lut {{.*}}activationMode = 1{{.*}}flpMinExp = 10{{.*}}kind = #pim.activation<identity>{{.*}}#pim.phase_spec<index = 1, bytes = 64, unit = activation, reads = [0]>
    // CHECK: pim.lut {{.*}}kind = #pim.activation<reciprocal>{{.*}}#pim.phase_spec<index = 2, bytes = 64, unit = activation, reads = [0]>{{.*}}specialOperators = 4
    // CHECK-SAME: transposePurpose = #pim.transpose_purpose<purpose = layout_reorder
    // 卡值只在 ODS 字段 `spec` 上出现一份——属性按字母序打印，所以
    // `phases` 在 `spec` 前面。
    // CHECK: pim.kantor {{.*}}fpsu = #pim.fpsu_spec<mode = floating_point>{{.*}}#pim.phase_spec<index = 3, bytes = 4096, unit = kantor>{{.*}}spec = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>
    // CHECK-NOT: {dynamic
    %q = pim.quantize %x, %s
       {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>}
       : tensor<1x4096xf16>, tensor<32xf16> -> tensor<1x4096xi8>
    tt.store %o, %q : tensor<1x4096x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// Softmax expands to 5 phases. The mul (p4) reads both the exp and the
// reciprocal -- a fan-out from p1 and p3.

// CHECK-LABEL: @softmax_five_phases
module {
  tt.func @softmax_five_phases(%s: tensor<1x1024xf16>,
                               %o: tensor<1x1024x!tt.ptr<f16>>) {
    // The stabilization `sub` is an FPSU affine folded into the exp phase, so
    // it is the one op here that carries no phase spec.
    // CHECK: pim.reduce_axis {{.*}}kind = #pim.eltwise<max>{{.*}}#pim.phase_spec<index = 0
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<sub>
    // CHECK-NOT: #pim.phase_spec
    // CHECK: pim.lut {{.*}}flpMinExp = 9{{.*}}kind = #pim.activation<exp>{{.*}}#pim.phase_spec<index = 1
    // CHECK: pim.reduce_axis {{.*}}kind = #pim.eltwise<add>{{.*}}#pim.phase_spec<index = 2
    // CHECK: pim.lut {{.*}}kind = #pim.activation<reciprocal>{{.*}}#pim.phase_spec<index = 3
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<mul>{{.*}}#pim.phase_spec<index = 4, bytes = 2048, unit = combiner, reads = [1, 3]>
    // CHECK-NOT: pim.softmax
    %p = pim.softmax %s {axis = 1 : i64, unit = #pim.unit<cstl>}
       : tensor<1x1024xf16> -> tensor<1x1024xf16>
    tt.store %o, %p : tensor<1x1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// RoPE expands to 3 consecutive eltwise phases. The sin path is tagged
// rotate-half; the hardware Kantor block folds the pairing.

// CHECK-LABEL: @rope_three_phases
module {
  tt.func @rope_three_phases(%x: tensor<1x4096xf16>, %cos: tensor<1x4096xf16>,
                             %sin: tensor<1x4096xf16>,
                             %o: tensor<1x4096x!tt.ptr<f16>>) {
    // All three share an intermediate buffer, hence forceConsecutive on each;
    // only the sin path is tagged rotate-half.
    // CHECK: pim.eltwise {{.*}}kantor = #pim.kantor_spec<mode = elementwise_mul_fp16, cardValue = 5>{{.*}}kind = #pim.eltwise<mul>{{.*}}#pim.phase_spec<index = 0, bytes = 8192, unit = kantor, forceConsecutive = true>
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<mul>{{.*}}#pim.phase_spec<index = 1, bytes = 8192, unit = kantor, forceConsecutive = true>{{.*}}rotateHalf
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<add>{{.*}}#pim.phase_spec<index = 2, bytes = 8192, unit = kantor, forceConsecutive = true>
    // CHECK-NOT: pim.rope
    %y = pim.rope %x, %cos, %sin {numHeads = 32 : i64, unit = #pim.unit<cstl>}
       : tensor<1x4096xf16>, tensor<1x4096xf16>, tensor<1x4096xf16>
       -> tensor<1x4096xf16>
    tt.store %o, %y : tensor<1x4096x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Static (non-dynamic) quantize is left alone.

// CHECK-LABEL: @static_quantize_untouched
module {
  tt.func @static_quantize_untouched(%x: tensor<4x16xf32>, %s: tensor<1xf32>) {
    // CHECK: pim.quantize
    // CHECK-NOT: pim.reduce_axis
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, spc = false>}
       : tensor<4x16xf32>, tensor<1xf32> -> tensor<4x16xi8>
    tt.return
  }
}

// -----

// The hardware fields are structured now. These are the ones this pass used to
// write as bare strings; a consumer reads #pim.phase_spec / #pim.fpsu_spec /
// #pim.kantor_spec / the LUT's own attributes instead, so none of the old
// spellings may survive expansion. The point is that renaming one of them is
// now a compile error rather than a field that quietly stops being emitted.

// CHECK-LABEL: @no_bare_hw_attrs
module {
  tt.func @no_bare_hw_attrs(%s: tensor<1x1024xf16>,
                            %o: tensor<1x1024x!tt.ptr<f16>>) {
    // CHECK: pim.reduce_axis
    // CHECK: #pim.phase_spec
    // CHECK-NOT: pim.phase =
    // CHECK-NOT: "pim.phase-bytes"
    // CHECK-NOT: "pim.force-consecutive"
    // CHECK-NOT: "pim.fpsu-mode"
    // CHECK-NOT: "pim.kantor-mode"
    // CHECK-NOT: "pim.activation-mode"
    // CHECK-NOT: "pim.flp-min-exp"
    // CHECK-NOT: "pim.flp-max-exp"
    // CHECK-NOT: "pim.flp-mantisa"
    // CHECK-NOT: "pim.rotate-half"
    %p = pim.softmax %s {axis = 1 : i64, unit = #pim.unit<cstl>}
       : tensor<1x1024xf16> -> tensor<1x1024xf16>
    tt.store %o, %p : tensor<1x1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Matmul and RMSNorm are single-phase and must not be expanded.

// CHECK-LABEL: @single_phase_untouched
module {
  tt.func @single_phase_untouched(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>,
                                  %x: tensor<4x16xf16>, %w: tensor<16xf16>) {
    // CHECK: pim.matmul
    // CHECK: pim.normalize
    // CHECK-NOT: pim.reduce_axis
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    %n = pim.normalize %x, %w {axis = -1 : i64, rmsNorm, unit = #pim.unit<vpu>}
       : tensor<4x16xf16>, tensor<16xf16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// The K-path RoPE requantizes on its way into the cache, and the graph format
// says so with a *flat* flag rather than a nested block: there is no folded
// operator to name, only the fact that this node ends in a requantization.
// `pim-fuse-activation` cannot produce it -- that pass folds a `pim.lut`, and
// there is none here.

// CHECK-LABEL: @rope_k_path_tail_is_a_quantization
module {
  tt.func @rope_k_path_tail_is_a_quantization(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %o: tensor<1x32x16x128x!tt.ptr<f16>>) {
    // CHECK: pim.eltwise {{.*}}contraction = #pim.contraction<form = flat, flagName = "dq_contraction">{{.*}}kind = #pim.eltwise<add>
    %y = pim.rope %x, %c, %s
       {numHeads = 32 : i64, tailCardValue = 3 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %o, %y : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// The Q path leaves its result in fp16 for the following dynamic quantize, so
// it has no quantized tail and must carry no contraction at all.

// CHECK-LABEL: @rope_q_path_has_no_contraction
module {
  tt.func @rope_q_path_has_no_contraction(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>, %o: tensor<1x32x16x128x!tt.ptr<f16>>) {
    // CHECK: pim.eltwise
    // CHECK-NOT: contraction
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.store %o, %y : tensor<1x32x16x128x!tt.ptr<f16>>
    tt.return
  }
}


// -----

// CHECK-LABEL: @dynamic_quant_four_phases
module {
  tt.func @dynamic_quant_four_phases(%arg0: tensor<1x4096xf16>, %arg1: tensor<1x4096x!tt.ptr<i8>>) {
    %q, %s = pim.dynamic_quant %arg0
       {groupSize = 128 : i64, axis = 1 : i64,
        spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>}
       : tensor<1x4096xf16> -> tensor<1x4096xi8>, tensor<1x32xf16>
    tt.store %arg1, %q : tensor<1x4096x!tt.ptr<i8>>
    tt.return
  }
}
// CHECK: %[[P0:.+]] = pim.global_pool
// CHECK: %[[P1:.+]] = pim.lut %[[P0]] {{.*}}kind = #pim.activation<identity>
// CHECK: %[[P2:.+]] = pim.lut %[[P0]] {{.*}}kind = #pim.activation<reciprocal>
// CHECK: %[[P3:.+]] = pim.kantor %{{.*}} scale %[[P2]]
// 这一相不得携带 `datapath`：KantorOp 的 ODS 没有这个字段（A/B 两块写在 `spec`
// 里），挂上去只是一个没人读、也没有 verifier 查的裸属性名。
// CHECK-NOT: datapath
// CHECK-SAME: shift = -8 : i8
// CHECK: tt.store %{{.*}}, %[[P3]]

// -----

// The scale result is phase 1 (identity, p0/256), not phase 2 (reciprocal).
// Wiring it to the reciprocal would make a later dequant `q * scale` compute
// `q / p0` instead of `q * (p0/256)`.
// CHECK-LABEL: @dynamic_quant_scale_is_phase_one
module {
  tt.func @dynamic_quant_scale_is_phase_one(%arg0: tensor<1x4096xf16>) -> (tensor<1x4096xi8>, tensor<1x32xf16>) {
    %q, %s = pim.dynamic_quant %arg0
       {groupSize = 128 : i64, axis = 1 : i64,
        spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>}
       : tensor<1x4096xf16> -> tensor<1x4096xi8>, tensor<1x32xf16>
    tt.return %q, %s : tensor<1x4096xi8>, tensor<1x32xf16>
  }
}
// CHECK: %[[P0:.+]] = pim.global_pool
// CHECK: %[[P1:.+]] = pim.lut %[[P0]] {{.*}}kind = #pim.activation<identity>
// CHECK: %[[P2:.+]] = pim.lut %[[P0]] {{.*}}kind = #pim.activation<reciprocal>
// CHECK: %[[P3:.+]] = pim.kantor %{{.*}} scale %[[P2]] {{.*}}shift = -8 : i8
// CHECK: tt.return %[[P3]], %[[P1]]
