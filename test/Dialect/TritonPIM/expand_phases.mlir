// RUN: triton-opt %s -split-input-file -pim-expand-phases | FileCheck %s

// Dynamic quantize expands to 4 phases. p1 (identity scale) and p2
// (reciprocal) both read p0 -- a fan-out, not a chain.

// CHECK-LABEL: @dq_four_phases
module {
  tt.func @dq_four_phases(%x: tensor<1x4096xf16>, %s: tensor<32xf16>,
                          %o: tensor<1x4096x!tt.ptr<i8>>) {
    // CHECK: pim.reshape
    // CHECK: pim.reduce_axis {{.*}}kind = #pim.eltwise<absmax>
    // CHECK: pim.reshape
    // CHECK: pim.lut {{.*}}activation_mode = 1 {{.*}}kind = #pim.activation<relu>
    // CHECK: pim.lut {{.*}}kind = #pim.activation<reciprocal>
    // CHECK: pim.quantize {{.*}}kantorBlocks = [#pim.kantor_block<id = "A", mode = fp2int_converter>]
    // CHECK-NOT: {dynamic
    %q = pim.quantize %x, %s
       {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128>}
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
    // CHECK: pim.reduce_axis {{.*}}kind = #pim.eltwise<max>
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<sub>
    // CHECK: pim.lut {{.*}}kind = #pim.activation<exp>
    // CHECK: pim.reduce_axis {{.*}}kind = #pim.eltwise<add>
    // CHECK: pim.lut {{.*}}kind = #pim.activation<reciprocal>
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<mul>
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
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<mul>{{.*}}pim.force-consecutive
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<mul>{{.*}}pim.force-consecutive{{.*}}pim.rotate-half
    // CHECK: pim.eltwise {{.*}}kind = #pim.eltwise<add>{{.*}}pim.force-consecutive
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
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor>}
       : tensor<4x16xf32>, tensor<1xf32> -> tensor<4x16xi8>
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
