// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// Round-trip of the operator-level primitives: the granularity the target's
// graph format works in, one level above the tile-level ops in `ops.mlir`.

// CHECK-LABEL: @quantize_dequantize
module {
  tt.func @quantize_dequantize(%x: tensor<128x256xf32>, %s: tensor<1xf32>,
                               %ps: tensor<256xf32>, %zp: tensor<256xi8>) {
    // CHECK: pim.quantize %{{.*}}, %{{.*}} {spec = #pim.quant_spec<granularity = per_tensor, spc = false>}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, spc = false>}
       : tensor<128x256xf32>, tensor<1xf32> -> tensor<128x256xi8>
    // CHECK: pim.quantize %{{.*}}, %{{.*}}, %{{.*}} {spec = #pim.quant_spec<granularity = per_channel, axis = 1>}
    %qc = pim.quantize %x, %ps, %zp
        {spec = #pim.quant_spec<granularity = per_channel, axis = 1>}
        : tensor<128x256xf32>, tensor<256xf32>, tensor<256xi8> -> tensor<128x256xi8>
    // CHECK: pim.dequantize
    %d = pim.dequantize %q, %s {spec = #pim.quant_spec<granularity = per_tensor, spc = false>}
       : tensor<128x256xi8>, tensor<1xf32> -> tensor<128x256xf32>
    tt.return
  }
}

// -----

// A per-group spec carries the group size along its axis.
// CHECK-LABEL: @quantize_per_group
module {
  tt.func @quantize_per_group(%x: tensor<128x512xf32>, %s: tensor<4xf32>) {
    // CHECK: granularity = per_group, axis = 1, groupSize = 128
    %q = pim.quantize %x, %s
       {spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>}
       : tensor<128x512xf32>, tensor<4xf32> -> tensor<128x512xi8>
    tt.return
  }
}

// -----

// The datapath configuration survives round-trip, including several rescaling
// blocks and an absent one.
// CHECK-LABEL: @datapath_variants
module {
  tt.func @datapath_variants(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>,
                             %bias: tensor<256xi32>) {
    // CHECK: kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>]
    %m = pim.matmul %a, %b, %bias
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                 scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1>,
                                 kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>]>,
        unit = #pim.unit<nmu>}
       : tensor<128x512xi8>, tensor<512x256xi8>, tensor<256xi32> -> tensor<128x256xi8>
    // CHECK: #pim.kantor_block<id = "A", mode = scalar>, #pim.kantor_block<id = "B", mode = elementwise_mul_fp16
    %m2 = pim.matmul %a, %b
        {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                  kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>,
                                                  #pim.kantor_block<id = "B", mode = elementwise_mul_fp16>]>}
        : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    // CHECK: pim.matmul{{.*}}nmuMode = fixed2float
    %m3 = pim.matmul %a, %b
        {datapath = #pim.datapath<nmuMode = fixed2float, scaleMode = floating_point>}
        : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xf16>
    tt.return
  }
}

// -----

// `transposeB` takes the second operand as [N, K]; `bIsActivation` marks it as
// another operator's output rather than a weight.
// CHECK-LABEL: @matmul_flags
module {
  tt.func @matmul_flags(%a: tensor<128x512xi8>, %bt: tensor<256x512xi8>) {
    // CHECK: pim.matmul{{.*}}bIsActivation{{.*}}transposeB
    %m = pim.matmul %a, %bt
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>,
        transposeB, bIsActivation}
       : tensor<128x512xi8>, tensor<256x512xi8> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// A grouped reduction on the pooling unit: dynamic quantization's phase 0,
// which takes one absmax per quantized group. Distinct from `pim.pool`, whose
// window is geometry rather than a quantization group.
// CHECK-LABEL: @global_pool
module {
  tt.func @global_pool(%x: tensor<1x4096xf16>) {
    // CHECK: pim.global_pool %{{.*}} {groupSize = 128 : i64, kind = #pim.pool_kind<absmax>{{.*}}unit = #pim.unit<vpu>}
    %m = pim.global_pool %x {groupSize = 128 : i64, kind = #pim.pool_kind<absmax>,
                             unit = #pim.unit<vpu>}
        : tensor<1x4096xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// The lookup table's addressing window and table selection. The one entry the
// three together would address is on the activation unit; the identity kind is
// what dynamic quantization's phase 1 evaluates, and it is spelled out rather
// than borrowed from `relu`.
// CHECK-LABEL: @lut_window_and_table
module {
  tt.func @lut_window_and_table(%x: tensor<1x32xf16>) {
    // CHECK: activationMode = 1{{.*}}flpMantisa = 3{{.*}}flpMaxExp = 17{{.*}}flpMinExp = 10{{.*}}kind = #pim.activation<identity>{{.*}}specialOperators = 0
    %a = pim.lut %x {kind = #pim.activation<identity>, activationMode = 1 : i64,
                     flpMinExp = 10 : i64, flpMaxExp = 17 : i64, flpMantisa = 3 : i64,
                     specialOperators = 0 : i64}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    // CHECK: kind = #pim.activation<reciprocal>{{.*}}specialOperators = 4
    %b = pim.lut %a {kind = #pim.activation<reciprocal>, activationMode = 0 : i64,
                     flpMinExp = 15 : i64, flpMaxExp = 15 : i64, flpMantisa = 0 : i64,
                     specialOperators = 4 : i64}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// The fixed-point unit's rescale, standalone. `fixed_point` is the setting the
// KV write uses and the only place the target leaves the 16-bit phase format.
// CHECK-LABEL: @fpsu_scale_fixed_point
module {
  tt.func @fpsu_scale_fixed_point(%x: tensor<1x32x128xf16>,
                                  %b: tensor<1x32x128xf32>,
                                  %s: tensor<1x32x128xf16>) {
    // CHECK: pim.fpsu_scale %{{.*}}, %{{.*}}, %{{.*}} {shift = 14 : i8, spec = #pim.fpsu_spec<mode = fixed_point>}
    %y = pim.fpsu_scale %x, %b, %s
        {shift = 14 : i8, spec = #pim.fpsu_spec<mode = fixed_point>}
        : tensor<1x32x128xf16>, tensor<1x32x128xf32>, tensor<1x32x128xf16>
        -> tensor<1x32x128xi8>
    tt.return
  }
}

// -----

// The elementwise-multiply unit. `cardValue` is separate from `mode` on
// purpose: the RoPE multiplies are card value 5 -- `scalar`, if read off the
// mode enum -- while actually being elementwise multiplies.
// CHECK-LABEL: @kantor_modes
module {
  tt.func @kantor_modes(%x: tensor<1x4096xf16>, %m: tensor<1x4096xf16>) {
    // CHECK: pim.kantor %{{.*}} {shift = -8 : i8, spec = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>}
    %q = pim.kantor %x {shift = -8 : i8, spec = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>}
        : tensor<1x4096xf16> -> tensor<1x4096xi8>
    // CHECK: pim.kantor %{{.*}} rhs %{{.*}} {spec = #pim.kantor_spec<mode = elementwise_mul_fp16, cardValue = 5>}
    %p = pim.kantor %x rhs %m
        {spec = #pim.kantor_spec<mode = elementwise_mul_fp16, cardValue = 5>}
        : tensor<1x4096xf16> rhs tensor<1x4096xf16> -> tensor<1x4096xf16>
    tt.return
  }
}

// -----

// How the weight operand is staged on the weight path. The element width is
// what separates a model weight from an activation used as one, and once both
// are signless quantized integers the type no longer says which is which.
// CHECK-LABEL: @weight_binding
module {
  tt.func @weight_binding(%a: tensor<128x512xi8>, %w: tensor<512x256xi8>) {
    // int4 is stored one byte per element, so `elemBits` is a width and not a
    // packing: there is nothing to unpack.
    // CHECK: #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>
    %m = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// A transposed weight and a KV slice staged as a weight, with the subnormal
// guard and a content fingerprint.
// CHECK-LABEL: @weight_binding_variants
module {
  tt.func @weight_binding_variants(%a: tensor<128x512xi8>, %w: tensor<256x512xi8>) {
    // CHECK: #pim.weight_binding<format = weights_transpose, role = model_weight, elemBits = 4, sfMultiplier = 4>
    %m1 = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        transposeB,
        weightBinding = #pim.weight_binding<format = weights_transpose, role = model_weight,
                                            elemBits = 4, sfMultiplier = 4>}
       : tensor<128x512xi8>, tensor<256x512xi8> -> tensor<128x256xi8>
    // CHECK: role = activation_as_weight, elemBits = 8{{.*}}contentHash = "{{[0-9a-f]+}}"
    %m2 = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        transposeB,
        weightBinding = #pim.weight_binding<format = weights_transpose,
                                            role = activation_as_weight, elemBits = 8,
                                            contentHash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef">}
       : tensor<128x512xi8>, tensor<256x512xi8> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// The quantization role. The layout says how many scales there are; the role
// says whether they mean anything -- a placeholder that is not really
// quantized and an fp16 tensor that is merely clamped have the same layout as
// a genuinely quantized one.
// CHECK-LABEL: @quant_spec_roles
module {
  tt.func @quant_spec_roles(%x: tensor<128x256xf16>, %s: tensor<1xf16>,
                            %w: tensor<256x128xi8>) {
    // The default role is `intermediate`, so it does not print -- which is what
    // keeps existing IR parsing unchanged.
    // CHECK: pim.quantize %{{.*}}, %{{.*}} {spec = #pim.quant_spec<granularity = per_tensor, spc = false>}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, spc = false>}
       : tensor<128x256xf16>, tensor<1xf16> -> tensor<128x256xi8>
    // `fp_clamp`: the element type stays f16 and the bound is real, so only an
    // explicit range can express it.
    // CHECK: role = fp_clamp, range = [-6.550400e+04, 6.550400e+04]
    %c = pim.quantize %x, %s
       {spec = #pim.quant_spec<granularity = per_tensor, role = fp_clamp,
                               range = [-6.5504e+04, 6.5504e+04], spc = false>}
       : tensor<128x256xf16>, tensor<1xf16> -> tensor<128x256xf16>
    // A placeholder that occupies the slot without quantizing.
    // CHECK: role = transparent
    %t = pim.quantize %x, %s
       {spec = #pim.quant_spec<granularity = per_tensor, role = transparent, spc = false>}
       : tensor<128x256xf16>, tensor<1xf16> -> tensor<128x256xf16>
    // CHECK: role = weight
    %d = pim.dequantize %w, %s
       {spec = #pim.quant_spec<granularity = per_tensor, role = weight, spc = false>}
       : tensor<256x128xi8>, tensor<1xf16> -> tensor<256x128xf16>
    tt.return
  }
}

// -----

// The word embedding: pure addressing, no arithmetic. The result takes the
// indices' shape followed by the table's row shape.
// CHECK-LABEL: @gather_rows
module {
  tt.func @gather_rows(%table: tensor<32000x4096xf16>, %ids: tensor<1x16xi32>) {
    // CHECK: pim.gather %{{.*}}, %{{.*}} : tensor<32000x4096xf16>, tensor<1x16xi32> -> tensor<1x16x4096xf16>
    %y = pim.gather %table, %ids
        : tensor<32000x4096xf16>, tensor<1x16xi32> -> tensor<1x16x4096xf16>
    tt.return
  }
}

// -----

// The add/concat unit takes more than two slots: the residual add has two, a
// concat thirty-two. Two operands stay the common case and print exactly as
// before -- cost extraction counts these by mnemonic, so the binary spelling
// must not shift.
// CHECK-LABEL: @eltwise_variadic_slots
module {
  tt.func @eltwise_variadic_slots(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>,
                                  %c: tensor<4x8xf16>) {
    // CHECK: pim.eltwise %{{.*}}, %{{.*}} {{{.*}}kind = #pim.eltwise<add>
    %two = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    // Three slots, each with its own scaling. The residual add is why this
    // exists: its inputs carry separate scale sets.
    // CHECK: pim.eltwise %{{.*}}, %{{.*}}, %{{.*}} {{{.*}}perSlotDatapath = [#pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>, #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>]
    %three = pim.eltwise %a, %b, %c
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        perSlotDatapath = [#pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
                           #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>,
                           #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>]}
       : tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// Normalization's epsilon is an operand, not an attribute: the target reads it
// from a buffer of its own (`RMSNorm_Add_Const`), and an f64 attribute could not
// be pointed at that buffer.
//
// The two trailing operands are keyword-tagged because they are both optional
// and both tensors. Positionally, `%x, %w, %e` could mean either (weight, bias)
// or (weight, epsilon) -- the parser would have to guess, and it guessed bias,
// so a round-trip silently turned an epsilon into a bias.
// CHECK-LABEL: @normalize_epsilon_operand
module {
  tt.func @normalize_epsilon_operand(%x: tensor<4x4096xf16>, %w: tensor<4096xi8>,
                                     %e: tensor<1xf32>) {
    // Epsilon with no bias -- the shape RMS normalization actually has.
    // CHECK: pim.normalize %{{.*}}, %{{.*}} eps %{{.*}} {{{.*}}rmsNorm{{.*}}vpuParams = #pim.vpu_params<>
    %y = pim.normalize %x, %w eps %e
       {axis = -1 : i64, rmsNorm, vpuParams = #pim.vpu_params<>,
        unit = #pim.unit<vpu>}
       : tensor<4x4096xf16>, tensor<4096xi8> eps tensor<1xf32>
       -> tensor<4x4096xf16>
    tt.return
  }
}

// -----

// An absent `vpuParams` and an explicit all-defaults one describe the same
// hardware: axis -1, no input scaling, which is every occurrence measured in
// the reference artifact.
// CHECK-LABEL: @normalize_vpu_params_defaults
module {
  tt.func @normalize_vpu_params_defaults(%x: tensor<4x4096xf16>) {
    // CHECK: vpuParams = #pim.vpu_params<useScaling = true>
    %y = pim.normalize %x
       {axis = -1 : i64, rmsNorm,
        vpuParams = #pim.vpu_params<axis = -1, useScaling = true>}
       : tensor<4x4096xf16> -> tensor<4x4096xf16>
    tt.return
  }
}

// -----

// The mask's geometry. Decode masks one query position against the whole cache
// (a row); prefill masks a square where each position sees only what precedes
// it. Once a batch axis is 1 the shapes alone no longer say which is meant,
// which is why the layout is written down rather than inferred.
// CHECK-LABEL: @mask_layouts
module {
  tt.func @mask_layouts(%sv: tensor<1x32x1x128xf16>, %mv: tensor<1x1x1x128xf16>,
                        %st: tensor<1x32x16x16xf16>, %mt: tensor<1x1x16x16xf16>) {
    // `vector` is the default, so decode's mask prints without it.
    // CHECK: pim.mask %{{.*}}, %{{.*}} {transposeInput}
    %yv = pim.mask %sv, %mv {transposeInput}
        : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x32x1x128xf16>
    // CHECK: pim.mask %{{.*}}, %{{.*}} {layout = #pim.mask_layout<causal_tril>}
    %yt = pim.mask %st, %mt {layout = #pim.mask_layout<causal_tril>}
        : tensor<1x32x16x16xf16>, tensor<1x1x16x16xf16> -> tensor<1x32x16x16xf16>
    tt.return
  }
}

// -----

// Which operand stays resident on the weight path. This is the prefill/decode
// difference in one attribute, and the largest cost difference inside a PIM
// matmul: a resident operand is loaded once, a streamed one is re-read every
// pass. It cannot be inferred -- the shapes are the same either way.
// CHECK-LABEL: @matmul_stationarity
module {
  tt.func @matmul_stationarity(%a: tensor<128x512xi8>, %w: tensor<512x256xi8>,
                               %kv: tensor<256x512xi8>) {
    // A model weight stays resident: the seven projections.
    // CHECK: stationarity = #pim.stationarity<weight>
    %proj = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>,
        stationarity = #pim.stationarity<weight>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    // A cached K/V slice rides the weight path: decode's attention multiplies.
    // CHECK: stationarity = #pim.stationarity<kv>
    %attn = pim.matmul %a, %kv
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        weightBinding = #pim.weight_binding<format = weights_transpose, role = activation_as_weight, elemBits = 8>,
        stationarity = #pim.stationarity<kv>, transposeB, bIsActivation}
       : tensor<128x512xi8>, tensor<256x512xi8> -> tensor<128x256xi8>
    // Prefill multiplies activation by activation: nothing is resident, so
    // there is no weight binding at all.
    // CHECK: stationarity = #pim.stationarity<activation>
    // CHECK-NOT: weightBinding
    %pre = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        stationarity = #pim.stationarity<activation>, bIsActivation}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// The grouped-dequantization accumulator. The integer MAC accumulates one group
// at a time and dequantizes each group before folding it into the running
// total. The alternative -- accumulate the whole row in integers, then
// dequantize once -- gives different numbers, because two groups with different
// scales cannot share one integer accumulator.
// CHECK-LABEL: @matmul_group_dequant_accum
module {
  tt.func @matmul_group_dequant_accum(%a: tensor<128x512xi8>, %w: tensor<512x256xi8>,
                                     %s: tensor<4x256xf16>) {
    // The factors are an operand, one per (group, column) = [512/128, 256].
    // A single scalar could not express this: applied at each boundary it equals
    // the same scalar applied once at the end of the row, so the two orders this
    // datapath distinguishes would compute identical numbers.
    // CHECK: pim.matmul %{{.*}}, %{{.*}} scales %
    // CHECK-SAME: groupDequantAccum = true, groupSize = 128
    // CHECK-SAME: scales tensor<4x256xf16>
    %m = pim.matmul %a, %w scales %s
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point,
                                 groupDequantAccum = true, groupSize = 128>,
        weightBinding = #pim.weight_binding<format = weight, role = model_weight,
                                            elemBits = 4, groupSize = 128>}
       : tensor<128x512xi8>, tensor<512x256xi8> scales tensor<4x256xf16>
       -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// A permutation says why it exists, which is a different question from how:
// swapping the head axis is a layer of its own, while reordering the halves of
// a rotary embedding is absorbed into the rotary op, and transposing K for a
// score matmul disappears into the weight's layout.
// CHECK-LABEL: @transpose_purpose
module {
  tt.func @transpose_purpose(%x: tensor<1x128x32x128xf16>) {
    // CHECK: #pim.transpose_purpose<purpose = tensor_transpose>
    %a = pim.transpose %x {axes = array<i64: 0, 2, 1, 3>, purpose = #pim.transpose_purpose<purpose = tensor_transpose>}
       : tensor<1x128x32x128xf16> -> tensor<1x32x128x128xf16>
    // CHECK: #pim.transpose_purpose<purpose = layout_reorder, cardValue = 2>
    %b = pim.transpose %a {axes = array<i64: 0, 2, 1, 3>, purpose = #pim.transpose_purpose<purpose = layout_reorder, cardValue = 2>}
       : tensor<1x32x128x128xf16> -> tensor<1x128x32x128xf16>
    // `onthefly` claims the layout already agrees, so no node is emitted for
    // it and the graph's node count is unaffected.
    // CHECK: onthefly
    %c = pim.transpose %b {axes = array<i64: 0, 2, 1, 3>, onthefly, purpose = #pim.transpose_purpose<purpose = absorbed>}
       : tensor<1x128x32x128xf16> -> tensor<1x32x128x128xf16>
    tt.return
  }
}

// -----

// `onthefly` on the shape- and order-changing ops is a separate marker rather
// than something inferred from `Pure`: a reshape reads as side-effect free and
// still needs a pass over memory unless the layout already agrees.
// CHECK-LABEL: @view_ops_are_onthefly
module {
  tt.func @view_ops_are_onthefly(%x: tensor<4x8xf16>, %y: tensor<4x8xf16>) {
    // CHECK: onthefly
    %r = pim.reshape %x {onthefly} : tensor<4x8xf16> -> tensor<2x16xf16>
    // CHECK: onthefly
    %s:2 = pim.split %x {axis = 1 : i64, onthefly}
       : tensor<4x8xf16> -> tensor<4x4xf16>, tensor<4x4xf16>
    // CHECK: onthefly
    %c = pim.concat %x, %y {axis = 0 : i64, onthefly}
       : tensor<4x8xf16>, tensor<4x8xf16> -> tensor<8x8xf16>
    // CHECK: onthefly
    %h:2 = pim.split_heads %x {axis = 1 : i64, numHeads = 2 : i64, onthefly}
       : tensor<4x8xf16> -> tensor<4x4xf16>, tensor<4x4xf16>
    tt.return
  }
}

// -----

// How a partial result rejoins its consumer. The graph format has no field for
// it -- an exhaustive search of the merge-unit, partial-sum and skip-connection
// families finds nothing -- so setting it must leave the emitted graph
// unchanged. Residual addition is a skip connection: the bypass is folded in
// after rescaling rather than taking a traversal of its own.
// CHECK-LABEL: @combine_mode
module {
  tt.func @combine_mode(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>) {
    // CHECK: #pim.combine_mode<skip_connection>
    %s = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, combineMode = #pim.combine_mode<skip_connection>}
       : tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// A format conversion carries no scaling and no bias, so the extension digits
// are statements about the layout rather than about arithmetic. Left out, they
// are derivable from the element type (fp16 is 3, int8 is 1) -- which is why
// nothing emits this op on the reference graph: a digit appears on every edge,
// and an op for each would be hundreds of nodes for a value nothing reads.
// CHECK-LABEL: @convert_digits
module {
  tt.func @convert_digits(%x: tensor<1x16xf16>) {
    // CHECK: pim.convert {{.*}} : tensor<1x16xf16> -> tensor<1x16xi8>
    %a = pim.convert %x : tensor<1x16xf16> -> tensor<1x16xi8>
    // Attributes print in alphabetical order, so the destination digit comes
    // first even though the source one is written first.
    // CHECK: {dstExtension = 1 : i64, srcExtension = 3 : i64}
    %b = pim.convert %x {srcExtension = 3 : i64, dstExtension = 1 : i64}
       : tensor<1x16xf16> -> tensor<1x16xi8>
    tt.return
  }
}

// -----

// The revision the graph is written for belongs to the module as a whole, the
// same way the hardware description does. It is the dynamic-quantization
// family's marker rather than a phase-shaped marker: 37 nodes carry it while 69
// are phase-shaped, so anything counting nodes by it would be wrong by 32.
// CHECK-LABEL: @rtl_version_is_a_module_attribute
module attributes {"pim.rtl-version" = "1.4"} {
  tt.func @rtl_version_is_a_module_attribute() {
    tt.return
  }
}

// -----

// The two cache write paths differ in addressing rather than in extent, so one
// op carries both: a scatter takes its destination from an index, a range write
// walks contiguous addresses. The index is required in one and forbidden in the
// other, because an index nothing reads is a second source of truth.
//
// Both run through the one fixed-point datapath in the network -- scaling 2.0,
// right shift 14 -- which is why the scaling side is an operand of its own.
// CHECK-LABEL: @kv_cache_modes
module {
  tt.func @kv_cache_modes(%k: tensor<1x8x128xi8>, %n: tensor<1x8x128xi8>,
                          %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>, %pos: i32,
                          %slot: tensor<1xi16>, %sf: tensor<1xf16>, %zp: tensor<1xf16>) {
    // `scatter` is written in the source and absent from the print, which is
    // how the default makes itself known -- a value equal to the default is
    // elided, and no other mode would round-trip the same way.
    // CHECK: pim.kv_cache %{{.*}}, %{{.*}}, %{{.*}}[%{{.*}}] sf %{{.*}} zp %{{.*}} {inputBufferPolicy = "L2A_ignore", isKey, layer = 0 : i64}
    pim.kv_cache %k, %cache, %pos[%slot] sf %sf zp %zp
        {layer = 0 : i64, isKey, mode = #pim.kv_mode<scatter>, inputBufferPolicy = "L2A_ignore"}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32 [tensor<1xi16>] sf tensor<1xf16> zp tensor<1xf16>
    // CHECK: pim.kv_cache %{{.*}}, %{{.*}}, %{{.*}} {isKey, layer = 0 : i64, mode = #pim.kv_mode<range_write>}
    pim.kv_cache %n, %cache, %pos {layer = 0 : i64, isKey, mode = #pim.kv_mode<range_write>}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32
    // CHECK: pim.kv_cache read %{{.*}}, %{{.*}}, %{{.*}} {layer = 0 : i64, mode = #pim.kv_mode<read>}
    pim.kv_cache read %n, %cache, %pos {layer = 0 : i64, mode = #pim.kv_mode<read>}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32
    tt.return
  }
}


// -----

// CHECK-LABEL: @dynamic_quant_roundtrip
module {
  tt.func @dynamic_quant_roundtrip(%x: tensor<1x4096xf16>) {
    // CHECK: pim.dynamic_quant %{{.*}} {axis = 1 : i64, groupSize = 128 : i64
    %q, %s = pim.dynamic_quant %x
       {groupSize = 128 : i64, axis = 1 : i64,
        spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>}
       : tensor<1x4096xf16> -> tensor<1x4096xi8>, tensor<32xf16>
    tt.return
  }
}
