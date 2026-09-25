// RUN: triton-opt %s -split-input-file -verify-diagnostics

// The verifiers on the operator-level ops exist to stop malformed IR before it
// reaches the graph hand-off, where a bad datapath or a mismatched scale count
// would be much harder to trace. These check that each one actually fires.

module {
  tt.func @per_tensor_takes_no_axis(%x: tensor<4x4xf32>, %s: tensor<1xf32>) {
    // expected-error @below {{per_tensor takes no axis or groupSize}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, axis = 1, spc = false>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @per_group_needs_group_size(%x: tensor<4x16xf32>, %s: tensor<4xf32>) {
    // expected-error @below {{per_group requires a positive groupSize}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_group, axis = 1, spg = true, spgAxis = 3, spgGroupSize = 0>} : tensor<4x16xf32>, tensor<4xf32> -> tensor<4x16xi8>
    tt.return
  }
}

// -----

// A per-channel scale set must have exactly one entry per channel.
module {
  tt.func @scale_count_mismatch(%x: tensor<4x16xf32>, %s: tensor<8xf32>) {
    // expected-error @+1 {{scale must supply 16 values for this quantization spec; got 8}}
    %q = pim.quantize %x, %s
       {spec = #pim.quant_spec<granularity = per_channel, axis = 1>}
       : tensor<4x16xf32>, tensor<8xf32> -> tensor<4x16xi8>
    tt.return
  }
}

// -----

module {
  tt.func @matmul_contraction_mismatch(%a: tensor<4x8xi8>, %b: tensor<16x4xi8>) {
    // expected-error @+1 {{contraction dimensions disagree: a has 8, b has 16}}
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4x8xi8>, tensor<16x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// A bias runs along the output columns.
module {
  tt.func @matmul_bad_bias(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>, %bias: tensor<7xi32>) {
    // expected-error @+1 {{bias must supply 4 or 1 values; got 7}}
    %m = pim.matmul %a, %b, %bias
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4x8xi8>, tensor<8x4xi8>, tensor<7xi32> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// The scaling block has nothing to convert, so it cannot be a converting mode.
module {
  tt.func @scale_mode_cannot_convert(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{scaleMode cannot be fixed2float}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed2float>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// A bypassed rescaling block reads no parameters.
module {
  tt.func @off_block_takes_no_spec(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{spec is meaningless when mode is off}}
    // expected-error @below {{failed to parse TTPIM_DatapathAttr parameter 'kantorBlocks'}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point, kantorBlocks = [#pim.kantor_block<id = "A", mode = off, spec = #pim.quant_spec<granularity = per_channel, axis = 0>>]>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// One physical block cannot be configured twice.
module {
  tt.func @duplicate_kantor_block(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{kantor block A is configured more than once}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point, kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>, #pim.kantor_block<id = "A", mode = scalar>]>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// The DMA engine moves data; it does not compute.
module {
  tt.func @dma_cannot_compute(%x: tensor<4x4xi8>) {
    // expected-error @+1 {{cannot run on the DMA unit}}
    %a = pim.lut %x {kind = #pim.activation<silu>, unit = #pim.unit<dma>}
       : tensor<4x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @alpha_only_for_leaky_relu(%x: tensor<4x4xi8>) {
    // expected-error @+1 {{alpha is only meaningful for leaky_relu}}
    %a = pim.lut %x {kind = #pim.activation<silu>, alpha = 1.000000e-01 : f64}
       : tensor<4x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @clip_only_for_relu_x(%x: tensor<4x4xi8>) {
    // expected-error @+1 {{clip bounds are only meaningful for relu_x}}
    %a = pim.lut %x {kind = #pim.activation<relu>, clipMax = 6.000000e+00 : f64}
       : tensor<4x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// L2 is shared by an NPU's units, so naming one would be a claim the hardware
// cannot honour.
module {
  tt.func @l2_takes_no_unit() {
    // expected-error @+1 {{an L2 allocation is shared and takes no unit}}
    %b = pim.buffer_alloc {unit = #pim.unit<nmu>} : !pim.memdesc<4x4xi8, #pim.l2>
    tt.return
  }
}

// -----

module {
  tt.func @alloc_must_be_on_chip() {
    // expected-error @+1 {{must allocate in #pim.l1 or #pim.l2}}
    %b = pim.buffer_alloc : !pim.memdesc<4x4xi8, #pim.wram>
    tt.return
  }
}

// -----

module attributes {"pim.l1-bytes" = 1024 : i64} {
  tt.func @alloc_exceeds_l1() {
    // expected-error @+1 {{exceeds the L1 budget of 1024 bytes}}
    %b = pim.buffer_alloc {unit = #pim.unit<nmu>} : !pim.memdesc<64x64xi8, #pim.l1>
    tt.return
  }
}

// -----

// A copy within one level moves nothing the allocator could not have avoided.
module {
  tt.func @copy_must_cross_a_level() {
    %a = pim.buffer_alloc : !pim.memdesc<4x4xi8, #pim.l2>
    %b = pim.buffer_alloc : !pim.memdesc<4x4xi8, #pim.l2>
    // expected-error @+1 {{an on-chip copy must cross a level}}
    pim.buffer_copy %a -> %b
      : !pim.memdesc<4x4xi8, #pim.l2> -> !pim.memdesc<4x4xi8, #pim.l2>
    tt.return
  }
}

// -----

// Decompression expands; equal or smaller output means the compressed form
// bought nothing.
module {
  tt.func @decompress_must_expand(%p: !pim.memdesc<1024xi8, #pim.mram>) {
    %d = pim.buffer_alloc : !pim.memdesc<256xi8, #pim.l2>
    // expected-error @+1 {{must be larger than the compressed}}
    pim.decompress_weight %p -> %d
      : !pim.memdesc<1024xi8, #pim.mram> -> !pim.memdesc<256xi8, #pim.l2>
    tt.return
  }
}

// -----

// The rank is kept and the reduced axis collapses to 1.
module {
  tt.func @reduce_keeps_rank(%x: tensor<8x16xf32>) {
    // expected-error @+1 {{does not match the reduction of}}
    %r = pim.reduce_axis %x {kind = #pim.eltwise<max>, axis = 1 : i64}
       : tensor<8x16xf32> -> tensor<8x16xf32>
    tt.return
  }
}

// -----

module {
  tt.func @global_pool_takes_no_window(%x: tensor<1x4x8x8xi8>) {
    // expected-error @+1 {{global_average takes no window}}
    %p = pim.pool %x
       {kind = #pim.pool<global_average>,
        window = #pim.window<kernel = [2, 2], strides = [1, 1],
                             pads = [0, 0, 0, 0], dilations = [1, 1]>}
       : tensor<1x4x8x8xi8> -> tensor<1x4x1x1xi8>
    tt.return
  }
}

// -----

module {
  tt.func @windowed_pool_needs_window(%x: tensor<1x4x8x8xi8>) {
    // expected-error @+1 {{windowed pooling requires a window}}
    %p = pim.pool %x {kind = #pim.pool<max>}
       : tensor<1x4x8x8xi8> -> tensor<1x4x4x4xi8>
    tt.return
  }
}

// -----

// Two pads per spatial dimension: [top, right, bottom, left] for the 2-D case.
module {
  tt.func @pads_two_per_dimension(%x: tensor<1x4x8x8xi8>) {
    // expected-error @below {{pads must have two entries per kernel dimension (4)}}
    %p = pim.pool %x {kind = #pim.pool<max>, window = #pim.window<kernel = [2, 2], strides = [1, 1], pads = [0, 0], dilations = [1, 1]>} : tensor<1x4x8x8xi8> -> tensor<1x4x4x4xi8>
    tt.return
  }
}

// -----

// The name is the only link to the buffer holding the parameter's bytes.
module {
  tt.func @param_needs_a_name() {
    // expected-error @+1 {{name must not be empty}}
    %w = pim.param {name = ""} : tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @transpose_axes_must_permute(%x: tensor<4x8xf32>) {
    // expected-error @+1 {{axis 0 appears more than once}}
    %t = pim.transpose %x {axes = array<i64: 0, 0>}
       : tensor<4x8xf32> -> tensor<8x4xf32>
    tt.return
  }
}

// -----

// No data moves, so the element count is preserved exactly.
module {
  tt.func @reshape_preserves_element_count(%x: tensor<4x8xf32>) {
    // expected-error @+1 {{result holds 16 elements but the source holds 32}}
    %r = pim.reshape %x : tensor<4x8xf32> -> tensor<4x4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @split_pieces_must_tile(%x: tensor<8x16xf32>) {
    // expected-error @+1 {{results sum to 12 along axis 1 but the source has 16}}
    %a, %b = pim.split %x {axis = 1 : i64}
       : tensor<8x16xf32> -> tensor<8x8xf32>, tensor<8x4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @concat_inputs_must_tile(%a: tensor<8x4xf32>, %b: tensor<8x4xf32>) {
    // expected-error @+1 {{inputs sum to 8 along axis 1 but the result has 16}}
    %c = pim.concat %a, %b {axis = 1 : i64}
       : tensor<8x4xf32>, tensor<8x4xf32> -> tensor<8x16xf32>
    tt.return
  }
}

// -----

module {
  tt.func @softmax_axis_out_of_range(%x: tensor<8x16xf32>) {
    // expected-error @+1 {{axis -5 is out of range for rank 2}}
    %p = pim.softmax %x {axis = -5 : i64} : tensor<8x16xf32> -> tensor<8x16xf32>
    tt.return
  }
}

// -----

// Both affine parameters run along the normalized axis.
module {
  tt.func @normalize_weight_count(%x: tensor<8x16xf16>, %w: tensor<8xf16>) {
    // expected-error @+1 {{weight must supply 16 values; got 8}}
    %n = pim.normalize %x, %w {axis = -1 : i64, rmsNorm}
       : tensor<8x16xf16>, tensor<8xf16> -> tensor<8x16xf16>
    tt.return
  }
}

// -----

module {
  tt.func @fused_global_pool_takes_no_window(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{global_average takes no window}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>, fusedPool = #pim.pool_spec<kind = global_average, window = #pim.window<kernel = [2, 2], strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]>>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @fused_windowed_pool_needs_window(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{a windowed pool requires a window}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>, fusedPool = #pim.pool_spec<kind = max>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// The LUT window is one triple. Two thirds of it would address the wrong
// segment of the table, which is wrong numerics with nothing to report it.
module {
  tt.func @lut_window_must_be_complete(%x: tensor<1x32xf16>) {
    // expected-error @below {{describe one window and must be given together}}
    %y = pim.lut %x {kind = #pim.activation<exp>, flpMinExp = 9 : i64} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// 0 selects the regular table and 4 the reciprocal one; anything else is a
// table the unit does not have.
module {
  tt.func @lut_special_operators_out_of_range(%x: tensor<1x32xf16>) {
    // expected-error @below {{specialOperators is 0 (regular) or 4 (reciprocal); got 3}}
    %y = pim.lut %x {kind = #pim.activation<exp>, specialOperators = 3 : i64} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// 144 fp16 segments is the whole table the activation unit can address.
module {
  tt.func @lut_table_wrong_size(%x: tensor<1x32xf16>, %t: tensor<64xf16>) {
    // expected-error @below {{table must be 144 f16 entries (288 bytes); got 64 x 'f16'}}
    %y = pim.lut %x, %t {kind = #pim.activation<silu>} : tensor<1x32xf16>, tensor<64xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// A grouped reduction collapses each group to one value, so the group size has
// to divide the axis it walks.
module {
  tt.func @global_pool_uneven_groups(%x: tensor<1x4096xf16>) {
    // expected-error @below {{is not divisible by groupSize 100}}
    %m = pim.global_pool %x {groupSize = 100 : i64, kind = #pim.pool_kind<absmax>} : tensor<1x4096xf16> -> tensor<1x40xf16>
    tt.return
  }
}

// -----

module {
  tt.func @global_pool_result_shape(%x: tensor<1x4096xf16>) {
    // expected-error @below {{result shape [1, 64] must be [1, 32]}}
    %m = pim.global_pool %x {groupSize = 128 : i64, kind = #pim.pool_kind<absmax>} : tensor<1x4096xf16> -> tensor<1x64xf16>
    tt.return
  }
}

// -----

// The three rescale constants are three different widths on purpose: a 32-bit
// bias, a 16-bit scale, an 8-bit shift.
module {
  tt.func @fpsu_bias_must_be_f32(%x: tensor<1x32xf16>, %b: tensor<1x32xf16>,
                                 %s: tensor<1x32xf16>) {
    // expected-error @below {{bias must be f32; got 'f16'}}
    %y = pim.fpsu_scale %x, %b, %s {shift = 14 : i8, spec = #pim.fpsu_spec<mode = fixed_point>} : tensor<1x32xf16>, tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x32xi8>
    tt.return
  }
}

// -----

// A format conversion has nothing to multiply by; a multiply without a
// multiplier would produce garbage rather than fail.
module {
  tt.func @kantor_fp2int_takes_no_rhs(%x: tensor<1x32xf16>, %m: tensor<1x32xf16>) {
    // expected-error @below {{mode fp2int_converter takes no rhs operand}}
    %q = pim.kantor %x rhs %m {spec = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>} : tensor<1x32xf16> rhs tensor<1x32xf16> -> tensor<1x32xi8>
    tt.return
  }
}

// -----

module {
  tt.func @kantor_multiply_needs_rhs(%x: tensor<1x32xf16>) {
    // expected-error @below {{mode elementwise_mul_fp16 requires a rhs operand}}
    %p = pim.kantor %x {spec = #pim.kantor_spec<mode = elementwise_mul_fp16, cardValue = 5>} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// One physical block can only be configured once.
module {
  tt.func @kantor_block_configured_twice(%x: tensor<1x32xf16>, %m: tensor<1x32xf16>) {
    // expected-error @+2 {{block A is configured more than once}}
    %p = pim.kantor %x rhs %m
        {spec = #pim.kantor_spec<mode = elementwise_mul_fp16, cardValue = 5, blocks = [#pim.kantor_block<id = "A", mode = elementwise_mul_fp16>, #pim.kantor_block<id = "A", mode = off>]>}
        : tensor<1x32xf16>, tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// Two element widths exist on the weight path and no others.
module {
  tt.func @weight_bits_must_be_4_or_8(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{elemBits is 4 (model weight) or 8 (activation used as a weight); got 3}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 3>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// A model weight is int4; the role and the width have to agree, or the buffer
// comes out the wrong size for what the reader expects of the role.
module {
  tt.func @model_weight_is_int4(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{a model weight is int4; got elemBits = 8}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 8>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// The subnormal guard is undone by a shift, so only a power of two can be
// expressed.
module {
  tt.func @sf_multiplier_is_power_of_two(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{sfMultiplier must be a power of two; got 3}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4, sfMultiplier = 3>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// The digest is a sha256 in hex; a raw 32-byte digest would be half this.
module {
  tt.func @content_hash_is_hex_sha256(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{contentHash must be a 64-character hex digest; got 8 characters}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4, contentHash = "deadbeef">} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @content_hash_is_lowercase(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{contentHash must be lowercase hexadecimal}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4, contentHash = "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef">} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// The binding and `transposeB` state the same fact; both are kept for now, so
// they have to agree rather than leave a reader to pick one.
module {
  tt.func @transpose_binding_needs_transposeB(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>) {
    // expected-error @below {{weightBinding format weights_transpose must agree with transposeB}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weights_transpose, role = model_weight, elemBits = 4>} : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// A placeholder is not really quantized, so a range would describe a bound that
// is never applied -- and a consumer reading it would emit quantization fields
// for a tensor that has none.
module {
  tt.func @transparent_takes_no_range(%x: tensor<4x8xf16>, %s: tensor<1xf16>) {
    // expected-error @below {{a transparent spec is a placeholder and takes no range}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, role = transparent, range = [-1.0, 1.0], spc = false>} : tensor<4x8xf16>, tensor<1xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// The clamped role exists because the element type cannot express the bound, so
// leaving the bound out defeats the point of the role.
module {
  tt.func @fp_clamp_needs_a_range(%x: tensor<4x8xf16>, %s: tensor<1xf16>) {
    // expected-error @below {{role fp_clamp needs an explicit range}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, role = fp_clamp, spc = false>} : tensor<4x8xf16>, tensor<1xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// It clamps to fp16's own range; a different bound is a different operation.
module {
  tt.func @fp_clamp_bound_is_fp16_max(%x: tensor<4x8xf16>, %s: tensor<1xf16>) {
    // expected-error @below {{clamps to fp16's range}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, role = fp_clamp, range = [-1.0e+03, 1.0e+03], spc = false>} : tensor<4x8xf16>, tensor<1xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

module {
  tt.func @range_is_min_max(%x: tensor<4x8xf16>, %s: tensor<1xf16>) {
    // expected-error @below {{range is [min, max]; got 3 entries}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, range = [-1.0, 0.0, 1.0], spc = false>} : tensor<4x8xf16>, tensor<1xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

module {
  tt.func @range_min_not_above_max(%x: tensor<4x8xf16>, %s: tensor<1xf16>) {
    // expected-error @below {{range min must not exceed max}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, range = [1.0, -1.0], spc = false>} : tensor<4x8xf16>, tensor<1xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// The two contraction forms are mutually exclusive: a nested block that also
// carries a flat flag describes a fusion the graph format cannot write.
module {
  tt.func @named_takes_no_flag(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{form named takes no flagName}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, contraction = #pim.contraction<form = named, blockName = "fused_Silu_act", innerOp = "Lut", actKind = "Silu", flagName = "dq_contraction">} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @flat_takes_no_block(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{form flat takes no blockName/innerOp/actKind}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, contraction = #pim.contraction<form = flat, flagName = "dq_contraction", blockName = "fused_Silu_act">} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @named_needs_its_fields(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{form named needs blockName, innerOp and actKind}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, contraction = #pim.contraction<form = named, blockName = "fused_Silu_act">} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @flat_needs_its_flag(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{form flat needs flagName}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, contraction = #pim.contraction<form = flat>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// Indices address rows. A float index would have to be rounded somewhere, and
// wherever that happened would be a silent decision about which row is read.
module {
  tt.func @gather_indices_must_be_integers(%t: tensor<100x8xf16>, %ids: tensor<1x5xf16>) {
    // expected-error @below {{indices must be an integer type; got 'f16'}}
    %y = pim.gather %t, %ids : tensor<100x8xf16>, tensor<1x5xf16> -> tensor<1x5x8xf16>
    tt.return
  }
}

// -----

module {
  tt.func @gather_result_shape(%t: tensor<100x8xf16>, %ids: tensor<1x5xi32>) {
    // expected-error @below {{must be the indices shape followed by the table's row shape}}
    %y = pim.gather %t, %ids : tensor<100x8xf16>, tensor<1x5xi32> -> tensor<1x5x4xf16>
    tt.return
  }
}

// -----

// A lookup copies rows; it does not convert them.
module {
  tt.func @gather_keeps_element_type(%t: tensor<100x8xf16>, %ids: tensor<1x5xi32>) {
    // expected-error @below {{a lookup copies rows, so the element type is the table's}}
    %y = pim.gather %t, %ids : tensor<100x8xf16>, tensor<1x5xi32> -> tensor<1x5x8xf32>
    tt.return
  }
}

// -----

// An elementwise op combines operands, so one is not an operation -- and the
// accessors the rest of this dialect uses would read out of bounds.
module {
  tt.func @eltwise_needs_two_operands(%a: tensor<4x8xf16>) {
    // expected-error @below {{takes at least two operands; got 1}}
    %y = pim.eltwise %a {kind = #pim.eltwise<add>, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>} : tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// A short per-slot list would silently leave the tail slots on slot 0's
// scaling, which is a different operation from the one written down.
module {
  tt.func @per_slot_list_covers_every_slot(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>, %c: tensor<4x8xf16>) {
    // expected-error @below {{perSlotDatapath has 2 entries for 3 operands}}
    %y = pim.eltwise %a, %b, %c {kind = #pim.eltwise<add>, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, perSlotDatapath = [#pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>]} : tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// RMS normalization divides by the root mean square and subtracts no mean, so
// there is no shift for a bias to apply.
module {
  tt.func @rms_norm_takes_no_bias(%x: tensor<4x16xf16>, %w: tensor<16xi8>, %b: tensor<16xf16>) {
    // expected-error @below {{rmsNorm subtracts no mean, so it takes no bias}}
    %y = pim.normalize %x, %w bias %b {axis = -1 : i64, rmsNorm} : tensor<4x16xf16>, tensor<16xi8> bias tensor<16xf16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// The epsilon buffer holds one value; a wider shape would mean a per-element
// epsilon, which the vector unit does not do.
module {
  tt.func @epsilon_is_one_value(%x: tensor<4x16xf16>, %e: tensor<4xf32>) {
    // expected-error @below {{epsilon is a single value; got 4 elements}}
    %y = pim.normalize %x eps %e {axis = -1 : i64, rmsNorm} : tensor<4x16xf16> eps tensor<4xf32> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

module {
  tt.func @epsilon_is_f32(%x: tensor<4x16xf16>, %e: tensor<1xf16>) {
    // expected-error @below {{epsilon is read as f32; got 'f16'}}
    %y = pim.normalize %x eps %e {axis = -1 : i64, rmsNorm} : tensor<4x16xf16> eps tensor<1xf16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// Normalization runs on the vector unit; an empty datapath field set is what
// says so. Filling one in emits keys the reference graph does not have.
module {
  tt.func @normalize_carries_no_kantor(%x: tensor<4x16xf16>) {
    // expected-error @below {{must not carry kantor}}
    %y = pim.normalize %x {axis = -1 : i64, rmsNorm, kantor = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>} : tensor<4x16xf16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// Masking adds a large negative number so the following softmax drives that
// position to zero. A boolean mask would have to be converted somewhere, and
// wherever that happened would decide the masked value -- which is the
// numerics, not a detail.
module {
  tt.func @mask_is_additive_not_boolean(%s: tensor<1x4x1x8xf16>, %m: tensor<1x1x1x8xi1>) {
    // expected-error @below {{the mask is additive (fp16), not boolean}}
    %y = pim.mask %s, %m : tensor<1x4x1x8xf16>, tensor<1x1x1x8xi1> -> tensor<1x4x1x8xf16>
    tt.return
  }
}

// -----

// A square mask is not a vector one; saying `vector` about it describes a
// geometry the tensor does not have.
module {
  tt.func @vector_layout_needs_a_row(%s: tensor<1x4x8x8xf16>, %m: tensor<1x1x8x8xf16>) {
    // expected-error @below {{layout vector masks a single query position}}
    %y = pim.mask %s, %m : tensor<1x4x8x8xf16>, tensor<1x1x8x8xf16> -> tensor<1x4x8x8xf16>
    tt.return
  }
}

// -----

module {
  tt.func @causal_tril_needs_a_square(%s: tensor<1x4x1x8xf16>, %m: tensor<1x1x1x8xf16>) {
    // expected-error @below {{layout causal_tril masks a square}}
    %y = pim.mask %s, %m {layout = #pim.mask_layout<causal_tril>} : tensor<1x4x1x8xf16>, tensor<1x1x1x8xf16> -> tensor<1x4x1x8xf16>
    tt.return
  }
}

// -----

// Residency and the binding describe the same operand from two sides. A
// disagreement means one is wrong, and a reader cannot tell which.
module {
  tt.func @kv_residency_needs_activation_role(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{weightBinding must say role = activation_as_weight}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>, stationarity = #pim.stationarity<kv>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @weight_residency_needs_model_weight(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{weightBinding must say role = model_weight}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, stationarity = #pim.stationarity<weight>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// Prefill's activation-by-activation multiply uses no weight path, so a binding
// would describe a path it does not take.
module {
  tt.func @activation_residency_takes_no_binding(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{it takes no weightBinding}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>, stationarity = #pim.stationarity<activation>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// `bIsActivation` covers two of the three cases; while both exist they must
// agree.
module {
  tt.func @flag_and_residency_agree(%a: tensor<4x8xi8>, %b: tensor<4x8xi8>) {
    // expected-error @below {{bIsActivation and stationarity disagree}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weights_transpose, role = activation_as_weight, elemBits = 8>, stationarity = #pim.stationarity<kv>, transposeB} : tensor<4x8xi8>, tensor<4x8xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// The accumulator dequantizes a group at a time, so it has to use the same
// grouping the weight was quantized at. Two different group sizes would fold in
// a scale belonging to a different run of elements.
module {
  tt.func @group_dequant_matches_the_weight_grouping(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{groupSize 64 must equal the weight's groupSize 128}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, groupDequantAccum = true, groupSize = 64>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @group_dequant_needs_a_binding(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{it needs a weightBinding}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, groupDequantAccum = true, groupSize = 128>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// A group size with nothing to apply it to describes a grouping the
// accumulator does not perform.
module {
  tt.func @group_size_needs_the_flag(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{groupSize is only meaningful with groupDequantAccum}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, groupSize = 128>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @group_dequant_needs_a_size(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    // expected-error @below {{needs a positive groupSize}}
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, groupDequantAccum = true>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4, groupSize = 0>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

// `onthefly` says no layer is produced, so a purpose that does produce one
// contradicts it. One of the two was set by mistake, and the op cannot say
// which -- so it refuses rather than picking one.
module {
  tt.func @onthefly_contradicts_the_purpose(%x: tensor<1x4x2x4xf16>) {
    // expected-error @below {{which does produce one}}
    %t = pim.transpose %x {axes = array<i64: 0, 2, 1, 3>, onthefly, purpose = #pim.transpose_purpose<purpose = tensor_transpose>}
       : tensor<1x4x2x4xf16> -> tensor<1x2x4x4xf16>
    tt.return
  }
}

// -----

// A tensor transpose permutes axes; it does not walk a card. A card value here
// would describe an addressing mode nothing performs.
module {
  tt.func @transpose_carries_no_card(%x: tensor<1x4x2x4xf16>) {
    // expected-error @below {{carries no cardValue}}
    %t = pim.transpose %x {axes = array<i64: 0, 2, 1, 3>, purpose = #pim.transpose_purpose<purpose = tensor_transpose, cardValue = 1>}
       : tensor<1x4x2x4xf16> -> tensor<1x2x4x4xf16>
    tt.return
  }
}

// -----

module {
  tt.func @card_cannot_be_negative(%x: tensor<1x4x2x4xf16>) {
    // expected-error @below {{cannot be negative}}
    %t = pim.transpose %x {axes = array<i64: 0, 2, 1, 3>, purpose = #pim.transpose_purpose<purpose = layout_reorder, cardValue = -1>}
       : tensor<1x4x2x4xf16> -> tensor<1x2x4x4xf16>
    tt.return
  }
}

// -----

// An extension digit that disagrees with the element type describes a layout no
// reader can act on: the digit and the type are two spellings of the same fact.
module {
  tt.func @extension_must_match_the_type(%x: tensor<1x16xf16>) {
    // expected-error @below {{implies 3}}
    %a = pim.convert %x {srcExtension = 1 : i64} : tensor<1x16xf16> -> tensor<1x16xi8>
    tt.return
  }
}

// -----

// 两侧元素类型相同就不是一次转换。扩展位由元素类型决定（f16→3、i8→1），
// 类型不变时扩展位也不变，所以"只改扩展位"不是一个真实的场景。
module {
  tt.func @convert_needs_a_real_conversion(%x: tensor<1x16xf16>) {
    // expected-error @below {{not a conversion}}
    %a = pim.convert %x : tensor<1x16xf16> -> tensor<1x16xf16>
    tt.return
  }
}

// -----

// The revision belongs to the module, so putting it on an op inside one is a
// mistake about scope rather than about the value.
module {
  tt.func @rtl_version_is_not_for_ops(%x: tensor<1x16xf16>) {
    // expected-error @below {{expected only on `module` ops}}
    %a = pim.convert %x {"pim.rtl-version" = "1.4"} : tensor<1x16xf16> -> tensor<1x16xi8>
    tt.return
  }
}

// -----

// A scatter without an index has no destination, and the op cannot invent one.
module {
  tt.func @scatter_needs_an_index(%k: tensor<1x8x128xi8>, %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>, %pos: i32) {
    // expected-error @below {{needs an indices operand}}
    pim.kv_cache %k, %cache, %pos {layer = 0 : i64, isKey}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32
    tt.return
  }
}

// -----

// A contiguous walk has nothing to index with, so an index here would be read
// by nobody.
module {
  tt.func @range_write_has_no_index(%k: tensor<1x8x128xi8>, %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>, %pos: i32, %slot: tensor<1xi16>) {
    // expected-error @below {{nothing to index with}}
    pim.kv_cache %k, %cache, %pos[%slot] {layer = 0 : i64, isKey, mode = #pim.kv_mode<range_write>}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32 [tensor<1xi16>]
    tt.return
  }
}

// -----

// A cache row the hardware cannot reach is not a narrower index problem: i32
// would name rows the cache does not have.
module {
  tt.func @scatter_index_is_i16(%k: tensor<1x8x128xi8>, %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>, %pos: i32, %slot: tensor<1xi32>) {
    // expected-error @below {{scatter indices must be i16}}
    pim.kv_cache %k, %cache, %pos[%slot] {layer = 0 : i64, isKey}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32 [tensor<1xi32>]
    tt.return
  }
}

// -----

// `isRead` is the older spelling of `mode = read`. Two spellings of one fact
// have to agree, or a reader cannot tell which was meant.
module {
  tt.func @read_keyword_and_mode_disagree(%k: tensor<1x8x128xi8>, %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>, %pos: i32) {
    // expected-error @below {{disagree}}
    pim.kv_cache read %k, %cache, %pos {layer = 0 : i64, mode = #pim.kv_mode<range_write>}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32
    tt.return
  }
}

// -----

// Only two RoPE chains exist, and they use exactly two card values: 0 for the Q
// chain (result stays fp16 for the following dynamic quantize) and 3 for the K
// chain (requantized on its way into the cache). Any other integer names an
// addressing mode the hardware does not have -- and since this value used to be
// a bare `pim.kantor-mode` string with no verifier at all, a typo silently
// became card value 0 and the K chain's requantization vanished.
module {
  tt.func @rope_rejects_a_card_value_that_is_neither_chain(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>) {
    // expected-error @below {{tailCardValue must be 0 (Q chain, fp16 tail) or 3 (K chain, requantized into the cache), got 5}}
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64, tailCardValue = 5 : i64}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.return
  }
}


// -----

// A weight-role quant spec states the range the weight was stored in; the
// binding states the element width. A range the width cannot hold means one of
// them is wrong, and the rescale would then apply a bound that never fires.
module {
  tt.func @weight_range_must_fit_the_element_width(%a: tensor<1x32xi8>, %b: tensor<32x16xi8>) {
    // expected-error @below {{role weight states range [-1.280000e+02, 1.270000e+02], but elemBits = 4 holds [-8.000000e+00, 7.000000e+00]}}
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                 scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1,
                                                             role = weight, range = [-128.0, 127.0]>,
                                 kantorBlocks = [#pim.kantor_block<id = "A", mode = fp2int_converter>]>,
        weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>}
       : tensor<1x32xi8>, tensor<32x16xi8> -> tensor<1x16xi8>
    tt.return
  }
}

// -----

// The operand attributes and the spec describe one quantization. Checking
// only one of them lets the two disagree: the verifier would pass on 64 while
// the expansion reads 128, and the declared scale shape becomes an orphan.
module {
  tt.func @dynamic_quant_spec_must_agree_with_the_op(%x: tensor<1x4096xf16>) {
    // expected-error @below {{spec groupSize is 128 but the op says 64}}
    %q, %s = pim.dynamic_quant %x
       {groupSize = 64 : i64, axis = 1 : i64,
        spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>}
       : tensor<1x4096xf16> -> tensor<1x4096xi8>, tensor<64xf16>
    tt.return
  }
}

// -----

// The float-to-fixed conversion is a scaling by 2^-shift. Without the shift
// the result is undefined, and a lowering that hardcoded 256 would keep
// compiling while ignoring the IR.
module {
  tt.func @fp2int_needs_the_shift(%x: tensor<1x32xf16>) {
    // expected-error @below {{the float-to-fixed conversion needs the shift attribute}}
    %q = pim.kantor %x {spec = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>}
        : tensor<1x32xf16> -> tensor<1x32xi8>
    tt.return
  }
}

// -----

// A quantizer computes; it does not move data. These three carry a `unit` but
// used to skip the unit check, so `unit = #pim.unit<dma>` was accepted on the
// ops whose whole job is arithmetic.
module {
  tt.func @quantize_cannot_run_on_dma(%x: tensor<4x4xf32>, %s: tensor<1xf32>) {
    // expected-error @below {{cannot run on the DMA unit}}
    %q = pim.quantize %x, %s
       {spec = #pim.quant_spec<granularity = per_tensor, spc = false>, unit = #pim.unit<dma>}
       : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @dynamic_quant_cannot_run_on_dma(%x: tensor<1x256xf16>) {
    // expected-error @below {{cannot run on the DMA unit}}
    %q, %s = pim.dynamic_quant %x
       {groupSize = 128 : i64, axis = 1 : i64,
        spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>,
        unit = #pim.unit<dma>}
       : tensor<1x256xf16> -> tensor<1x256xi8>, tensor<2xf16>
    tt.return
  }
}

// -----

module {
  tt.func @dequantize_cannot_run_on_dma(%x: tensor<4x4xi8>, %s: tensor<1xf16>) {
    // expected-error @below {{cannot run on the DMA unit}}
    %d = pim.dequantize %x, %s
       {spec = #pim.quant_spec<granularity = per_tensor, spc = false>, unit = #pim.unit<dma>}
       : tensor<4x4xi8>, tensor<1xf16> -> tensor<4x4xf16>
    tt.return
  }
}

// -----

// 散写是 mode 的默认值，而它正是最常用的写入形态。类型不一致与容量不足
// 两条必须对它生效——只在 range_write 上拦会让默认形态悄悄绕过。
module {
  tt.func @scatter_rejects_mismatched_element_type(%k: tensor<1x8x128xf16>, %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>, %pos: i32, %slot: tensor<1xi16>) {
    // expected-error @below {{cache element type 'i8' must match value element type 'f16'}}
    pim.kv_cache %k, %cache, %pos[%slot] {layer = 0 : i64, isKey}
        : tensor<1x8x128xf16>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32 [tensor<1xi16>]
    tt.return
  }
}

// -----

// 缓存装的是 value 的多个步，容量小于单次写入就是放不下。
module {
  tt.func @scatter_rejects_a_cache_smaller_than_the_value(%k: tensor<8x128xi8>, %cache: !pim.memdesc<4x128xi8, #pim.mram>, %pos: i32, %slot: tensor<1xi16>) {
    // expected-error @below {{cache holds 512 elements, fewer than the 1024 being moved}}
    pim.kv_cache %k, %cache, %pos[%slot] {layer = 0 : i64, isKey}
        : tensor<8x128xi8>, !pim.memdesc<4x128xi8, #pim.mram>, i32 [tensor<1xi16>]
    tt.return
  }
}

// -----

// 6 个子块的顺序是硬件语义的一部分：乱序会让 GML 侧按位置展开时整块错位，
// 所以只查"名字在集合里、不重复"不够。
module {
  tt.func @rope_rejects_subblocks_out_of_order(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>) {
    // expected-error @below {{subBlocks 顺序不对}}
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64,
       subBlocks = ["Llama2Activation_Sin", "Llama2Activation_Add_Cos",
                    "Llama2Activation_Add_Sin", "Llama2Activation_Sin",
                    "Llama2Activation_Cos", "Llama2Activation_Cos"]}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.return
  }
}

// -----

// 广播是 broadcastSpec 的事，不是子块名。`Sin_Broadcast` 这种名字会让
// 读回侧少掉一个真正的子块而不报错。
module {
  tt.func @rope_rejects_a_broadcast_subblock_name(
      %x: tensor<1x32x16x128xf16>, %c: tensor<1x1x16x128xf16>,
      %s: tensor<1x1x16x128xf16>) {
    // expected-error @below {{subBlocks 顺序不对}}
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64,
       subBlocks = ["Llama2Activation_Add_Cos", "Llama2Activation_Add_Sin",
                    "Llama2Activation_Sin", "Llama2Activation_Sin_Broadcast",
                    "Llama2Activation_Cos", "Llama2Activation_Cos_Broadcast"]}
       : tensor<1x32x16x128xf16>, tensor<1x1x16x128xf16>, tensor<1x1x16x128xf16>
       -> tensor<1x32x16x128xf16>
    tt.return
  }
}

// -----

// 逐通道 / 逐组是量化决策本身，必须与 granularity 一致：per_tensor 没有
// 通道可分，开着 spc 就是自相矛盾。
module {
  tt.func @spc_disagrees_with_per_tensor(%x: tensor<4x8xf16>, %s: tensor<1xf16>) {
    // expected-error @below {{disagrees with granularity per_tensor}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, spc = true>} : tensor<4x8xf16>, tensor<1xf16> -> tensor<4x8xi8>
    tt.return
  }
}

// -----

// 只有 per_group 才谈得上分组，per_channel 开着 spg 没有对应的布局。
module {
  tt.func @spg_disagrees_with_per_channel(%x: tensor<4x8xf16>, %s: tensor<8xf16>) {
    // expected-error @below {{disagrees with granularity per_channel}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_channel, axis = 1, spg = true>} : tensor<4x8xf16>, tensor<8xf16> -> tensor<4x8xi8>
    tt.return
  }
}
