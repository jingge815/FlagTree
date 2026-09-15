// RUN: triton-opt %s -split-input-file -verify-diagnostics

// The verifiers on the operator-level ops exist to stop malformed IR before it
// reaches the graph hand-off, where a bad datapath or a mismatched scale count
// would be much harder to trace. These check that each one actually fires.

module {
  tt.func @per_tensor_takes_no_axis(%x: tensor<4x4xf32>, %s: tensor<1xf32>) {
    // expected-error @below {{per_tensor takes no axis or groupSize}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor, axis = 1>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xi8>
    tt.return
  }
}

// -----

module {
  tt.func @per_group_needs_group_size(%x: tensor<4x16xf32>, %s: tensor<4xf32>) {
    // expected-error @below {{per_group requires a positive groupSize}}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_group, axis = 1>} : tensor<4x16xf32>, tensor<4xf32> -> tensor<4x16xi8>
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
