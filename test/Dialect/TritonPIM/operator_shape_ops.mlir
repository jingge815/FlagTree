// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// Round-trip of the activation, normalization, shape and on-chip staging
// primitives.

// CHECK-LABEL: @activations
module {
  tt.func @activations(%x: tensor<128x256xi16>, %t: tensor<144xf16>) {
    // CHECK: pim.lut %{{.*}} {kind = #pim.activation<silu>, unit = #pim.unit<cstl>}
    %a = pim.lut %x {kind = #pim.activation<silu>, unit = #pim.unit<cstl>}
       : tensor<128x256xi16> -> tensor<128x256xi8>
    // CHECK: pim.lut %{{.*}}, %{{.*}} {clipMax = 6.000000e+00 : f64, clipMin = 0.000000e+00 : f64, kind = #pim.activation<relu_x>}
    %b = pim.lut %x, %t
       {kind = #pim.activation<relu_x>, clipMin = 0.0 : f64, clipMax = 6.0 : f64}
       : tensor<128x256xi16>, tensor<144xf16> -> tensor<128x256xi8>
    // CHECK: #pim.activation<leaky_relu>
    %c = pim.lut %x {kind = #pim.activation<leaky_relu>, alpha = 1.000000e-01 : f64}
       : tensor<128x256xi16> -> tensor<128x256xi8>
    // CHECK: mode = #pim.lut_mode<even>
    %d = pim.lut %x {kind = #pim.activation<sigmoid>, mode = #pim.lut_mode<even>}
       : tensor<128x256xi16> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// An activation folded into its producer, which is how the target's graph format
// stores it.
// CHECK-LABEL: @fused_activation
module {
  tt.func @fused_activation(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>,
                            %l: tensor<128x256xi8>) {
    // CHECK: pim.matmul{{.*}}activation = #pim.act_spec<kind = silu>
    %m = pim.matmul %a, %b
       {activation = #pim.act_spec<kind = silu>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    // CHECK: pim.eltwise{{.*}}activation = #pim.act_spec<kind = relu, mode = even>
    %e = pim.eltwise %m, %l
       {kind = #pim.eltwise<add>,
        activation = #pim.act_spec<kind = relu, mode = even>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>,
        unit = #pim.unit<vpu>}
       : tensor<128x256xi8>, tensor<128x256xi8> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// CHECK-LABEL: @norm_and_softmax
module {
  tt.func @norm_and_softmax(%x: tensor<128x4096xf16>, %w: tensor<4096xf16>,
                            %s: tensor<32x128x128xf16>) {
    // CHECK: pim.normalize %{{.*}}, %{{.*}} {axis = -1 : i64, rmsNorm, unit = #pim.unit<vpu>}
    %n = pim.normalize %x, %w {axis = -1 : i64, rmsNorm, unit = #pim.unit<vpu>}
       : tensor<128x4096xf16>, tensor<4096xf16> -> tensor<128x4096xf16>
    // CHECK: pim.softmax %{{.*}} {axis = -1 : i64
    %p = pim.softmax %s {axis = -1 : i64, unit = #pim.unit<cstl>}
       : tensor<32x128x128xf16> -> tensor<32x128x128xf16>
    // CHECK: pim.reduce_axis %{{.*}} {axis = 1 : i64, kind = #pim.eltwise<max>
    %r = pim.reduce_axis %x {kind = #pim.eltwise<max>, axis = 1 : i64, unit = #pim.unit<vpu>}
       : tensor<128x4096xf16> -> tensor<128x1xf16>
    tt.return
  }
}

// -----

// CHECK-LABEL: @rope
module {
  tt.func @rope(%x: tensor<1x32x128x128xf16>, %cos: tensor<128x128xf16>,
                %sin: tensor<128x128xf16>) {
    // CHECK: pim.rope %{{.*}}, %{{.*}}, %{{.*}} {numHeads = 32 : i64, transposeInput
    %y = pim.rope %x, %cos, %sin
       {numHeads = 32 : i64, transposeInput, unit = #pim.unit<cstl>}
       : tensor<1x32x128x128xf16>, tensor<128x128xf16>, tensor<128x128xf16>
       -> tensor<1x32x128x128xf16>
    tt.return
  }
}

// -----

// CHECK-LABEL: @shape_ops
module {
  tt.func @shape_ops(%x: tensor<1x128x32x128xf16>, %qkv: tensor<128x12288xf16>,
                     %cached: tensor<1x32x127x128xf16>, %new: tensor<1x32x1x128xf16>) {
    // CHECK: pim.param {name = "layers.0.self_attn.q_proj.weight"} : tensor<4096x4096xi8>
    %w = pim.param {name = "layers.0.self_attn.q_proj.weight"} : tensor<4096x4096xi8>
    // CHECK: pim.transpose %{{.*}} {axes = array<i64: 0, 2, 1, 3>}
    %t = pim.transpose %x {axes = array<i64: 0, 2, 1, 3>}
       : tensor<1x128x32x128xf16> -> tensor<1x32x128x128xf16>
    // CHECK: pim.reshape
    %r = pim.reshape %t : tensor<1x32x128x128xf16> -> tensor<128x4096xf16>
    // CHECK: pim.split %{{.*}} {axis = 1 : i64}
    %q, %k, %v = pim.split %qkv {axis = 1 : i64}
       : tensor<128x12288xf16> -> tensor<128x4096xf16>, tensor<128x4096xf16>, tensor<128x4096xf16>
    // CHECK: pim.concat %{{.*}}, %{{.*}} {axis = 2 : i64}
    %kv = pim.concat %cached, %new {axis = 2 : i64}
        : tensor<1x32x127x128xf16>, tensor<1x32x1x128xf16> -> tensor<1x32x128x128xf16>
    tt.return
  }
}

// -----

// The three on-chip levels: L2 is shared by an NPU's units, L1 is private to
// one, and a weight expands on its way in.
// CHECK-LABEL: @on_chip_staging
module attributes {"pim.l2-bytes" = 1048576 : i64, "pim.l1-bytes" = 1048576 : i64} {
  tt.func @on_chip_staging(%packed: !pim.memdesc<131072xi8, #pim.mram>) {
    // CHECK: pim.buffer_alloc : !pim.memdesc<512x512xi8, #pim.l2>
    %l2 = pim.buffer_alloc : !pim.memdesc<512x512xi8, #pim.l2>
    // CHECK: pim.buffer_alloc {unit = #pim.unit<nmu>} : !pim.memdesc<512x512xi8, #pim.l1>
    %l1 = pim.buffer_alloc {unit = #pim.unit<nmu>} : !pim.memdesc<512x512xi8, #pim.l1>
    // CHECK: pim.decompress_weight %{{.*}} -> %{{.*}} {ratio = 2.000000e+00 : f64}
    pim.decompress_weight %packed -> %l2 {ratio = 2.0 : f64}
      : !pim.memdesc<131072xi8, #pim.mram> -> !pim.memdesc<512x512xi8, #pim.l2>
    // CHECK: pim.buffer_copy %{{.*}} -> %{{.*}} : !pim.memdesc<512x512xi8, #pim.l2> -> !pim.memdesc<512x512xi8, #pim.l1>
    pim.buffer_copy %l2 -> %l1
      : !pim.memdesc<512x512xi8, #pim.l2> -> !pim.memdesc<512x512xi8, #pim.l1>
    tt.return
  }
}

// -----

// CHECK-LABEL: @conv_and_pool
module {
  tt.func @conv_and_pool(%x: tensor<1x64x56x56xi8>, %w: tensor<64x64x3x3xi8>,
                         %b: tensor<64xi32>, %p: tensor<1x64x112x112xi8>) {
    // CHECK: pim.conv{{.*}}#pim.window<kernel = [3, 3]
    %y = pim.conv %x, %w, %b
       {window = #pim.window<kernel = [3, 3], strides = [1, 1],
                             pads = [1, 1, 1, 1], dilations = [1, 1]>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>,
        activation = #pim.act_spec<kind = relu>, unit = #pim.unit<nmu>}
       : tensor<1x64x56x56xi8>, tensor<64x64x3x3xi8>, tensor<64xi32> -> tensor<1x64x56x56xi8>
    // CHECK: pim.pool{{.*}}kind = #pim.pool<max>
    %q = pim.pool %p
       {kind = #pim.pool<max>,
        window = #pim.window<kernel = [3, 3], strides = [2, 2],
                             pads = [1, 1, 1, 1], dilations = [1, 1]>,
        unit = #pim.unit<vpu>}
       : tensor<1x64x112x112xi8> -> tensor<1x64x56x56xi8>
    // CHECK: kind = #pim.pool<global_average>
    %g = pim.pool %x {kind = #pim.pool<global_average>, unit = #pim.unit<vpu>}
       : tensor<1x64x56x56xi8> -> tensor<1x64x1x1xi8>
    tt.return
  }
}

// -----

// A convolution, its activation and a following pool in one node -- the shape
// the reference graph uses for the stem of a ResNet.
// CHECK-LABEL: @conv_relu_pool
module {
  tt.func @conv_relu_pool(%x: tensor<1x3x224x224xi8>, %w: tensor<64x3x7x7xi8>,
                          %b: tensor<64xi32>) {
    // CHECK: pim.conv{{.*}}activation = #pim.act_spec<kind = relu>
    // CHECK-SAME: fusedPool = #pim.pool_spec<kind = max
    %y = pim.conv %x, %w, %b
       {window = #pim.window<kernel = [7, 7], strides = [2, 2],
                             pads = [3, 3, 3, 3], dilations = [1, 1]>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                 kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>]>,
        activation = #pim.act_spec<kind = relu>,
        fusedPool = #pim.pool_spec<kind = max,
                     window = #pim.window<kernel = [3, 3], strides = [2, 2],
                                          pads = [1, 1, 1, 1], dilations = [1, 1]>>,
        unit = #pim.unit<nmu>}
       : tensor<1x3x224x224xi8>, tensor<64x3x7x7xi8>, tensor<64xi32>
       -> tensor<1x64x56x56xi8>
    // CHECK: fusedPool = #pim.pool_spec<kind = global_average>
    %g = pim.conv %x, %w, %b
       {window = #pim.window<kernel = [7, 7], strides = [2, 2],
                             pads = [3, 3, 3, 3], dilations = [1, 1]>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>,
        fusedPool = #pim.pool_spec<kind = global_average>}
       : tensor<1x3x224x224xi8>, tensor<64x3x7x7xi8>, tensor<64xi32>
       -> tensor<1x64x1x1xi8>
    tt.return
  }
}

// -----

// Each operand of an elementwise op is rescaled by its own datapath before the
// two meet -- the reference graph gives a two-input add a full FPSU
// configuration per input slot, not one shared between them.
// CHECK-LABEL: @eltwise_per_slot_datapath
module {
  tt.func @eltwise_per_slot_datapath(%x: tensor<128x256xi8>, %y: tensor<128x256xi8>) {
    // CHECK: pim.eltwise{{.*}}rhsDatapath = #pim.datapath<
    %z = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                 scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1>,
                                 kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>]>,
        rhsDatapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                    scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1>>,
        activation = #pim.act_spec<kind = relu>,
        unit = #pim.unit<vpu>}
       : tensor<128x256xi8>, tensor<128x256xi8> -> tensor<128x256xi8>
    // Two operands quantized alike share one datapath.
    // CHECK: pim.eltwise
    // CHECK-NOT: rhsDatapath
    %w = pim.eltwise %x, %y
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<128x256xi8>, tensor<128x256xi8> -> tensor<128x256xi8>
    tt.return
  }
}

// -----

// Attention pieces the target's graph format models as nodes of their own.
// Verified against a real llama2 W4A8 decode block: 32 `Mask` nodes (one per
// head) and 2 `KV_Cache_DMA` nodes (the key and value writebacks).
// CHECK-LABEL: @attention_primitives
module {
  tt.func @attention_primitives(%scores: tensor<32x1x128xf16>, %m: tensor<1x1x128xf16>,
                                %k: tensor<1x8x128xf16>,
                                %cache: !pim.memdesc<2048x8x128xf16, #pim.mram>,
                                %pos: i32, %qkv: tensor<1x4096xf16>) {
    // CHECK: pim.mask %{{.*}}, %{{.*}} {unit = #pim.unit<vpu>}
    %slot = arith.constant dense<[7]> : tensor<1xi16>
    %masked = pim.mask %scores, %m {unit = #pim.unit<vpu>}
        : tensor<32x1x128xf16>, tensor<1x1x128xf16> -> tensor<32x1x128xf16>
    // A scatter takes its destination from the index, so the index is not
    // optional in that mode -- which is the default, because a cache write
    // during decoding lands wherever the step counter says.
    // CHECK: pim.kv_cache %{{.*}}, %{{.*}}, %{{.*}}[%{{.*}}] {isKey, layer = 0 : i64}
    pim.kv_cache %k, %cache, %pos[%slot] {layer = 0 : i64, isKey}
        : tensor<1x8x128xf16>, !pim.memdesc<2048x8x128xf16, #pim.mram>, i32 [tensor<1xi16>]
    // CHECK: pim.split_heads %{{.*}} {axis = 1 : i64, numHeads = 32 : i64}
    %h:32 = pim.split_heads %qkv {axis = 1 : i64, numHeads = 32 : i64}
        : tensor<1x4096xf16> -> tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>, tensor<1x128xf16>
    tt.return
  }
}
