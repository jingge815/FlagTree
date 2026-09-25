// RUN: triton-opt %s -split-input-file -pim-fuse-activation | FileCheck %s

// The target's graph format keeps an activation inside its producer's node, so
// this fusion is a correctness requirement rather than an optimization.

// CHECK-LABEL: @matmul_lut
module {
  tt.func @matmul_lut(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>,
                      %o: tensor<128x256x!tt.ptr<i8>>) {
    // 融合同时写下两样：`activation` 记录折了什么，`contraction` 记录图格式
    // 怎么写它。表体与窗口留在主算子上——嵌套块只给融合命名，不重述数据通路。
    // CHECK: pim.matmul{{.*}}activation = #pim.act_spec<kind = silu>{{.*}}contraction = #pim.contraction<form = named, blockName = "fused_Silu_act", innerOp = "Lut", actKind = "Silu">
    // CHECK-NOT: pim.lut
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    %y = pim.lut %m {kind = #pim.activation<silu>}
       : tensor<128x256xi8> -> tensor<128x256xi8>
    tt.store %o, %y : tensor<128x256x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// A trailing pool folds into the same node, after the activation. This is the
// shape of the reference graph's ResNet stem.
// CHECK-LABEL: @conv_lut_pool
module {
  tt.func @conv_lut_pool(%x: tensor<1x64x64x64xi8>, %w: tensor<64x64x3x3xi8>,
                         %o: tensor<1x64x32x32x!tt.ptr<i8>>) {
    // CHECK: pim.conv
    // CHECK-SAME: activation = #pim.act_spec<kind = relu>
    // CHECK-SAME: fusedPool = #pim.pool_spec<kind = max
    // CHECK-SAME: -> tensor<1x64x32x32xi8>
    // CHECK-NOT: pim.lut
    // CHECK-NOT: pim.pool
    %c = pim.conv %x, %w
       {window = #pim.window<kernel = [3, 3], strides = [1, 1],
                             pads = [1, 1, 1, 1], dilations = [1, 1]>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<1x64x64x64xi8>, tensor<64x64x3x3xi8> -> tensor<1x64x64x64xi8>
    %r = pim.lut %c {kind = #pim.activation<relu>}
       : tensor<1x64x64x64xi8> -> tensor<1x64x64x64xi8>
    %p = pim.pool %r
       {kind = #pim.pool<max>,
        window = #pim.window<kernel = [3, 3], strides = [2, 2],
                             pads = [1, 1, 1, 1], dilations = [1, 1]>}
       : tensor<1x64x64x64xi8> -> tensor<1x64x32x32xi8>
    tt.store %o, %p : tensor<1x64x32x32x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// An elementwise add also holds a folded activation -- the reference graph has 16
// of these.
// CHECK-LABEL: @eltwise_lut
module {
  tt.func @eltwise_lut(%x: tensor<128x256xi8>, %y: tensor<128x256xi8>,
                       %o: tensor<128x256x!tt.ptr<i8>>) {
    // CHECK: pim.eltwise{{.*}}activation = #pim.act_spec<kind = relu>
    // CHECK-NOT: pim.lut
    %e = pim.eltwise %x, %y
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<128x256xi8>, tensor<128x256xi8> -> tensor<128x256xi8>
    %a = pim.lut %e {kind = #pim.activation<relu>}
       : tensor<128x256xi8> -> tensor<128x256xi8>
    tt.store %o, %a : tensor<128x256x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// The activation's parameters travel with it.
// CHECK-LABEL: @activation_parameters_survive
module {
  tt.func @activation_parameters_survive(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>,
                                         %o: tensor<128x256x!tt.ptr<i8>>) {
    // CHECK: activation = #pim.act_spec<kind = leaky_relu, mode = odd, alpha = 1.000000e-01
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    %y = pim.lut %m
       {kind = #pim.activation<leaky_relu>, mode = #pim.lut_mode<odd>,
        alpha = 1.000000e-01 : f64}
       : tensor<128x256xi8> -> tensor<128x256xi8>
    tt.store %o, %y : tensor<128x256x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// Fusing would strand the second reader of the pre-activation value, so the lut
// stays where it is.
// CHECK-LABEL: @two_readers_block_fusion
module {
  tt.func @two_readers_block_fusion(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>,
                                    %o1: tensor<128x256x!tt.ptr<i8>>,
                                    %o2: tensor<128x256x!tt.ptr<i8>>) {
    // CHECK: pim.matmul
    // CHECK-NOT: activation
    // CHECK: pim.lut
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    %y = pim.lut %m {kind = #pim.activation<silu>}
       : tensor<128x256xi8> -> tensor<128x256xi8>
    tt.store %o1, %y : tensor<128x256x!tt.ptr<i8>>
    tt.store %o2, %m : tensor<128x256x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// One node holds one activation, so a producer that already has one is left
// alone.
// CHECK-LABEL: @existing_activation_blocks_fusion
module {
  tt.func @existing_activation_blocks_fusion(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>,
                                             %o: tensor<128x256x!tt.ptr<i8>>) {
    // CHECK: pim.matmul{{.*}}activation = #pim.act_spec<kind = relu>
    // CHECK: pim.lut{{.*}}#pim.activation<silu>
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>,
        activation = #pim.act_spec<kind = relu>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    %y = pim.lut %m {kind = #pim.activation<silu>}
       : tensor<128x256xi8> -> tensor<128x256xi8>
    tt.store %o, %y : tensor<128x256x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// Only matmul/conv/eltwise nodes can hold an activation.
// CHECK-LABEL: @non_fusable_producer
module {
  tt.func @non_fusable_producer(%x: tensor<128x256xf16>, %o: tensor<128x256x!tt.ptr<i8>>) {
    // CHECK: pim.softmax
    // CHECK: pim.lut
    %s = pim.softmax %x {axis = -1 : i64} : tensor<128x256xf16> -> tensor<128x256xi8>
    %y = pim.lut %s {kind = #pim.activation<silu>}
       : tensor<128x256xi8> -> tensor<128x256xi8>
    tt.store %o, %y : tensor<128x256x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// Real CNN shapes work as long as the graph stays clear of `tt.store` and
// `tt.return` values, which carry a power-of-two element-count restriction of
// their own. An operator-level graph does not need either.
// CHECK-LABEL: @real_resnet_stem
module {
  tt.func @real_resnet_stem(%x: tensor<1x3x224x224xi8>, %w: tensor<64x3x7x7xi8>) {
    // CHECK: pim.conv
    // CHECK-SAME: activation = #pim.act_spec<kind = relu>
    // CHECK-SAME: fusedPool = #pim.pool_spec<kind = max
    // CHECK-SAME: -> tensor<1x64x56x56xi8>
    // CHECK-NOT: pim.lut
    %c = pim.conv %x, %w
       {window = #pim.window<kernel = [7, 7], strides = [2, 2],
                             pads = [3, 3, 3, 3], dilations = [1, 1]>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<1x3x224x224xi8>, tensor<64x3x7x7xi8> -> tensor<1x64x112x112xi8>
    %r = pim.lut %c {kind = #pim.activation<relu>}
       : tensor<1x64x112x112xi8> -> tensor<1x64x112x112xi8>
    %p = pim.pool %r
       {kind = #pim.pool<max>,
        window = #pim.window<kernel = [3, 3], strides = [2, 2],
                             pads = [1, 1, 1, 1], dilations = [1, 1]>}
       : tensor<1x64x112x112xi8> -> tensor<1x64x56x56xi8>
    tt.return
  }
}

// -----

// `silu` folds into the gate projection, and the gate projection is a matmul.
// Folding it into an elementwise producer would put a `fused_Silu_act`
// contraction on a node the graph format never carries one on -- a nested block
// the other side's parser reads as a different operator.
// CHECK-LABEL: @eltwise_keeps_silu_separate
module {
  tt.func @eltwise_keeps_silu_separate(%a: tensor<4x8xi8>, %b: tensor<4x8xi8>) {
    // CHECK: pim.eltwise
    // CHECK-NOT: contraction
    // CHECK: pim.lut
    // CHECK: kind = #pim.activation<silu>
    %s = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4x8xi8>, tensor<4x8xi8> -> tensor<4x8xi8>
    %y = pim.lut %s {kind = #pim.activation<silu>}
       : tensor<4x8xi8> -> tensor<4x8xi8>
    tt.return
  }
}

// -----

// A table folds too. The table is data, so it cannot live in an attribute: it
// moves onto the producer, which is where the target format keeps the gate
// projection's table -- the nested contraction block names the fusion, it does
// not restate the datapath.
// CHECK-LABEL: @matmul_lut_with_table
module {
  tt.func @matmul_lut_with_table(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>,
                                 %t: tensor<144xf16>,
                                 %o: tensor<128x256x!tt.ptr<i8>>) {
    // CHECK: pim.matmul {{.*}} lut {{.*}}activation = #pim.act_spec<kind = silu>{{.*}}contraction = #pim.contraction<form = named, blockName = "fused_Silu_act"
    // CHECK-NOT: pim.lut
    %m = pim.matmul %a, %b
       {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    %y = pim.lut %m, %t {kind = #pim.activation<silu>}
       : tensor<128x256xi8>, tensor<144xf16> -> tensor<128x256xi8>
    tt.store %o, %y : tensor<128x256x!tt.ptr<i8>>
    tt.return
  }
}


// -----

// rsqrt 属于 RMSNorm，不折进主算子——折了 RMSNorm 就不再是独立节点。
// CHECK-LABEL: @rsqrt_stays_separate
module {
  tt.func @rsqrt_stays_separate(%a: tensor<4x8xf16>, %b: tensor<8x4xf16>) {
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>} : tensor<4x8xf16>, tensor<8x4xf16> -> tensor<4x4xf16>
    // CHECK: pim.lut
    // CHECK-NOT: contraction
    %y = pim.lut %m {kind = #pim.activation<rsqrt>} : tensor<4x4xf16> -> tensor<4x4xf16>
    tt.return
  }
}
