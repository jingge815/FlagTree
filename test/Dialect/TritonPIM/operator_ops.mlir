// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// Round-trip of the operator-level primitives: the granularity the target's
// graph format works in, one level above the tile-level ops in `ops.mlir`.

// CHECK-LABEL: @quantize_dequantize
module {
  tt.func @quantize_dequantize(%x: tensor<128x256xf32>, %s: tensor<1xf32>,
                               %ps: tensor<256xf32>, %zp: tensor<256xi8>) {
    // CHECK: pim.quantize %{{.*}}, %{{.*}} {spec = #pim.quant_spec<granularity = per_tensor>}
    %q = pim.quantize %x, %s {spec = #pim.quant_spec<granularity = per_tensor>}
       : tensor<128x256xf32>, tensor<1xf32> -> tensor<128x256xi8>
    // CHECK: pim.quantize %{{.*}}, %{{.*}}, %{{.*}} {spec = #pim.quant_spec<granularity = per_channel, axis = 1>}
    %qc = pim.quantize %x, %ps, %zp
        {spec = #pim.quant_spec<granularity = per_channel, axis = 1>}
        : tensor<128x256xf32>, tensor<256xf32>, tensor<256xi8> -> tensor<128x256xi8>
    // CHECK: pim.dequantize
    %d = pim.dequantize %q, %s {spec = #pim.quant_spec<granularity = per_tensor>}
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
       {spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128>}
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
