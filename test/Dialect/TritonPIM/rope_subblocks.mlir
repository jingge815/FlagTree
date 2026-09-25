// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// `pim.rope` 的 6 个编号子块与广播语义必须是结构化属性，往返后原样保留。
// 少了它们，sin/cos 广播到 32 头那一路的独立量化参数在 IR 里无处可放。

// CHECK-LABEL: @rope_with_subblocks
module {
  tt.func @rope_with_subblocks(%x: tensor<1x32x1x128xf16>,
                               %c: tensor<1x1x1x128xf16>,
                               %s: tensor<1x1x1x128xf16>) {
    // CHECK: broadcastSpec = #pim.broadcast_spec<axes = [1], repeats = [32]>
    // CHECK: subBlocks = ["Llama2Activation_Add_Cos", "Llama2Activation_Add_Sin", "Llama2Activation_Sin", "Llama2Activation_Sin", "Llama2Activation_Cos", "Llama2Activation_Cos"]
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64,
        subBlocks = ["Llama2Activation_Add_Cos", "Llama2Activation_Add_Sin",
                     "Llama2Activation_Sin", "Llama2Activation_Sin",
                     "Llama2Activation_Cos", "Llama2Activation_Cos"],
        broadcastSpec = #pim.broadcast_spec<axes = [1], repeats = [32]>}
       : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
       -> tensor<1x32x1x128xf16>
    tt.return
  }
}
