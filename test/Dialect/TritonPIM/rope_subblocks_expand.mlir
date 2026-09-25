// RUN: triton-opt %s -pim-expand-phases | FileCheck %s

// `pim.rope` 展开成 3 个 eltwise 相位后，6 个子块名和广播规格必须跟着
// 最后一相（add，原 op 的直接替代者）走，不能随原 op 一起被擦掉。
// 少了它们，sin/cos 广播到 32 头那一路的独立量化参数在展开后的 IR 里
// 无处可放，下游读不到。

// CHECK-LABEL: @rope_subblocks_survive_expansion
module {
  tt.func @rope_subblocks_survive_expansion(%x: tensor<1x32x1x128xf16>,
                                             %c: tensor<1x1x1x128xf16>,
                                             %s: tensor<1x1x1x128xf16>) {
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64, tailCardValue = 0 : i64,
        subBlocks = ["Llama2Activation_Add_Cos", "Llama2Activation_Add_Sin",
                     "Llama2Activation_Sin", "Llama2Activation_Sin",
                     "Llama2Activation_Cos", "Llama2Activation_Cos"],
        broadcastSpec = #pim.broadcast_spec<axes = [1], repeats = [32]>}
       : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
       -> tensor<1x32x1x128xf16>
    tt.return
  }
}

// CHECK: pim.eltwise
// CHECK: pim.eltwise
// CHECK: pim.eltwise {{.*}}sourceBroadcastSpec = #pim.broadcast_spec<axes = [1], repeats = [32]>
// CHECK-SAME: sourceSubBlocks = ["Llama2Activation_Add_Cos"
// CHECK-NOT: pim.rope
