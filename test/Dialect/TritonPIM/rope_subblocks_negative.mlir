// RUN: not triton-opt %s -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// 子块必须正好 6 个、且是已知名字。

// CHECK: subBlocks 必须正好 6 个
module {
  tt.func @too_few(%x: tensor<1x32x1x128xf16>, %c: tensor<1x1x1x128xf16>,
                   %s: tensor<1x1x1x128xf16>) {
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64,
        subBlocks = ["Llama2Activation_Add_Cos"]}
       : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
       -> tensor<1x32x1x128xf16>
    tt.return
  }
}

// -----

// CHECK: subBlocks 顺序不对
module {
  tt.func @bad_name(%x: tensor<1x32x1x128xf16>, %c: tensor<1x1x1x128xf16>,
                    %s: tensor<1x1x1x128xf16>) {
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64,
        subBlocks = ["nope", "Llama2Activation_Add_Sin", "Llama2Activation_Sin",
                     "Llama2Activation_Sin", "Llama2Activation_Cos",
                     "Llama2Activation_Cos"]}
       : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
       -> tensor<1x32x1x128xf16>
    tt.return
  }
}

// -----

// CHECK: 广播轴超出秩
module {
  tt.func @axis_oob(%x: tensor<1x32x1x128xf16>, %c: tensor<1x1x1x128xf16>,
                    %s: tensor<1x1x1x128xf16>) {
    %y = pim.rope %x, %c, %s {numHeads = 32 : i64,
        broadcastSpec = #pim.broadcast_spec<axes = [7], repeats = [32]>}
       : tensor<1x32x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
       -> tensor<1x32x1x128xf16>
    tt.return
  }
}
