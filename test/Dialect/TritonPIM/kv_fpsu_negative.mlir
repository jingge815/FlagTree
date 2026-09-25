// RUN: not triton-opt %s -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// 带定标规格却不是定点模式：定标会按一个不适用的模式解释。

// CHECK: fpsuSpec 要求 mode 为 fixed_point
module {
  tt.func @fpsu_on_float(%k: tensor<1x8x128xi8>,
                         %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>,
                         %pos: i32, %slot: tensor<1xi16>) {
    pim.kv_cache %k, %cache, %pos[%slot]
        {layer = 0 : i64, isKey,
         fpsuSpec = #pim.fpsu_spec<mode = floating_point>}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32 [tensor<1xi16>]
    tt.return
  }
}
