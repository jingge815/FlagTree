// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// KV 写入走定点通路时，定标规格必须留在 op 上并往返保留。

// CHECK-LABEL: @kv_with_fpsu
module {
  tt.func @kv_with_fpsu(%k: tensor<1x8x128xi8>,
                        %cache: !pim.memdesc<2048x8x128xi8, #pim.mram>,
                        %pos: i32, %slot: tensor<1xi16>) {
    // CHECK: fpsuSpec = #pim.fpsu_spec<mode = fixed_point>
    pim.kv_cache %k, %cache, %pos[%slot]
        {layer = 0 : i64, isKey,
         fpsuSpec = #pim.fpsu_spec<mode = fixed_point>}
        : tensor<1x8x128xi8>, !pim.memdesc<2048x8x128xi8, #pim.mram>, i32 [tensor<1xi16>]
    tt.return
  }
}
