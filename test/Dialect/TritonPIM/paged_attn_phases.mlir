// RUN: triton-opt %s -split-input-file -pim-expand-phases -pim-verify-gml-contract | FileCheck %s

// `pim.paged_attn` is not expanded -- it is lowered straight to C -- so its
// phases are stamped on the op. What the stamp has to say beyond the buffer
// sizes is *which* phases repeat: 0-4 run once per history block, 5 runs once
// per operator. Without that a cost model bills one block for the whole
// history and the KV traffic never appears at all.

// CHECK-LABEL: @paged_attn_block_loop
module {
  // q is one token of 128 halves, the slot 16 tokens, the table 4 blocks.
  tt.func @paged_attn_block_loop(%q: tensor<1x128xf16>,
                                 %cache: !pim.memdesc<8192xf16, #pim.mram>,
                                 %bt: tensor<4xi32>, %ko: i32, %vo: i32,
                                 %pos: i32, %nb: i32, %nt: i32) {
    // CHECK: pim.paged_attn
    // CHECK-SAME: #pim.phase_spec<index = 0, bytes = 32, unit = nmu, perBlock = true, poolBytes = 4096>
    // CHECK-SAME: #pim.phase_spec<index = 1, bytes = 4, unit = vpu, reads = [0], perBlock = true>
    // CHECK-SAME: #pim.phase_spec<index = 2, bytes = 32, unit = activation, reads = [0, 1], perBlock = true>
    // CHECK-SAME: #pim.phase_spec<index = 3, bytes = 4, unit = vpu, reads = [2], perBlock = true>
    // CHECK-SAME: #pim.phase_spec<index = 4, bytes = 512, unit = nmu, reads = [2], perBlock = true, poolBytes = 4096>
    // CHECK-SAME: #pim.phase_spec<index = 5, bytes = 256, unit = combiner, reads = [3, 4]>
    %y = pim.paged_attn %q, %cache, %bt, %ko, %vo, %pos, %nb, %nt
        {slotTokens = 16 : i64, slotElems = 2048 : i64,
         scale = 0.08838834764831845 : f64}
        : tensor<1x128xf16>, !pim.memdesc<8192xf16, #pim.mram>,
          tensor<4xi32>, i32, i32, i32, i32, i32 -> tensor<1x128xf16>
    tt.return
  }
}
