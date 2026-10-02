// RUN: not triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=2 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' -split-input-file 2>&1 | FileCheck %s --check-prefix=TOOMANY
// RUN: not triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=2 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' -split-input-file 2>&1 | FileCheck %s --check-prefix=BADID

// A placement can contradict the hardware it is compiled for. That cannot be
// caught by the attribute's own verifier -- the DPU count it contradicts is a
// pass option, which an attribute cannot see -- so the conversion pass checks it
// against `num-dpus` and fails there.

// TOOMANY: spreads a tensor over 4 DPUs but the device has only 2
module attributes {pim.placement = #pim.placement<kind = shard, dim = 0, numDpus = 4>} {
  tt.func public @more_dpus_than_the_device_has() {
    %c = arith.constant dense<0.0> : tensor<8x32xf16>
    tt.return
  }
}

// -----

// The ids can be in range as a count yet name a DPU the device does not have:
// a pipeline stage that was placed on DPUs 4..7 compiled for a 2-DPU device.
// BADID: names DPU 7 but the device has only 2
module attributes {pim.placement = #pim.placement<kind = shard, dim = 0, numDpus = 2, dpuIds = [0, 7]>} {
  tt.func public @names_a_dpu_the_device_lacks() {
    %c = arith.constant dense<0.0> : tensor<8x32xf16>
    tt.return
  }
}
