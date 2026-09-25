// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// `#pim.phase_spec` is how an expanded operator records that one of its ops
// owns a single engine traversal, so this file is the attribute on its own --
// no pass runs over most of it.

// CHECK-LABEL: @phase_spec_round_trip
module {
  tt.func @phase_spec_round_trip(%x: tensor<1x32xf16>) {
    // A single phase with no fan-in: `reads` is absent, not empty.
    // CHECK: #pim.phase_spec<index = 0, bytes = 64, unit = vpu>
    %a = pim.lut %x {kind = #pim.activation<reciprocal>,
                     phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    // The dynamic-quantization fan-out: phases 1 and 2 both read phase 0.
    // CHECK: #pim.phase_spec<index = 1, bytes = 64, unit = cstl, reads = [0]>
    %b = pim.lut %a {kind = #pim.activation<relu>,
                     phases = [#pim.phase_spec<index = 1, bytes = 64, unit = cstl, reads = [0]>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    // Softmax's last phase reads two earlier ones.
    // CHECK: #pim.phase_spec<index = 2, bytes = 64, unit = cstl, reads = [0, 1]>
    %c = pim.lut %b {kind = #pim.activation<exp>,
                     phases = [#pim.phase_spec<index = 2, bytes = 64, unit = cstl, reads = [0, 1]>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    // A phase that shares its intermediate buffer with the neighbours.
    // CHECK: #pim.phase_spec<index = 3, bytes = 64, unit = cstl, forceConsecutive = true>
    %d = pim.lut %c {kind = #pim.activation<exp>,
                     phases = [#pim.phase_spec<index = 3, bytes = 64, unit = cstl, forceConsecutive = true>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}
