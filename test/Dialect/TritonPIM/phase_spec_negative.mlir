// RUN: triton-opt %s -split-input-file -verify-diagnostics

// A phase that reads itself, or a later phase, describes a cycle the hardware
// cannot schedule. Getting this wrong is silent -- serializing dynamic
// quantization's fan-out puts the reciprocal off by 256x -- so the attribute
// rejects it at parse time rather than leaving it to a consumer.

module {
  tt.func @reads_itself(%x: tensor<1x32xf16>) {
    // expected-error @+2 {{cannot read phase 1}}
    %a = pim.lut %x {kind = #pim.activation<exp>,
                     phases = [#pim.phase_spec<index = 1, bytes = 64, unit = cstl, reads = [1]>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

module {
  tt.func @reads_a_later_phase(%x: tensor<1x32xf16>) {
    // expected-error @+2 {{cannot read phase 3}}
    %a = pim.lut %x {kind = #pim.activation<exp>,
                     phases = [#pim.phase_spec<index = 1, bytes = 64, unit = cstl, reads = [3]>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// A phase that streams nothing occupies no traversal, so admitting one would
// inflate every consumer's phase count.

module {
  tt.func @zero_bytes(%x: tensor<1x32xf16>) {
    // expected-error @+2 {{phase bytes must be positive}}
    %a = pim.lut %x {kind = #pim.activation<exp>,
                     phases = [#pim.phase_spec<index = 0, bytes = 0, unit = cstl>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

module {
  tt.func @negative_index(%x: tensor<1x32xf16>) {
    // expected-error @+2 {{phase index must be non-negative}}
    %a = pim.lut %x {kind = #pim.activation<exp>,
                     phases = [#pim.phase_spec<index = -1, bytes = 64, unit = cstl>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

module {
  tt.func @reads_must_be_indices(%x: tensor<1x32xf16>) {
    // expected-error @+2 {{reads must contain only integer phase indices}}
    %a = pim.lut %x {kind = #pim.activation<exp>,
                     phases = [#pim.phase_spec<index = 2, bytes = 64, unit = cstl, reads = ["0"]>]}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// `phases` is declared on the shared operator base class, so the dialect
// rejects a value of the wrong shape instead of accepting it and leaving the
// consumer to read nothing back. Before the declaration this compiled: the
// attribute was written with a raw `setAttr` and no type was ever checked, so
// the failure only showed up later as a phase family silently missing from the
// graph.
module {
  tt.func @phases_must_be_an_array(%x: tensor<1x32xf16>) {
    // expected-error @+1 {{attribute 'phases' failed to satisfy constraint}}
    %a = pim.lut %x {kind = #pim.activation<exp>, phases = 3 : i64}
        : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

