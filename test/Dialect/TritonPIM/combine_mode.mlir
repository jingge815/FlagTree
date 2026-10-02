// RUN: triton-opt %s -split-input-file -verify-diagnostics

// `combineMode` says how an eltwise result rejoins the chain that consumes it.
//
// Before this, the attribute had five producers (all inside `-pim-expand-phases`,
// all passing a null placeholder) and no reader anywhere in the dialect:
// `getCombineMode()` was never called. Any value written to it -- right or wrong
// -- had exactly the same effect, which is none. These two cases are what make
// the field readable.

// A skip connection adds the skipped value back. Multiplying by it would scale
// the residual instead of restoring it, so the mode and the kind have to agree.
module {
  tt.func @skip_connection_must_be_an_add(
      %a: tensor<4x8xf16>, %b: tensor<4x8xf16>) {
    // expected-error @below {{skip_connection adds the skipped value back, but this op's kind is mul}}
    %s = pim.eltwise %a, %b
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        combineMode = #pim.combine_mode<skip_connection>}
       : tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// The honest values the expansion writes. `straightforward` is an intra-chain
// intermediate and an add is what a skip connection looks like when it is one.
module {
  tt.func @straightforward_and_a_real_skip_connection(
      %a: tensor<4x8xf16>, %b: tensor<4x8xf16>) {
    %s = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        combineMode = #pim.combine_mode<skip_connection>}
       : tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    %t = pim.eltwise %s, %b
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        combineMode = #pim.combine_mode<straightforward>}
       : tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}
