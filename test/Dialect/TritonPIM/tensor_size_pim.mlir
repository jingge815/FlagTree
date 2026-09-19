// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The power-of-two tensor-size rule is relaxed for modules targeting PIM, whose
// extents come from the model (llama2's MLP width is 11008) rather than from a
// vector width. Keyed on the module's `pim.target` attribute so every other
// backend keeps the rule.

// A PIM module accepts a non-power-of-two `tt.store`.
// CHECK-LABEL: @pim_non_pow2_store
module attributes {pim.target = "pim:v1"} {
  tt.func @pim_non_pow2_store(%x: tensor<1x11008xf16>,
                              %o: tensor<1x11008x!tt.ptr<f16>>) {
    // CHECK: tt.store
    tt.store %o, %x : tensor<1x11008x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Powers of two are of course still fine there.
// CHECK-LABEL: @pim_pow2_store
module attributes {pim.target = "pim:v1"} {
  tt.func @pim_pow2_store(%x: tensor<1x4096xf16>,
                          %o: tensor<1x4096x!tt.ptr<f16>>) {
    // CHECK: tt.store
    tt.store %o, %x : tensor<1x4096x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Without `pim.target` the rule still applies -- this is what keeps the
// relaxation from leaking into the GPU backends.
module {
  tt.func @plain_non_pow2_store(%x: tensor<1x11008xf16>,
                                %o: tensor<1x11008x!tt.ptr<f16>>) {
    // expected-error @below {{Number of elements must be power-of-two}}
    tt.store %o, %x : tensor<1x11008x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// The size ceiling is independent of the power-of-two rule, so it must still
// fire inside a PIM module.
module attributes {pim.target = "pim:v1"} {
  tt.func @pim_too_many_elements(%x: tensor<16777216x8xf16>,
                                 %o: tensor<16777216x8x!tt.ptr<f16>>) {
    // expected-error @below {{Maximum allowed number of elements is}}
    tt.store %o, %x : tensor<16777216x8x!tt.ptr<f16>>
    tt.return
  }
}
