// RUN: triton-opt %s -split-input-file -pim-expand-phases -verify-diagnostics

// Dynamic quantize without a per-group spec has no grouping to reduce over.
// (A group size that does not divide the axis is already rejected by
// QuantizeOp's own verifier, see operator_ops_negative.mlir.)

module {
  tt.func @dq_not_per_group(%x: tensor<1x16xf16>, %s: tensor<1xf16>) {
    // expected-error @+1 {{dynamic quantize expansion requires a per_group spec}}
    %q = pim.quantize %x, %s
       {dynamic, spec = #pim.quant_spec<granularity = per_tensor>}
       : tensor<1x16xf16>, tensor<1xf16> -> tensor<1x16xi8>
    tt.return
  }
}
