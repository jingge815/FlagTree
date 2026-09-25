// RUN: triton-opt %s -split-input-file -verify-diagnostics -pim-fuse-activation

// A table folds into a matmul — that is where the target format keeps the gate
// projection's table. The other producers have nowhere to put it, and letting
// the lut stay separate would drop a whole activation/contraction field family
// from the graph with no error. Refuse loudly instead.

module {
  tt.func @table_on_a_non_matmul_producer_is_refused(
      %a: tensor<4x8xi8>, %b: tensor<4x8xi8>, %t: tensor<144xf16>) {
    %s = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}
       : tensor<4x8xi8>, tensor<4x8xi8> -> tensor<4x8xi8>
    %y = pim.lut %s, %t {kind = #pim.activation<silu>} : tensor<4x8xi8>, tensor<144xf16> -> tensor<4x8xi8>
    tt.return
  }
}
