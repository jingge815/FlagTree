// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=65536 mram-bytes=4294967296 dma-align=64' -pim-tile-to-budget | FileCheck %s --check-prefix=OK

module {
  tt.func public @tiny_linear(%a_ptr: !tt.ptr<f16>, %w_ptr: !tt.ptr<f16>,
                              %o_ptr: !tt.ptr<f16>) {
    %arange_m = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %arange_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %arange_n = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>

    %m2 = tt.expand_dims %arange_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %k2 = tt.expand_dims %arange_k {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %m2b = tt.broadcast %m2 : tensor<4x1xi32> -> tensor<4x16xi32>
    %k2b = tt.broadcast %k2 : tensor<1x16xi32> -> tensor<4x16xi32>
    %ck = arith.constant dense<16> : tensor<4x16xi32>
    %m_scaled = arith.muli %m2b, %ck : tensor<4x16xi32>
    %a_off = arith.addi %m_scaled, %k2b : tensor<4x16xi32>
    %a_ptrs0 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<4x16x!tt.ptr<f16>>
    %a_ptrs = tt.addptr %a_ptrs0, %a_off : tensor<4x16x!tt.ptr<f16>>, tensor<4x16xi32>
    %a = tt.load %a_ptrs : tensor<4x16x!tt.ptr<f16>>

    %n2 = tt.expand_dims %arange_n {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %k2_ = tt.expand_dims %arange_k {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2b = tt.broadcast %n2 : tensor<8x1xi32> -> tensor<8x16xi32>
    %k2b_ = tt.broadcast %k2_ : tensor<1x16xi32> -> tensor<8x16xi32>
    %ck2 = arith.constant dense<16> : tensor<8x16xi32>
    %n_scaled = arith.muli %n2b, %ck2 : tensor<8x16xi32>
    %w_off = arith.addi %n_scaled, %k2b_ : tensor<8x16xi32>
    %w_ptrs0 = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
    %w_ptrs = tt.addptr %w_ptrs0, %w_off : tensor<8x16x!tt.ptr<f16>>, tensor<8x16xi32>
    %w = tt.load %w_ptrs : tensor<8x16x!tt.ptr<f16>>
    %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>

    %acc0 = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
    %d0 = tt.dot %a, %wt, %acc0 : tensor<4x16xf16> * tensor<16x8xf16> -> tensor<4x8xf32>
    %d = arith.truncf %d0 : tensor<4x8xf32> to tensor<4x8xf16>

    %m2o = tt.expand_dims %arange_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %n2o = tt.expand_dims %arange_n {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %m2ob = tt.broadcast %m2o : tensor<4x1xi32> -> tensor<4x8xi32>
    %n2ob = tt.broadcast %n2o : tensor<1x8xi32> -> tensor<4x8xi32>
    %ck3 = arith.constant dense<8> : tensor<4x8xi32>
    %m_scaled_o = arith.muli %m2ob, %ck3 : tensor<4x8xi32>
    %o_off = arith.addi %m_scaled_o, %n2ob : tensor<4x8xi32>
    %o_ptrs0 = tt.splat %o_ptr : !tt.ptr<f16> -> tensor<4x8x!tt.ptr<f16>>
    %o_ptrs = tt.addptr %o_ptrs0, %o_off : tensor<4x8x!tt.ptr<f16>>, tensor<4x8xi32>
    tt.store %o_ptrs, %d : tensor<4x8x!tt.ptr<f16>>
    tt.return
  }
}

// OK-DAG: "pim.tile-m" =
// OK-DAG: "pim.tile-n" =
// OK-DAG: "pim.tile-k" =
// OK-DAG: "pim.tile-wram-bytes" =
