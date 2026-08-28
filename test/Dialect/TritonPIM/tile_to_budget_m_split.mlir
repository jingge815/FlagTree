// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=32 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget | FileCheck %s --check-prefix=OK
// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=32 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget -pim-explicit-dma -pim-lower-to-emitc -convert-func-to-emitc | FileCheck %s --check-prefix=EMITC

// Unlike the other tile_to_budget_*.mlir fixtures (M=1 or M left untouched),
// this one picks a WRAM/dma-align combination that forces the M dimension
// itself to shrink (M=8 -> tile_m=2), not just N/K -- covering the pass's
// general M/N/K-symmetric search and IR rewrite, not only the N/K path the
// real llama2-7b `linear` shapes currently exercise (decode's M is always 1
// there, so M-splitting has no other test coverage today).

module {
  tt.func public @linear_m_split(%a_ptr: !tt.ptr<f16>, %w_ptr: !tt.ptr<f16>, %o_ptr: !tt.ptr<f16>) {
    %arange_m = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %arange_k = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %arange_n = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

    %m2 = tt.expand_dims %arange_m {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %k2 = tt.expand_dims %arange_k {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %m2b = tt.broadcast %m2 : tensor<8x1xi32> -> tensor<8x16xi32>
    %k2b = tt.broadcast %k2 : tensor<1x16xi32> -> tensor<8x16xi32>
    %ck = arith.constant dense<16> : tensor<8x16xi32>
    %m_scaled = arith.muli %m2b, %ck : tensor<8x16xi32>
    %a_off = arith.addi %m_scaled, %k2b : tensor<8x16xi32>
    %a_ptrs0 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
    %a_ptrs = tt.addptr %a_ptrs0, %a_off : tensor<8x16x!tt.ptr<f16>>, tensor<8x16xi32>
    %a = tt.load %a_ptrs : tensor<8x16x!tt.ptr<f16>>

    %n2 = tt.expand_dims %arange_n {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %k2_ = tt.expand_dims %arange_k {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2b = tt.broadcast %n2 : tensor<4x1xi32> -> tensor<4x16xi32>
    %k2b_ = tt.broadcast %k2_ : tensor<1x16xi32> -> tensor<4x16xi32>
    %ck2 = arith.constant dense<16> : tensor<4x16xi32>
    %n_scaled = arith.muli %n2b, %ck2 : tensor<4x16xi32>
    %w_off = arith.addi %n_scaled, %k2b_ : tensor<4x16xi32>
    %w_ptrs0 = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<4x16x!tt.ptr<f16>>
    %w_ptrs = tt.addptr %w_ptrs0, %w_off : tensor<4x16x!tt.ptr<f16>>, tensor<4x16xi32>
    %w = tt.load %w_ptrs : tensor<4x16x!tt.ptr<f16>>
    %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<4x16xf16> -> tensor<16x4xf16>

    %acc0 = arith.constant dense<0.000000e+00> : tensor<8x4xf32>
    %d0 = tt.dot %a, %wt, %acc0 : tensor<8x16xf16> * tensor<16x4xf16> -> tensor<8x4xf32>
    %d = arith.truncf %d0 : tensor<8x4xf32> to tensor<8x4xf16>

    %m2o = tt.expand_dims %arange_m {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %n2o = tt.expand_dims %arange_n {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %n2ob = tt.broadcast %n2o : tensor<1x4xi32> -> tensor<8x4xi32>
    %ck3 = arith.constant dense<4> : tensor<8x1xi32>
    %m_scaled_o = arith.muli %m2o, %ck3 : tensor<8x1xi32>
    %m_scaled_ob = tt.broadcast %m_scaled_o : tensor<8x1xi32> -> tensor<8x4xi32>
    %o_off = arith.addi %m_scaled_ob, %n2ob : tensor<8x4xi32>
    %o_ptrs0 = tt.splat %o_ptr : !tt.ptr<f16> -> tensor<8x4x!tt.ptr<f16>>
    %o_ptrs = tt.addptr %o_ptrs0, %o_off : tensor<8x4x!tt.ptr<f16>>, tensor<8x4xi32>
    tt.store %o_ptrs, %d : tensor<8x4x!tt.ptr<f16>>
    tt.return
  }
}

// The front-end tile here is the whole (untiled) M=8/N=4/K=16 -- no `scf.for`
// at all, unlike the other fixtures. `wram-bytes=32`/`dma-align=8` forces a
// tile that fits in only 24 bytes (2*2*2 + 2*2*2 + 2*2*2), which needs all
// three dimensions shrunk to 2.
// OK-DAG: "pim.tile-m" = 2 : i64
// OK-DAG: "pim.tile-n" = 2 : i64
// OK-DAG: "pim.tile-k" = 2 : i64
// OK-DAG: "pim.tile-wram-bytes" = 24 : i64
// The rewrite must introduce a real M-tile loop (there was none before) --
// this is the one thing that distinguishes M-splitting from the N/K-only
// path the other fixtures cover.
// OK: scf.for
// OK: scf.for
// OK: scf.for

// The full pass chain must still produce valid EmitC: pim-lower-to-emitc's
// own M-splitting-across-tasklets logic operates on top of whatever M-tile
// loop this pass built, so this also exercises "two independent layers of
// M splitting stack correctly" (this pass's WRAM-budget tile vs.
// pim-lower-to-emitc's per-tasklet split).
// EMITC: emitc.func
// EMITC: for
// EMITC: return
