// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-tasklets=1 wram-bytes=65536' -pim-explicit-dma -pim-lower-to-emitc -convert-func-to-emitc | FileCheck %s --check-prefix=SINGLE
// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-tasklets=2 wram-bytes=65536' -pim-explicit-dma -pim-lower-to-emitc -convert-func-to-emitc | FileCheck %s --check-prefix=MULTI
// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-tasklets=3 wram-bytes=65536' -pim-explicit-dma -pim-lower-to-emitc -convert-func-to-emitc | FileCheck %s --check-prefix=REMAINDER

// pim-lower-to-emitc (formerly pim-lower-single-tasklet) is generalized to
// accept any pim.num-tasklets >= 1, splitting a tt.dot's M dimension into
// that many statically-unrolled row-range blocks -- see the pass description
// in Passes.td and LowerPIMToEmitC.cpp's file header for the full rationale.
//
// A single tiny "linear" kernel (a: 4x16, w: 8x16 transposed to 16x8,
// out: 4x8 -- M=4, K=16, N=8) run through three different num-tasklets
// values checks:
//   SINGLE (num-tasklets=1): the degenerate case reproduces the old
//     single-block behavior exactly -- one malloc for `a`'s whole M range,
//     one shared malloc for `w`, a single (m, n, k) loop nest.
//   MULTI (num-tasklets=2): M=4 splits evenly into two 2-row blocks. `w` is
//     snapshotted once and shared; `a` gets one snapshot per block, freed
//     before the next block's snapshot.
//   REMAINDER (num-tasklets=3): M=4 does not split evenly
//     (ceil(4/3) = 2 rows/tasklet) -- rows [0,2) and [2,4) are covered by two
//     blocks, and the third tasklet's range would start at row 4 (== M), so
//     no third block is emitted at all. Exactly two `a` snapshots, not three.

module {
  tt.func public @tiny_linear(%a_ptr: !tt.ptr<f32>, %w_ptr: !tt.ptr<f32>, %o_ptr: !tt.ptr<f32>) {
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
    %a_ptrs0 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
    %a_ptrs = tt.addptr %a_ptrs0, %a_off : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
    %a = tt.load %a_ptrs : tensor<4x16x!tt.ptr<f32>>

    %n2 = tt.expand_dims %arange_n {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %k2_ = tt.expand_dims %arange_k {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2b = tt.broadcast %n2 : tensor<8x1xi32> -> tensor<8x16xi32>
    %k2b_ = tt.broadcast %k2_ : tensor<1x16xi32> -> tensor<8x16xi32>
    %ck2 = arith.constant dense<16> : tensor<8x16xi32>
    %n_scaled = arith.muli %n2b, %ck2 : tensor<8x16xi32>
    %w_off = arith.addi %n_scaled, %k2b_ : tensor<8x16xi32>
    %w_ptrs0 = tt.splat %w_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
    %w_ptrs = tt.addptr %w_ptrs0, %w_off : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
    %w = tt.load %w_ptrs : tensor<8x16x!tt.ptr<f32>>
    %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>

    %acc0 = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
    %d = tt.dot %a, %wt, %acc0 : tensor<4x16xf32> * tensor<16x8xf32> -> tensor<4x8xf32>

    %m2o = tt.expand_dims %arange_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %n2o = tt.expand_dims %arange_n {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %m2ob = tt.broadcast %m2o : tensor<4x1xi32> -> tensor<4x8xi32>
    %n2ob = tt.broadcast %n2o : tensor<1x8xi32> -> tensor<4x8xi32>
    %ck3 = arith.constant dense<8> : tensor<4x8xi32>
    %m_scaled_o = arith.muli %m2ob, %ck3 : tensor<4x8xi32>
    %o_off = arith.addi %m_scaled_o, %n2ob : tensor<4x8xi32>
    %o_ptrs0 = tt.splat %o_ptr : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %o_ptrs = tt.addptr %o_ptrs0, %o_off : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    tt.store %o_ptrs, %d : tensor<4x8x!tt.ptr<f32>>
    tt.return
  }
}

// malloc/free are interleaved per block (each block's `a` snapshot is malloc'd
// and freed before the next block's malloc; `w`'s single shared snapshot is
// freed last) -- e.g. the actual num-tasklets=2 sequence is
// malloc(w), malloc(a0), free(a0), malloc(a1), free(a1), free(w). Checking
// just the malloc count is sufficient signal for "how many WRAM snapshots
// this lowering created" (each malloc has exactly one matching free, by
// construction of emitDotLoops/snapshotToLocal) without depending on
// FileCheck's monotonic forward scan being able to interleave two separate
// COUNT directives against interleaved malloc/free call sites.

// SINGLE-LABEL: emitc.func @tiny_linear
// One shared `w` snapshot + one `a` snapshot for the single [0, 4) block.
// SINGLE-COUNT-2: call_opaque "malloc"
// SINGLE-NOT: call_opaque "malloc"

// MULTI-LABEL: emitc.func @tiny_linear
// One shared `w` snapshot + one `a` snapshot per of the two 2-row blocks.
// MULTI-COUNT-3: call_opaque "malloc"
// MULTI-NOT: call_opaque "malloc"

// REMAINDER-LABEL: emitc.func @tiny_linear
// ceil(4/3) = 2 rows/tasklet -> blocks [0,2) and [2,4) cover all of M; the
// third tasklet's range starts at row 4 (== M) so no third block is emitted.
// REMAINDER-COUNT-3: call_opaque "malloc"
// REMAINDER-NOT: call_opaque "malloc"
