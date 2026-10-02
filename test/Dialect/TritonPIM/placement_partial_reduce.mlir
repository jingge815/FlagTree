// RUN: triton-opt %s -split-input-file -convert-triton-to-pim='target=pim:v1 num-dpus=4 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget | FileCheck %s

// The `partial` placement kind, and the decision it changes.
//
// `partial` says every DPU holds a whole-shaped *piece* of the sum: the value
// exists only once those pieces are reduced across DPUs. That reduction has to
// land somewhere -- a DPU receiving a peer's piece needs room for it beside its
// own -- so a partial operator occupies one output tile MORE per DPU than the
// same operator under `shard` or `replicate`.
//
// That is why the kind has to reach this pass instead of stopping at the graph
// compiler. `replicate` and `partial` are identical in shape (both keep the
// whole shape on every DPU) and were therefore indistinguishable here; they are
// not identical in cost, because only one owes a reduction. Before this change
// the graph compiler sent neither kind at all.
//
// The function body is real TTIR from the graph compiler's `linear` kernel:
// `-pim-tile-to-budget` requires a `tt.load` feeding `tt.dot` and rejects
// hand-written stand-ins.

// Partial over 2 DPUs: the staging buffer is one output tile, charged at the
// ACCUMULATOR's width. The piece arriving from a peer is itself a partial sum,
// so it has the accumulator's format -- f32 here, where the operands are f16:
// 4 x 512 x 4 = 8192.
// CHECK-DAG: "pim.placed-reduce-bytes" = 8192
// CHECK-LABEL: @partial_stages_the_reduction
module attributes {pim.placement = #pim.placement<kind = partial, numDpus = 2, reduce = sum>} {
  tt.func public @partial_stages_the_reduction(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<4x512xf32>
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<512> : tensor<4x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<512x1xi32>
    %cst_1 = arith.constant dense<256> : tensor<4x1xi32>
    %offs_m = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offs_n = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %acc_2 = scf.for %k0 = %c0_i32 to %c256_i32 step %c32_i32 iter_args(%acc_8 = %acc) -> (tensor<4x512xf32>)  : i32 {
      %offs_k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %offs_k_9 = tt.splat %k0 : i32 -> tensor<32xi32>
      %offs_k_10 = arith.addi %offs_k_9, %offs_k : tensor<32xi32>
      %x_off = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
      %x_off_11 = arith.muli %x_off, %cst_1 : tensor<4x1xi32>
      %x_off_12 = tt.expand_dims %offs_k_10 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %x_off_13 = tt.broadcast %x_off_11 : tensor<4x1xi32> -> tensor<4x32xi32>
      %x_off_14 = tt.broadcast %x_off_12 : tensor<1x32xi32> -> tensor<4x32xi32>
      %x_off_15 = arith.addi %x_off_13, %x_off_14 : tensor<4x32xi32>
      %w_off = tt.expand_dims %offs_n {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
      %w_off_16 = arith.muli %w_off, %cst_0 : tensor<512x1xi32>
      %w_off_17 = tt.broadcast %w_off_16 : tensor<512x1xi32> -> tensor<512x32xi32>
      %w_off_18 = tt.broadcast %x_off_12 : tensor<1x32xi32> -> tensor<512x32xi32>
      %w_off_19 = arith.addi %w_off_17, %w_off_18 : tensor<512x32xi32>
      %x_blk = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<4x32x!tt.ptr<f16>>
      %x_blk_20 = tt.addptr %x_blk, %x_off_15 : tensor<4x32x!tt.ptr<f16>>, tensor<4x32xi32>
      %x_blk_21 = tt.load %x_blk_20 : tensor<4x32x!tt.ptr<f16>>
      %w_blk = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<512x32x!tt.ptr<f16>>
      %w_blk_22 = tt.addptr %w_blk, %w_off_19 : tensor<512x32x!tt.ptr<f16>>, tensor<512x32xi32>
      %w_blk_23 = tt.load %w_blk_22 : tensor<512x32x!tt.ptr<f16>>
      %acc_24 = tt.trans %w_blk_23 {order = array<i32: 1, 0>} : tensor<512x32xf16> -> tensor<32x512xf16>
      %acc_25 = tt.dot %x_blk_21, %acc_24, %acc_8 : tensor<4x32xf16> * tensor<32x512xf16> -> tensor<4x512xf32>
      scf.yield %acc_25 : tensor<4x512xf32>
    }
    %o_off = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %o_off_3 = arith.muli %o_off, %cst : tensor<4x1xi32>
    %o_off_4 = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %o_off_5 = tt.broadcast %o_off_3 : tensor<4x1xi32> -> tensor<4x512xi32>
    %o_off_6 = tt.broadcast %o_off_4 : tensor<1x512xi32> -> tensor<4x512xi32>
    %o_off_7 = arith.addi %o_off_5, %o_off_6 : tensor<4x512xi32>
    %0 = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<4x512x!tt.ptr<f16>>
    %1 = tt.addptr %0, %o_off_7 : tensor<4x512x!tt.ptr<f16>>, tensor<4x512xi32>
    %2 = arith.truncf %acc_2 : tensor<4x512xf32> to tensor<4x512xf16>
    tt.store %1, %2 : tensor<4x512x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Replicate keeps the whole shape too, but owes no reduction -- so no
// landing buffer. Zero, not absent: the pass did answer.
// CHECK-DAG: "pim.placed-reduce-bytes" = 0
// CHECK-LABEL: @replicate_stages_nothing
module attributes {pim.placement = #pim.placement<kind = replicate, numDpus = 2>} {
  tt.func public @replicate_stages_nothing(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<4x512xf32>
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<512> : tensor<4x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<512x1xi32>
    %cst_1 = arith.constant dense<256> : tensor<4x1xi32>
    %offs_m = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offs_n = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %acc_2 = scf.for %k0 = %c0_i32 to %c256_i32 step %c32_i32 iter_args(%acc_8 = %acc) -> (tensor<4x512xf32>)  : i32 {
      %offs_k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %offs_k_9 = tt.splat %k0 : i32 -> tensor<32xi32>
      %offs_k_10 = arith.addi %offs_k_9, %offs_k : tensor<32xi32>
      %x_off = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
      %x_off_11 = arith.muli %x_off, %cst_1 : tensor<4x1xi32>
      %x_off_12 = tt.expand_dims %offs_k_10 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %x_off_13 = tt.broadcast %x_off_11 : tensor<4x1xi32> -> tensor<4x32xi32>
      %x_off_14 = tt.broadcast %x_off_12 : tensor<1x32xi32> -> tensor<4x32xi32>
      %x_off_15 = arith.addi %x_off_13, %x_off_14 : tensor<4x32xi32>
      %w_off = tt.expand_dims %offs_n {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
      %w_off_16 = arith.muli %w_off, %cst_0 : tensor<512x1xi32>
      %w_off_17 = tt.broadcast %w_off_16 : tensor<512x1xi32> -> tensor<512x32xi32>
      %w_off_18 = tt.broadcast %x_off_12 : tensor<1x32xi32> -> tensor<512x32xi32>
      %w_off_19 = arith.addi %w_off_17, %w_off_18 : tensor<512x32xi32>
      %x_blk = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<4x32x!tt.ptr<f16>>
      %x_blk_20 = tt.addptr %x_blk, %x_off_15 : tensor<4x32x!tt.ptr<f16>>, tensor<4x32xi32>
      %x_blk_21 = tt.load %x_blk_20 : tensor<4x32x!tt.ptr<f16>>
      %w_blk = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<512x32x!tt.ptr<f16>>
      %w_blk_22 = tt.addptr %w_blk, %w_off_19 : tensor<512x32x!tt.ptr<f16>>, tensor<512x32xi32>
      %w_blk_23 = tt.load %w_blk_22 : tensor<512x32x!tt.ptr<f16>>
      %acc_24 = tt.trans %w_blk_23 {order = array<i32: 1, 0>} : tensor<512x32xf16> -> tensor<32x512xf16>
      %acc_25 = tt.dot %x_blk_21, %acc_24, %acc_8 : tensor<4x32xf16> * tensor<32x512xf16> -> tensor<4x512xf32>
      scf.yield %acc_25 : tensor<4x512xf32>
    }
    %o_off = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %o_off_3 = arith.muli %o_off, %cst : tensor<4x1xi32>
    %o_off_4 = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %o_off_5 = tt.broadcast %o_off_3 : tensor<4x1xi32> -> tensor<4x512xi32>
    %o_off_6 = tt.broadcast %o_off_4 : tensor<1x512xi32> -> tensor<4x512xi32>
    %o_off_7 = arith.addi %o_off_5, %o_off_6 : tensor<4x512xi32>
    %0 = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<4x512x!tt.ptr<f16>>
    %1 = tt.addptr %0, %o_off_7 : tensor<4x512x!tt.ptr<f16>>, tensor<4x512xi32>
    %2 = arith.truncf %acc_2 : tensor<4x512xf32> to tensor<4x512xf16>
    tt.store %1, %2 : tensor<4x512x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// A shard's pieces are disjoint, so there is nothing to reduce either.
// CHECK-DAG: "pim.placed-reduce-bytes" = 0
// CHECK-LABEL: @shard_stages_nothing
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
  tt.func public @shard_stages_nothing(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<4x512xf32>
    %c32_i32 = arith.constant 32 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<512> : tensor<4x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<512x1xi32>
    %cst_1 = arith.constant dense<256> : tensor<4x1xi32>
    %offs_m = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offs_n = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %acc_2 = scf.for %k0 = %c0_i32 to %c256_i32 step %c32_i32 iter_args(%acc_8 = %acc) -> (tensor<4x512xf32>)  : i32 {
      %offs_k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %offs_k_9 = tt.splat %k0 : i32 -> tensor<32xi32>
      %offs_k_10 = arith.addi %offs_k_9, %offs_k : tensor<32xi32>
      %x_off = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
      %x_off_11 = arith.muli %x_off, %cst_1 : tensor<4x1xi32>
      %x_off_12 = tt.expand_dims %offs_k_10 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %x_off_13 = tt.broadcast %x_off_11 : tensor<4x1xi32> -> tensor<4x32xi32>
      %x_off_14 = tt.broadcast %x_off_12 : tensor<1x32xi32> -> tensor<4x32xi32>
      %x_off_15 = arith.addi %x_off_13, %x_off_14 : tensor<4x32xi32>
      %w_off = tt.expand_dims %offs_n {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
      %w_off_16 = arith.muli %w_off, %cst_0 : tensor<512x1xi32>
      %w_off_17 = tt.broadcast %w_off_16 : tensor<512x1xi32> -> tensor<512x32xi32>
      %w_off_18 = tt.broadcast %x_off_12 : tensor<1x32xi32> -> tensor<512x32xi32>
      %w_off_19 = arith.addi %w_off_17, %w_off_18 : tensor<512x32xi32>
      %x_blk = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<4x32x!tt.ptr<f16>>
      %x_blk_20 = tt.addptr %x_blk, %x_off_15 : tensor<4x32x!tt.ptr<f16>>, tensor<4x32xi32>
      %x_blk_21 = tt.load %x_blk_20 : tensor<4x32x!tt.ptr<f16>>
      %w_blk = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<512x32x!tt.ptr<f16>>
      %w_blk_22 = tt.addptr %w_blk, %w_off_19 : tensor<512x32x!tt.ptr<f16>>, tensor<512x32xi32>
      %w_blk_23 = tt.load %w_blk_22 : tensor<512x32x!tt.ptr<f16>>
      %acc_24 = tt.trans %w_blk_23 {order = array<i32: 1, 0>} : tensor<512x32xf16> -> tensor<32x512xf16>
      %acc_25 = tt.dot %x_blk_21, %acc_24, %acc_8 : tensor<4x32xf16> * tensor<32x512xf16> -> tensor<4x512xf32>
      scf.yield %acc_25 : tensor<4x512xf32>
    }
    %o_off = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %o_off_3 = arith.muli %o_off, %cst : tensor<4x1xi32>
    %o_off_4 = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<512xi32> -> tensor<1x512xi32>
    %o_off_5 = tt.broadcast %o_off_3 : tensor<4x1xi32> -> tensor<4x512xi32>
    %o_off_6 = tt.broadcast %o_off_4 : tensor<1x512xi32> -> tensor<4x512xi32>
    %o_off_7 = arith.addi %o_off_5, %o_off_6 : tensor<4x512xi32>
    %0 = tt.splat %out_ptr : !tt.ptr<f16> -> tensor<4x512x!tt.ptr<f16>>
    %1 = tt.addptr %0, %o_off_7 : tensor<4x512x!tt.ptr<f16>>, tensor<4x512xi32>
    %2 = arith.truncf %acc_2 : tensor<4x512xf32> to tensor<4x512xf16>
    tt.store %1, %2 : tensor<4x512x!tt.ptr<f16>>
    tt.return
  }
}
