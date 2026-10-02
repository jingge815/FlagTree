// RUN: triton-opt %s -pim-tile-to-budget 2>&1 | FileCheck %s

// `pim.placed-shards` reports what the LAYOUT ENCODINGS say, not what the
// placement asked for.
//
// Reading the placement back would make this attribute a mirror of the intent:
// `placed-shards == numDpus` would hold by construction, and the consumer's
// "intent vs effect" comparison would be a tautology that can never fire --
// while the docs describe it as "what this pass actually saw". Counting the
// encodings makes the value honest and lets a dropped split show up as 1.
//
// This module is that case: the placement declares a split over 2 DPUs, and no
// tensor encoding records one. Before the fix this reported 2.

// The bytes are still this body's own per-DPU footprint: the shapes reach this
// pass already split (the graph compiler builds from the plan's local_shape).
// CHECK-DAG: "pim.placed-shards" = 1 : i64
// CHECK-DAG: "pim.placed-mram-bytes" = 272384 : i64
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>, "pim.dma-align" = 8 : i32, "pim.mram-bytes" = 4294967296 : i64, "pim.wram-bytes" = 65536 : i32} {
  tt.func public @sharded_over_two(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
  }}
