// RUN: triton-opt %s -split-input-file -convert-triton-to-pim='target=pim:v1 num-dpus=4 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget | FileCheck %s

// The Placement dimension's RETURN path, and the decision it changes.
//
// The graph compiler decides the split; it cannot know what the split cost,
// because the per-DPU footprint depends on the tile and the tile is resolved
// here. So `pim-tile-to-budget` reports two things back on the module: the bytes
// it charged one DPU, and the split width it actually saw. The second is not
// redundant with the placement's `numDpus` -- it is what this pass *did*, so a
// consumer can compare intent against effect instead of assuming they agree
// (`replicated` below is exactly the case where the two differ legitimately).
//
// The per-DPU bytes count the accumulator at its own width: x and w hold f16
// operands but `out` holds the f32 accumulator, so the footprint is
// (m*k + n*k)*2 + m*n*4 = 272384 for this 4x512x32 tile, not 268288. Charging
// the operand width for all three undercounts `out` by half.
//
// The reported bytes are NOT the global footprint divided by the split. The
// shapes that reach this pass are already one DPU's share: the graph compiler
// builds the kernel from the execution plan's `local_shape`, so a tp2 `linear`
// arrives with its weight already halved. Dividing again loosened the budget
// check by exactly the split factor -- measured, a kernel needing 7168 B per DPU
// passed a 5000 B budget. So all three modules below, having identical bodies,
// report the SAME bytes; only the width differs.
//
// The function bodies are real TTIR from the graph compiler's `linear` kernel:
// `-pim-tile-to-budget` requires a `tt.load` feeding `tt.dot` and rejects
// hand-written stand-ins, so a synthetic body cannot exercise this path.

// tp2: the width this pass saw is 2. The bytes are this body's own footprint,
// undivided -- the body already describes one DPU's share.
// CHECK-DAG: "pim.placed-shards" = 2
// CHECK-DAG: "pim.placed-mram-bytes" = 272384
// CHECK-LABEL: @sharded_over_two
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
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

// -----

// No placement: width 1, same footprint -- byte for byte what this pass
// reported before the return path existed. Identical to the tp2 case above,
// which is the point: the bytes come from the body, not from the split.
// CHECK-DAG: "pim.placed-shards" = 1
// CHECK-DAG: "pim.placed-mram-bytes" = 272384
// CHECK-LABEL: @not_sharded
module {
  tt.func public @not_sharded(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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

// -----

// Replicate keeps the whole shape on every DPU, so the width stays 1 even
// though `numDpus` is 4. This is the case that shows the returned width cannot
// be inferred from `numDpus`.
// CHECK-DAG: "pim.placed-shards" = 1
// CHECK-DAG: "pim.placed-mram-bytes" = 272384
// CHECK-LABEL: @replicated
module attributes {pim.placement = #pim.placement<kind = replicate, numDpus = 4>} {
  tt.func public @replicated(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
