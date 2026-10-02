// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=4 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' -pim-tile-to-budget | FileCheck %s

// The dtype dimension's return path, and the width the footprint is charged at.
//
// `tt.dot` on the graph compiler's `linear` takes f16 operands and produces an
// f32 accumulator (`tensor<4x32xf16> * tensor<32x512xf16> -> tensor<4x512xf32>`).
// The three staged buffers are therefore not all the same width: x and w hold
// operands, out holds the accumulator. Charging the operand width for all three
// undercounts the output tile by exactly the ratio, which loosens both the WRAM
// and the MRAM check on every mixed-precision kernel -- the common case.
//
// `pim.placed-elem-bytes` reports the operand width back, because a text-level
// consumer sees the element *type* but has to guess its width from a name table,
// and guessing wrong scales every byte count derived from it.
//
// Numbers for this kernel: tile 4x512x32, operands f16 (2 B), accumulator
// f32 (4 B).
//   x   = 4 x 32 x 2  =   256
//   w   = 512 x 32 x 2 = 32768
//   out = 4 x 512 x 4  =  8192   <- charged at the accumulator's width
//   tile-wram-bytes    = 41216   (was 37120 when `out` used 2 B)

// CHECK-DAG: "pim.placed-elem-bytes" = 2
// CHECK-DAG: "pim.tile-wram-bytes" = 41216
// CHECK-LABEL: @accumulator_is_charged_at_its_own_width
module {
  tt.func public @accumulator_is_charged_at_its_own_width(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %w_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
