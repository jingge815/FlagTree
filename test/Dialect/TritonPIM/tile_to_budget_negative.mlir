// RUN: not triton-opt %s -pim-tile-to-budget 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=128 mram-bytes=4294967296 dma-align=64' -pim-tile-to-budget 2>&1 | FileCheck %s --check-prefix=OVER
// RUN: not triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=65536 mram-bytes=4294967296 dma-align=256' -pim-tile-to-budget 2>&1 | FileCheck %s --check-prefix=BADALIGN

module {
  tt.func public @linear_negative(%a_ptr: !tt.ptr<f16>, %w_ptr: !tt.ptr<f16>) {
    %ao = arith.constant dense<0> : tensor<4x16xi32>
    %bo = arith.constant dense<0> : tensor<8x16xi32>
    %ap = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<4x16x!tt.ptr<f16>>
    %aa = tt.addptr %ap, %ao : tensor<4x16x!tt.ptr<f16>>, tensor<4x16xi32>
    %bp = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<8x16x!tt.ptr<f16>>
    %ba = tt.addptr %bp, %bo : tensor<8x16x!tt.ptr<f16>>, tensor<8x16xi32>
    %a = tt.load %aa : tensor<4x16x!tt.ptr<f16>>
    %w = tt.load %ba : tensor<8x16x!tt.ptr<f16>>
    %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>
    %acc = arith.constant dense<0.000000e+00> : tensor<4x8xf32>
    %d = tt.dot %a, %wt, %acc : tensor<4x16xf16> * tensor<16x8xf16> -> tensor<4x8xf32>
    tt.return
  }
}

// MISSING: error: pim-tile-to-budget requires pim.wram-bytes
// OVER: error: no legal power-of-two tile fits: M=4 N=8 K=16 dtype_bytes=2 wram_bytes=128 mram_bytes=4294967296 dma_align=64; smallest tried was M=1 N=1 K=1
// BADALIGN: error: no legal power-of-two tile fits: M=4 N=8 K=16 dtype_bytes=2 wram_bytes=65536 mram_bytes=4294967296 dma_align=256; smallest tried was M=1 N=1 K=1
