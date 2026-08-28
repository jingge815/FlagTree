// RUN: not triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=128 mram-bytes=4294967296 dma-align=64' -pim-tile-to-budget 2>&1 | FileCheck %s --check-prefix=OVER
// RUN: triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=65536 mram-bytes=4294967296 dma-align=64' -pim-tile-to-budget -pim-explicit-dma | FileCheck %s --check-prefix=OK

module {
  tt.func public @large_visible_tile(%a_ptr: !tt.ptr<f16>, %w_ptr: !tt.ptr<f16>) {
    %ao = arith.constant dense<0> : tensor<64x32xi32>
    %bo = arith.constant dense<0> : tensor<128x32xi32>
    %ap = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
    %aa = tt.addptr %ap, %ao : tensor<64x32x!tt.ptr<f16>>, tensor<64x32xi32>
    %bp = tt.splat %w_ptr : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
    %ba = tt.addptr %bp, %bo : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi32>
    %a = tt.load %aa : tensor<64x32x!tt.ptr<f16>>
    %w = tt.load %ba : tensor<128x32x!tt.ptr<f16>>
    %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<128x32xf16> -> tensor<32x128xf16>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %d = tt.dot %a, %wt, %acc : tensor<64x32xf16> * tensor<32x128xf16> -> tensor<64x128xf32>
    tt.return
  }
}

// OVER: error: no legal power-of-two tile fits: M=64 N=128 K=32 dtype_bytes=2 wram_bytes=128 mram_bytes=4294967296 dma_align=64; smallest tried was M=1 N=1 K=1
// OK-DAG: "pim.tile-m" = 64 : i64
// OK-DAG: "pim.tile-n" = 128 : i64
// OK-DAG: "pim.tile-k" = 32 : i64
// OK-DAG: "pim.tile-wram-bytes" = 28672 : i64
// OK: "pim.wram-bytes-used" =
