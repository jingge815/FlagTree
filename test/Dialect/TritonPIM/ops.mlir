// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// Round-trip of every PIM op and type through the parser and printer.

// CHECK-LABEL: @hierarchy_queries
module attributes {"pim.num-dpus" = 4 : i32, "pim.num-tasklets" = 16 : i32} {
  tt.func @hierarchy_queries() -> i32 {
    // CHECK: %[[TID:.*]] = pim.tasklet_id : i32
    %tid = pim.tasklet_id : i32
    // CHECK: %[[DID:.*]] = pim.dpu_id : i32
    %did = pim.dpu_id : i32
    %sum = arith.addi %tid, %did : i32
    tt.return %sum : i32
  }
}

// -----

// CHECK-LABEL: @wram_alloc_and_access
module {
  tt.func @wram_alloc_and_access(%t: tensor<64x32xf16>) -> tensor<64x32xf16> {
    // CHECK: %[[BUF:.*]] = pim.wram_alloc : !pim.memdesc<64x32xf16, #pim.wram>
    %buf = pim.wram_alloc : !pim.memdesc<64x32xf16, #pim.wram>
    // CHECK: pim.wram_store %{{.*}}, %[[BUF]] : tensor<64x32xf16> -> !pim.memdesc<64x32xf16, #pim.wram>
    pim.wram_store %t, %buf : tensor<64x32xf16> -> !pim.memdesc<64x32xf16, #pim.wram>
    // CHECK: pim.barrier
    pim.barrier
    // CHECK: pim.wram_load %[[BUF]] : !pim.memdesc<64x32xf16, #pim.wram> -> tensor<64x32xf16>
    %r = pim.wram_load %buf : !pim.memdesc<64x32xf16, #pim.wram> -> tensor<64x32xf16>
    tt.return %r : tensor<64x32xf16>
  }
}

// -----

// An alignment attribute and an immutable buffer both survive round-trip.
// CHECK-LABEL: @alloc_variants
module {
  tt.func @alloc_variants() {
    // CHECK: pim.wram_alloc {alignment = 16 : i32} : !pim.memdesc<128xi32, #pim.wram>
    %a = pim.wram_alloc {alignment = 16 : i32} : !pim.memdesc<128xi32, #pim.wram>
    // CHECK: pim.wram_alloc : !pim.memdesc<128xi32, #pim.wram, immutable>
    %b = pim.wram_alloc : !pim.memdesc<128xi32, #pim.wram, immutable>
    tt.return
  }
}

// -----

// DMA in both directions, with and without the optional operands.
// CHECK-LABEL: @dma_plain
module {
  tt.func @dma_plain(%ptrs: tensor<64x!tt.ptr<f32>>) {
    %buf = pim.wram_alloc : !pim.memdesc<64xf32, #pim.wram>
    // CHECK: pim.dma_load %{{.*}} -> %{{.*}} : tensor<64x!tt.ptr<f32>> -> !pim.memdesc<64xf32, #pim.wram>
    pim.dma_load %ptrs -> %buf : tensor<64x!tt.ptr<f32>> -> !pim.memdesc<64xf32, #pim.wram>
    pim.barrier
    // CHECK: pim.dma_store %{{.*}} -> %{{.*}} : !pim.memdesc<64xf32, #pim.wram> -> tensor<64x!tt.ptr<f32>>
    pim.dma_store %buf -> %ptrs : !pim.memdesc<64xf32, #pim.wram> -> tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: @dma_masked
module {
  tt.func @dma_masked(%ptrs: tensor<32x!tt.ptr<f32>>, %mask: tensor<32xi1>,
                      %fill: tensor<32xf32>) {
    %buf = pim.wram_alloc : !pim.memdesc<32xf32, #pim.wram>
    // CHECK: pim.dma_load %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}} {contiguous_dim = 0 : i64, elem_stride = 1 : i64}
    pim.dma_load %ptrs, %mask, %fill -> %buf
        {contiguous_dim = 0 : i64, elem_stride = 1 : i64}
        : tensor<32x!tt.ptr<f32>> -> !pim.memdesc<32xf32, #pim.wram>
    pim.barrier
    // CHECK: pim.dma_store %{{.*}} -> %{{.*}}, %{{.*}} {base_arg = 0 : i64
    pim.dma_store %buf -> %ptrs, %mask
        {base_arg = 0 : i64, contiguous_dim = 0 : i64, elem_stride = 1 : i64}
        : !pim.memdesc<32xf32, #pim.wram> -> tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// The tasklet layout encoding and a relayout between two of them.
// The all-ones dpusPerDevice is elided on print, since a single-DPU kernel is
// the common case; the two layouts differ only in which axis the tasklets span.
#src = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [1, 16], dpusPerDevice = [1, 1], order = [1, 0]}>
#dst = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [1, 1], order = [1, 0]}>
// CHECK-LABEL: @relayout
module {
  tt.func @relayout(%t: tensor<16x16xf32, #src>) -> tensor<16x16xf32, #dst> {
    // CHECK: pim.convert_layout
    // CHECK-SAME: taskletsPerDpu = [1, 16], order = [1, 0]
    // CHECK-SAME: taskletsPerDpu = [16, 1], order = [1, 0]
    %r = pim.convert_layout %t : tensor<16x16xf32, #src> -> tensor<16x16xf32, #dst>
    tt.return %r : tensor<16x16xf32, #dst>
  }
}

// -----

// An MRAM-space descriptor, for hand-written IR that names main memory.
// CHECK-LABEL: @mram_desc
module {
  tt.func @mram_desc(%d: !pim.memdesc<1024xf32, #pim.mram>) {
    // CHECK: !pim.memdesc<1024xf32, #pim.mram>
    tt.return
  }
}
