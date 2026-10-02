// RUN: triton-opt %s -split-input-file | FileCheck %s
// RUN: triton-opt %s -split-input-file -verify-diagnostics -o /dev/null
// Round-trip: printing the parsed encoding reproduces the same text, so a
// value written by the graph compiler survives a parse/print cycle unchanged.
// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// The cross-DPU half of the tasklet layout encoding: `dpusPerDevice` says which
// axis of a tensor is split across how many DPUs. The graph compiler writes it
// from its own sharding decision (tensor-parallel degree), so a non-all-ones
// value has to survive parsing and printing unchanged -- an all-ones value is
// elided on print, which is exactly why the non-trivial case needs its own test.

// tp=2 on the outer axis. The field is printed because it is not all ones.
// CHECK-LABEL: @dpus_on_axis_0
// CHECK: dpusPerDevice = [2, 1]
#tp2_dim0 = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [2, 1], order = [1, 0]}>
module attributes {pim.target = "pim:v1"} {
  tt.func @dpus_on_axis_0(%t: tensor<2048x4096xf16, #tp2_dim0>) {
    tt.return
  }
}

// -----

// Same degree, inner axis -- the axis carrying the split is part of the value,
// not a convention, so both placements must round-trip.
// CHECK-LABEL: @dpus_on_axis_1
// CHECK: dpusPerDevice = [1, 2]
#tp2_dim1 = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [1, 2], order = [1, 0]}>
module attributes {pim.target = "pim:v1"} {
  tt.func @dpus_on_axis_1(%t: tensor<1x4096xf16, #tp2_dim1>) {
    tt.return
  }
}

// -----

// Rank 3 with tp=4, and the module-level hardware attributes the graph compiler
// emits alongside it. The four dimensions travel together: op semantics (the op
// itself), dtype (f16), placement (`dpusPerDevice`), memory layout (the rest of
// the encoding plus `pim.dma-align`).
#tp4 = #pim.tasklet_tiled<{sizePerTasklet = [1, 1, 1], taskletsPerDpu = [16, 1, 1], dpusPerDevice = [1, 4, 1], order = [2, 1, 0]}>
// The module header prints before the function, so check it first.
// CHECK: "pim.dma-align" = 64
// CHECK-SAME: "pim.num-dpus" = 4
// CHECK-LABEL: @four_dimensions_together
// CHECK: dpusPerDevice = [1, 4, 1]
module attributes {pim.target = "pim:v1", "pim.num-dpus" = 4 : i32,
                   "pim.num-tasklets" = 16 : i32, "pim.dma-align" = 64 : i32} {
  tt.func @four_dimensions_together(%t: tensor<1x16x4096xf16, #tp4>) {
    tt.return
  }
}

// -----

// A single-DPU encoding prints without the field. This is the rule the graph
// compiler's byte-for-byte test leans on, so it is worth pinning here too.
// CHECK-LABEL: @all_ones_is_elided
// CHECK-NOT: dpusPerDevice
#single = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [1, 1], order = [1, 0]}>
module attributes {pim.target = "pim:v1"} {
  tt.func @all_ones_is_elided(%t: tensor<2048x4096xf16, #single>) {
    tt.return
  }
}
