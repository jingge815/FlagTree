// RUN: triton-opt %s -split-input-file | FileCheck %s
// Round-trip: a value the graph compiler wrote survives parse/print unchanged.
// RUN: triton-opt %s -split-input-file | triton-opt -split-input-file | FileCheck %s

// The Placement dimension: which DPUs hold a tensor and in what relationship to
// the whole. `#pim.tasklet_tiled`'s `dpusPerDevice` is the per-axis arithmetic;
// this attribute carries what that field cannot say -- replicate versus a
// single DPU (both all-ones), and the reduction a partial result still owes.

// CHECK: kind = shard, dim = 1, numDpus = 2
// CHECK-LABEL: @shard_names_an_axis
#shard = #pim.placement<kind = shard, dim = 1, numDpus = 2>
module attributes {pim.target = "pim:v1", pim.placement = #shard} {
  tt.func @shard_names_an_axis() {
    tt.return
  }
}

// -----

// Replicate carries no dim and no reduce: every DPU holds the whole shape.
// `numDpus` is how many copies exist, which is why it is still needed -- the
// layout encoding is all-ones here and cannot distinguish 4 copies from 1.
// CHECK: kind = replicate, numDpus = 4
// CHECK-LABEL: @replicate_holds_the_whole_shape
#rep = #pim.placement<kind = replicate, numDpus = 4>
module attributes {pim.target = "pim:v1", pim.placement = #rep} {
  tt.func @replicate_holds_the_whole_shape() {
    tt.return
  }
}

// -----

// Partial owes a reduction. The split is over the contraction that produced the
// value, not over an axis of the result, so there is no dim.
// CHECK: kind = partial, numDpus = 2, reduce = sum
// CHECK-LABEL: @partial_owes_a_reduction
#part = #pim.placement<kind = partial, numDpus = 2, reduce = sum>
module attributes {pim.target = "pim:v1", pim.placement = #part} {
  tt.func @partial_owes_a_reduction() {
    tt.return
  }
}

// -----

// Explicit DPU ids and a pipeline stage: a stage that owns DPUs 4..7 cannot be
// described by a count alone. `stage` is carried rather than derived because two
// stages can own disjoint DPU sets of the same size.
// CHECK: kind = shard, dim = 0, numDpus = 2, dpuIds = [4, 5], stage = 1
// CHECK-LABEL: @explicit_dpus_and_stage
#staged = #pim.placement<kind = shard, dim = 0, numDpus = 2, dpuIds = [4, 5], stage = 1>
module attributes {pim.target = "pim:v1", pim.placement = #staged} {
  tt.func @explicit_dpus_and_stage() {
    tt.return
  }
}

// -----

// Defaults are elided on print, so the single-DPU form stays as short as it was
// before this attribute existed.
// CHECK: #pim.placement<kind = replicate>
// CHECK-LABEL: @defaults_are_elided
// CHECK-NOT: dpuIds
#plain = #pim.placement<kind = replicate, numDpus = 1, stage = 0>
module attributes {pim.target = "pim:v1", pim.placement = #plain} {
  tt.func @defaults_are_elided() {
    tt.return
  }
}