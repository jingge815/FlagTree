// RUN: not triton-opt %s -split-input-file -convert-triton-to-pim='target=pim:v1 num-dpus=8 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' 2>&1 | FileCheck %s

// The Placement dimension has two carriers: `#pim.placement` on the module (what
// the graph compiler decided) and `dpusPerDevice` inside each tensor's layout
// encoding (what conversion wrote). Two carriers for one fact can drift, and a
// dropped or altered split costs downstream as if the tensor were unsharded --
// silently, because both halves parse fine on their own. The conversion pass
// compares them on the way out.

// CHECK: placement shards over 2 DPUs but the layout encoding spreads over 4
#wrong_count = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [1, 4], order = [1, 0]}>
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
  tt.func public @encoding_splits_too_many(%t: tensor<4x32xf16, #wrong_count>) {
    tt.return
  }
}

// -----

// Replicate means every DPU holds the whole shape, so no axis may be split.
// CHECK: a replicate placement holds the whole shape on each DPU
#splits = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [2, 1], order = [1, 0]}>
module attributes {pim.placement = #pim.placement<kind = replicate, numDpus = 2>} {
  tt.func public @replicate_must_not_split(%t: tensor<4x32xf16, #splits>) {
    tt.return
  }
}

// -----

// The axis does NOT have to match, and this case documents why: inside a kernel
// `tt.trans` permutes the encoding, so a tensor the graph compiler sharded on
// dim 1 legitimately appears as `dpusPerDevice = [2, 1]` after a transpose
// (measured on a real tp2 `linear`, which transposes its weight block). Only the
// split *width* is invariant, and that is what the check compares.
//
// So this module must be ACCEPTED. It is here as a negative-file member because
// it guards the opposite mistake: a check that pins the axis would reject it,
// and that rejection broke a correct compile.
// CHECK-NOT: error
#transposed = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [2, 1], order = [1, 0]}>
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
  tt.func public @a_transposed_split_is_fine(%t: tensor<4x32xf16, #transposed>) {
    tt.return
  }
}
