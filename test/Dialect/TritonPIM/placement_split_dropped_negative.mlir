// RUN: not triton-opt %s -convert-triton-to-pim='target=pim:v1 num-dpus=8 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' 2>&1 | FileCheck %s

// The split reached no tensor at all.
//
// Per-tensor agreement cannot catch this. All-ones is waved through there, and
// for good reason: index vectors and the broadcast columns derived from them are
// legitimately undistributed inside a sharded kernel. But that makes a module
// where EVERY encoding lost the split indistinguishable from a module of tensors
// that were never split -- and downstream then costs the operator as if it were
// not sharded. Measured before the module-level check existed: this exact module
// converted with rc=0 and zero `dpusPerDevice` in the output.
//
// So the invariant is checked once per module, not per tensor: a `shard`
// placement over N > 1 DPUs must leave at least one non-all-ones encoding
// behind. Only for `shard` -- replicate and partial keep the whole shape on
// every DPU, so all-ones everywhere is their correct answer (covered by
// `placement_to_layout.mlir`).

// CHECK: no tensor layout in the module records a split
#all_ones = #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1], dpusPerDevice = [1, 1], order = [1, 0]}>
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
  tt.func public @the_split_was_dropped_everywhere(%t: tensor<4x32xf16, #all_ones>) {
    tt.return
  }
}
