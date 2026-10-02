// RUN: triton-opt %s -pim-verify-gml-contract 2>&1 | FileCheck %s

// A split that lands only on a tensor of rank <= the placement's `dim` is still
// a recorded split.
//
// `verifyLayoutsMatchPlacement` skips same-axis comparison for those tensors --
// a rank-1 vector has no dim-1 to compare -- but the module-level fallback used
// to count `splitApplied` *after* that skip, so a module whose only split lived
// on such a tensor was reported as "no tensor layout records a split" and
// rejected. Measured: the module below converted with rc=1 before the counter
// moved ahead of the skip. Rejecting a legal module is worse than missing an
// illegal one -- it forces upstream to rewrite something that was already right.

// CHECK-NOT: error
// CHECK: dpusPerDevice = [2]
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
  tt.func public @the_split_lives_on_a_rank_one_tensor(
      %v: tensor<64xf16, #pim.tasklet_tiled<{sizePerTasklet = [1], taskletsPerDpu = [1], dpusPerDevice = [2], order = [0]}> >) {
    tt.return
  }
}
