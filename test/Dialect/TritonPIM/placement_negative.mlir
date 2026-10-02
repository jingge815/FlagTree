// RUN: triton-opt %s -split-input-file -verify-diagnostics

// The Placement verifier. Negatives live in their own file because
// `-verify-diagnostics` makes an error the expected result, and the positive
// file's RUN lines do not carry that flag -- mixing the two would make
// `triton-opt` exit non-zero there and lit (pipefail) would fail the whole line.
//
// RUN-line form follows `operator_ops_negative.mlir`: every expected-error is
// matched, so triton-opt exits zero and needs neither `not` nor FileCheck.

// A shard splits a specific axis, so it has to name one.
// expected-error @below {{dim must be non-negative}}
#no_dim = #pim.placement<kind = shard, numDpus = 2>
module {
  tt.func @shard_needs_a_dim() attributes {p = #no_dim} { tt.return }
}

// -----

// A shard over one DPU is not a split -- that is what replicate means.
// expected-error @below {{a shard over a single DPU is not a split}}
#one = #pim.placement<kind = shard, dim = 0, numDpus = 1>
module {
  tt.func @shard_of_one() attributes {p = #one} { tt.return }
}

// -----

// A sharded tensor is whole once gathered; it owes no reduction.
// expected-error @below {{a shard placement owes no reduction}}
#sh_red = #pim.placement<kind = shard, dim = 0, numDpus = 2, reduce = sum>
module {
  tt.func @shard_with_reduce() attributes {p = #sh_red} { tt.return }
}

// -----

// Replicate holds every axis whole, so naming an axis is meaningless.
// expected-error @below {{must not name a dim}}
#rep_dim = #pim.placement<kind = replicate, dim = 1, numDpus = 2>
module {
  tt.func @replicate_with_dim() attributes {p = #rep_dim} { tt.return }
}

// -----

// The reduction is the whole content of "partial": without it there is no way
// to turn the per-DPU pieces back into the value.
// expected-error @below {{must say how its pieces combine}}
#part_bare = #pim.placement<kind = partial, numDpus = 2>
module {
  tt.func @partial_without_reduce() attributes {p = #part_bare} { tt.return }
}

// -----

// A partial result is split over the contraction, not over an axis of the result.
// expected-error @below {{split over the contraction}}
#part_dim = #pim.placement<kind = partial, dim = 0, numDpus = 2, reduce = sum>
module {
  tt.func @partial_with_dim() attributes {p = #part_dim} { tt.return }
}

// -----

// numDpus counts the DPUs that hold the tensor, so zero is not a count.
// expected-error @below {{must be at least 1}}
#zero = #pim.placement<kind = replicate, numDpus = 0>
module {
  tt.func @zero_dpus() attributes {p = #zero} { tt.return }
}

// -----

// A pipeline stage index cannot be negative.
// expected-error @below {{cannot be negative}}
#neg_stage = #pim.placement<kind = replicate, numDpus = 2, stage = -1>
module {
  tt.func @negative_stage() attributes {p = #neg_stage} { tt.return }
}

// -----

// An id list that disagrees with the count would leave two answers to "how many
// DPUs" with nothing to say which is right.
// expected-error @below {{two answers to the same question}}
#mismatch = #pim.placement<kind = shard, dim = 0, numDpus = 2, dpuIds = [1, 2, 3]>
module {
  tt.func @ids_disagree_with_count() attributes {p = #mismatch} { tt.return }
}

// -----

// The same DPU cannot hold two of the shards.
// expected-error @below {{duplicate DPU id}}
#dup = #pim.placement<kind = shard, dim = 0, numDpus = 2, dpuIds = [1, 1]>
module {
  tt.func @duplicate_ids() attributes {p = #dup} { tt.return }
}

// -----

// expected-error @below {{negative DPU id}}
#neg_id = #pim.placement<kind = shard, dim = 0, numDpus = 2, dpuIds = [-1, 0]>
module {
  tt.func @negative_id() attributes {p = #neg_id} { tt.return }
}
