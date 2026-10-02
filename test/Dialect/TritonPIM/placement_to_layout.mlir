// RUN: triton-opt %s -split-input-file -convert-triton-to-pim='target=pim:v1 num-dpus=8 num-tasklets=16 wram-bytes=65536 mram-bytes=4294967296 dma-align=8' | FileCheck %s

// The Placement dimension travelling into the layout encoding.
//
// Before this existed, `dpusPerDevice` was reachable only by hand-written MLIR:
// the sole AttrBuilder discarded its `numDpus` argument (`(void)numDpus;`) on
// the grounds that kernels are single-DPU, and that argument is a *hardware*
// count anyway -- writing it in would have claimed every tensor was split
// `num-dpus` ways, which is false for the replicated ones. So no FlagTree code
// path could produce a non-all-ones value, and the graph compiler's sharding
// decision had nowhere to land. `pim.placement` on the module is that landing
// point, and this test is the proof it arrives.

// A shard of dim 1 over 2 DPUs becomes dpusPerDevice = [1, 2] on every tensor.
// CHECK-LABEL: @shard_reaches_the_encoding
// CHECK: dpusPerDevice = [1, 2]
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
  tt.func public @shard_reaches_the_encoding() {
    %c = arith.constant dense<0.0> : tensor<4x32xf16>
    tt.return
  }
}

// -----

// The shard axis is part of the value, not a convention: dim 0 lands on the
// other position.
// CHECK-LABEL: @the_axis_is_part_of_the_value
// CHECK: dpusPerDevice = [4, 1]
module attributes {pim.placement = #pim.placement<kind = shard, dim = 0, numDpus = 4>} {
  tt.func public @the_axis_is_part_of_the_value() {
    %c = arith.constant dense<0.0> : tensor<8x32xf16>
    tt.return
  }
}

// -----

// Replicate leaves the encoding all-ones -- every DPU holds the whole shape, so
// no axis of *this tensor* is split. The printer elides all-ones, so the field
// must not appear at all. This is also why the copy count cannot be recovered
// from the encoding and `#pim.placement` has to carry it separately.
// CHECK-LABEL: @replicate_splits_no_axis
// CHECK-NOT: dpusPerDevice
module attributes {pim.placement = #pim.placement<kind = replicate, numDpus = 4>} {
  tt.func public @replicate_splits_no_axis() {
    %c = arith.constant dense<0.0> : tensor<4x32xf16>
    tt.return
  }
}

// -----

// Partial likewise: each DPU holds a partial value over the whole shape.
// CHECK-LABEL: @partial_splits_no_axis
// CHECK-NOT: dpusPerDevice
module attributes {pim.placement = #pim.placement<kind = partial, numDpus = 2, reduce = sum>} {
  tt.func public @partial_splits_no_axis() {
    %c = arith.constant dense<0.0> : tensor<4x32xf16>
    tt.return
  }
}

// -----

// No placement: byte for byte what this pass produced before placements existed.
// CHECK-LABEL: @no_placement_is_unchanged
// CHECK-NOT: dpusPerDevice
module {
  tt.func public @no_placement_is_unchanged() {
    %c = arith.constant dense<0.0> : tensor<4x32xf16>
    tt.return
  }
}

// -----

// A mixed-rank kernel: the same `dim` lands on different physical axes.
//
// `dim` is indexed from the outside and every tensor with rank > dim is split,
// so a rank-2 tensor gets `[1, 2]` (axis 1 is the innermost) while a rank-3 one
// gets `[1, 2, 1]` (axis 1 is now the middle). The graph compiler states the
// axis in its own kernel coordinates -- a rank-2 view -- so there is no single
// axis all ranks can agree on, and only the width is invariant. That is exactly
// why `verifyAgreesWith` compares the width and not the axis.
//
// This block pins the behaviour so the convention is written down somewhere
// executable; it is a record, not an endorsement of mixing ranks in one kernel.
// CHECK-LABEL: @dim_indexes_from_the_outside
// CHECK: tensor<4x32xf16, #pim.tasklet_tiled<{{.*}}dpusPerDevice = [1, 2]
// CHECK: tensor<2x4x32xf16, #pim.tasklet_tiled<{{.*}}dpusPerDevice = [1, 2, 1]
module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 2>} {
  tt.func public @dim_indexes_from_the_outside() {
    %a = arith.constant dense<0.0> : tensor<4x32xf16>
    %b = arith.constant dense<0.0> : tensor<2x4x32xf16>
    tt.return
  }
}
