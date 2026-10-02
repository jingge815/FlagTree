// RUN: triton-opt %s -verify-diagnostics -pim-verify-gml-contract

// The two carriers of the Placement dimension disagree, and the B path never
// runs the pass that used to be the only place this was checked.
//
// `verifyLayoutsMatchPlacement` lives at the exit of `-convert-triton-to-pim`,
// which the operator-level chain (`-pim-fuse-activation -pim-expand-phases
// -pim-verify-gml-contract`) does not pass through. So a module whose
// `#pim.placement` says one split and whose tensor encodings record another
// used to sail through that chain with no diagnostic -- measured: the module
// below converted with rc=0. The check now also runs here, at the chain's
// exit, so both paths enforce the same invariant.
//
// The placement sits on the module, which is where the graph compiler writes
// it; the encoding spreads over 2 DPUs while the module says 4.


module attributes {pim.placement = #pim.placement<kind = shard, dim = 1, numDpus = 4>} {
  tt.func @the_encoding_disagrees_with_the_placement(
      %a: tensor<1x64xf16, #pim.tasklet_tiled<{sizePerTasklet = [1, 1],
          taskletsPerDpu = [4, 1], dpusPerDevice = [1, 2], order = [1, 0]}>>) {
    // expected-error @below {{placement shards over 4 DPUs but the layout encoding spreads over 2}}
    %0 = pim.eltwise %a, %a
       {kind = #pim.eltwise<mul>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}
       : tensor<1x64xf16, #pim.tasklet_tiled<{sizePerTasklet = [1, 1],
           taskletsPerDpu = [4, 1], dpusPerDevice = [1, 2], order = [1, 0]}>>,
         tensor<1x64xf16, #pim.tasklet_tiled<{sizePerTasklet = [1, 1],
           taskletsPerDpu = [4, 1], dpusPerDevice = [1, 2], order = [1, 0]}>>
       -> tensor<1x64xf16, #pim.tasklet_tiled<{sizePerTasklet = [1, 1],
           taskletsPerDpu = [4, 1], dpusPerDevice = [1, 2], order = [1, 0]}>>
    tt.return
  }
}
