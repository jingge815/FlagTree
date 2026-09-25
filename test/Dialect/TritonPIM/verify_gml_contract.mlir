// RUN: triton-opt %s -split-input-file -verify-diagnostics -pim-verify-gml-contract

// The invariants that span more than one attribute. Every case here is one
// that produces a valid-looking module and a graph that loads, so nothing else
// in the pipeline would catch it.

// -----

// A gap in the phase numbering means a consumer counting phases by distinct
// index gets fewer than the expansion meant, and the corresponding field
// family quietly disappears from the graph.
module {
  tt.func @phase_index_gap(%x: tensor<1x32xf16>) {
    %a = pim.lut %x {isPhased, kind = #pim.activation<exp>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = cstl>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    // expected-error @below {{phase indices must run 0..1 with no gap; 1 is missing}}
    %b = pim.lut %a {isPhased, kind = #pim.activation<reciprocal>, phases = [#pim.phase_spec<index = 2, bytes = 64, unit = cstl>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// Two ops claiming one phase number is two traversals folded into one field
// family. It also catches two separate chains sharing a function: each starts
// at 0, so the second would repeat the first's indices.
module {
  tt.func @phase_index_repeat(%x: tensor<1x32xf16>) {
    %a = pim.lut %x {isPhased, kind = #pim.activation<exp>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = cstl>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    // expected-error @below {{phase 0 is claimed more than once}}
    %b = pim.lut %a {isPhased, kind = #pim.activation<exp>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = cstl>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// The one that matters. Dynamic quantization's phase 1 must read phase 0.
// Serializing the identity scale and the reciprocal as a chain produces a graph
// that loads and a reciprocal that is wrong by 256x.
module {
  tt.func @dq_phase_one_must_read_zero(%x: tensor<1x4096xf16>) {
    %p0 = pim.global_pool %x {isPhased, groupSize = 128 : i64, kind = #pim.pool_kind<absmax>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu>]} : tensor<1x4096xf16> -> tensor<1x32xf16>
    // expected-error @below {{dynamic quantization's phase 1 must read phase 0}}
    %p1 = pim.lut %p0 {isPhased, kind = #pim.activation<identity>, phases = [#pim.phase_spec<index = 1, bytes = 64, unit = cstl>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    %p2 = pim.lut %p0 {isPhased, kind = #pim.activation<reciprocal>, phases = [#pim.phase_spec<index = 2, bytes = 64, unit = cstl, reads = [0]>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// And phase 2 likewise: it reads p0, not p1 -- p1 is p0/256, and taking the
// reciprocal of that is off by 256x.
module {
  tt.func @dq_phase_two_must_read_zero(%x: tensor<1x4096xf16>) {
    %p0 = pim.global_pool %x {isPhased, groupSize = 128 : i64, kind = #pim.pool_kind<absmax>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu>]} : tensor<1x4096xf16> -> tensor<1x32xf16>
    %p1 = pim.lut %p0 {isPhased, kind = #pim.activation<identity>, phases = [#pim.phase_spec<index = 1, bytes = 64, unit = cstl, reads = [0]>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    // expected-error @below {{dynamic quantization's phase 2 must read phase 0}}
    %p2 = pim.lut %p1 {isPhased, kind = #pim.activation<reciprocal>, phases = [#pim.phase_spec<index = 2, bytes = 64, unit = cstl, reads = [1]>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// Normalization runs on the vector unit; carrying a datapath configuration
// would emit keys the reference graph does not have for this operator.
module {
  tt.func @normalize_carries_no_datapath(%x: tensor<4x16xf16>, %w: tensor<16xf16>) {
    // expected-error @below {{must not carry datapath}}
    %n = pim.normalize %x, %w {axis = -1 : i64, rmsNorm, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>} : tensor<4x16xf16>, tensor<16xf16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

module {
  tt.func @normalize_carries_no_fpsu(%x: tensor<4x16xf16>, %w: tensor<16xf16>) {
    // expected-error @below {{must not carry fpsu}}
    %n = pim.normalize %x, %w {axis = -1 : i64, rmsNorm, fpsu = #pim.fpsu_spec<mode = floating_point>} : tensor<4x16xf16>, tensor<16xf16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

module {
  tt.func @normalize_carries_no_pooling(%x: tensor<4x16xf16>, %w: tensor<16xf16>) {
    // expected-error @below {{must not carry "pooling"}}
    %n = pim.normalize %x, %w {axis = -1 : i64, rmsNorm, "pooling" = 4 : i64} : tensor<4x16xf16>, tensor<16xf16> -> tensor<4x16xf16>
    tt.return
  }
}

// -----

// A well-formed chain comes through untouched -- the pass reads, it does not
// rewrite, so what it accepts is exactly what expansion produced.
module {
  tt.func @well_formed_module_passes_through(%x: tensor<1x4096xf16>) {
    %p0 = pim.global_pool %x {isPhased, groupSize = 128 : i64, kind = #pim.pool_kind<absmax>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu>]} : tensor<1x4096xf16> -> tensor<1x32xf16>
    %p1 = pim.lut %p0 {isPhased, kind = #pim.activation<identity>, phases = [#pim.phase_spec<index = 1, bytes = 64, unit = activation, reads = [0]>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    %p2 = pim.lut %p0 {isPhased, kind = #pim.activation<reciprocal>, phases = [#pim.phase_spec<index = 2, bytes = 64, unit = activation, reads = [0]>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// An expanded phase must name the block it runs on. `cstl` stood for six of
// them (fpsu / kantor / pooling / activation / combiner / store), and under one
// name cost extraction cannot tell a table lookup from a rescale from a grouped
// reduction -- the fields for all three get filled from whichever rule matched
// first, with no error anywhere.
//
// This is the one contract rule an attribute verifier cannot make: `cstl` is
// still legal on un-expanded IR (its ordinal is frozen so old assembly keeps
// round-tripping), so the rule is "not on an op that just came out of the
// expansion" -- and only a pass scheduled after the expansion knows that.
module {
  tt.func @expanded_phase_must_not_stay_on_cstl(%x: tensor<1x4096xf16>) {
    // expected-error @below {{phase 0 runs on `cstl`, which names six different blocks at once}}
    %p0 = pim.global_pool %x {isPhased, groupSize = 128 : i64, kind = #pim.pool_kind<absmax>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = cstl>]} : tensor<1x4096xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// Phase numbers are per chain, not per function. Two independent operators
// each numbering from 0 is the legal encoding -- it is how the graph format
// writes `*_phase_<k>` on two different nodes. A function-wide uniqueness
// check would reject this.
module {
  tt.func @two_independent_chains_may_reuse_indices(%a: tensor<8x16xf16>, %b: tensor<8x16xf16>) {
    %p0 = pim.reduce_axis %a {axis = 1 : i64, kind = #pim.eltwise<max>, unit = #pim.unit<vpu>,
        isPhased, phases = [#pim.phase_spec<index = 0, bytes = 16, unit = vpu>]}
        : tensor<8x16xf16> -> tensor<8x1xf16>
    %p1 = pim.lut %p0 {kind = #pim.activation<exp>, isPhased,
        phases = [#pim.phase_spec<index = 1, bytes = 16, unit = activation, reads = [0]>]}
        : tensor<8x1xf16> -> tensor<8x1xf16>
    %q0 = pim.reduce_axis %b {axis = 1 : i64, kind = #pim.eltwise<max>, unit = #pim.unit<vpu>,
        isPhased, phases = [#pim.phase_spec<index = 0, bytes = 16, unit = vpu>]}
        : tensor<8x16xf16> -> tensor<8x1xf16>
    %q1 = pim.lut %q0 {kind = #pim.activation<exp>, isPhased,
        phases = [#pim.phase_spec<index = 1, bytes = 16, unit = activation, reads = [0]>]}
        : tensor<8x1xf16> -> tensor<8x1xf16>
    tt.return
  }
}

// -----

// 同一函数里 DQ 链与 Softmax 链共存必须通过。相 0 必须是 global_pool 才
// 真的触发扇出规则：旧用例用 reduce_axis，恰好绕开这条检查。
module {
  tt.func @dq_and_softmax_chains_in_one_function(%x: tensor<1x4096xf16>, %s: tensor<8x16xf16>) {
    %p0 = pim.global_pool %x {isPhased, groupSize = 128 : i64, kind = #pim.pool_kind<absmax>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = pooling>]} : tensor<1x4096xf16> -> tensor<1x32xf16>
    %p1 = pim.lut %p0 {isPhased, kind = #pim.activation<identity>, phases = [#pim.phase_spec<index = 1, bytes = 64, unit = activation, reads = [0]>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    %p2 = pim.lut %p0 {isPhased, kind = #pim.activation<reciprocal>, phases = [#pim.phase_spec<index = 2, bytes = 64, unit = activation, reads = [0]>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    %q0 = pim.reduce_axis %s {axis = 1 : i64, kind = #pim.eltwise<max>, unit = #pim.unit<vpu>,
        isPhased, phases = [#pim.phase_spec<index = 0, bytes = 16, unit = vpu>]}
        : tensor<8x16xf16> -> tensor<8x1xf16>
    %q1 = pim.lut %q0 {kind = #pim.activation<exp>, isPhased,
        phases = [#pim.phase_spec<index = 1, bytes = 16, unit = activation>]}
        : tensor<8x1xf16> -> tensor<8x1xf16>
    tt.return
  }
}
