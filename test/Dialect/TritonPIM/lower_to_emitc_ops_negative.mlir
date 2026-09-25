// RUN: triton-opt %s -split-input-file -verify-diagnostics -pim-lower-to-emitc

// Every case here used to succeed and emit nothing: the pass hit an op it did
// not know, returned success, and the generated C silently computed a
// different function than the IR described.

// -----

// An opaque operator fed straight to the lowering means `-pim-expand-phases`
// was skipped. There is no C for it and there must not be a quiet fallthrough.
module {
  tt.func @unexpanded_softmax(%s: tensor<1x1024xf16>) {
    // expected-error @below {{does not lower the operator-level op pim.softmax yet}}
    %p = pim.softmax %s {axis = 1 : i64, unit = #pim.unit<cstl>} : tensor<1x1024xf16> -> tensor<1x1024xf16>
    tt.return
  }
}

// -----

// An unexpanded dynamic quantize would otherwise be treated as dead code, and
// the kernel would produce whatever the surrounding chain happened to leave in
// the output buffer.
module {
  tt.func @unexpanded_dynamic_quant(%x: tensor<1x4096xf16>, %s: tensor<32xf16>) {
    // expected-error @below {{an unexpanded dynamic quantize reached pim-lower-to-emitc; run -pim-expand-phases first}}
    %q = pim.quantize %x, %s {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>} : tensor<1x4096xf16>, tensor<32xf16> -> tensor<1x4096xi8>
    tt.return
  }
}

// -----

// 半旋转换的是"两半"，末维是奇数就没有两半可换。
module {
  tt.func @rotate_half_needs_even_width(%x: tensor<1x7xf16>, %c: tensor<1x7xf16>) {
    // expected-error @below {{rotateHalf needs an even last dimension}}
    %y = pim.eltwise %x, %c {kind = #pim.eltwise<mul>, rotateHalf, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, phases = [#pim.phase_spec<index = 0, bytes = 14, unit = kantor>]} : tensor<1x7xf16>, tensor<1x7xf16> -> tensor<1x7xf16>
    tt.return
  }
}

// -----

// 半旋转是 sin 项那次**乘法**的性质。挂在加法上说不出任何硬件行为。
module {
  tt.func @rotate_half_only_on_multiply(%x: tensor<1x8xf16>, %c: tensor<1x8xf16>) {
    // expected-error @below {{rotateHalf is the sin term's multiply}}
    %y = pim.eltwise %x, %c {kind = #pim.eltwise<add>, rotateHalf, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, phases = [#pim.phase_spec<index = 0, bytes = 16, unit = kantor>]} : tensor<1x8xf16>, tensor<1x8xf16> -> tensor<1x8xf16>
    tt.return
  }
}

// -----

// A reduction kind the pooling unit does not do, and this pass does not emit:
// guessing an order for it would produce a number that looks plausible.
module {
  tt.func @unknown_reduction_kind(%x: tensor<1x1024xf16>) {
    // expected-error @below {{has no C for reduction}}
    %m = pim.reduce_axis %x {kind = #pim.eltwise<div>, axis = 1 : i64, phases = [#pim.phase_spec<index = 0, bytes = 2, unit = vpu>]} : tensor<1x1024xf16> -> tensor<1x1xf16>
    tt.return
  }
}

// -----

// An activation kind with no C behind it. The kind is the mathematical meaning
// of the lookup, so a missing one has to be reported rather than approximated.
module {
  tt.func @unknown_activation_kind(%x: tensor<1x32xf16>) {
    // expected-error @below {{has no C for activation gelu}}
    %y = pim.lut %x {kind = #pim.activation<gelu>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = cstl>]} : tensor<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// A grouped reduction whose kind has no C: only absmax has a caller today.
module {
  tt.func @unknown_global_pool_kind(%x: tensor<1x4096xf16>) {
    // expected-error @below {{has no C for grouped reduction sum}}
    %m = pim.global_pool %x {groupSize = 128 : i64, kind = #pim.pool_kind<sum>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu>]} : tensor<1x4096xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

// 一个**可广播、但本 pass 的两种索引重映射都表达不了**的形态。
// `[2,6,8]` 对 `[1,6,8]` 是前导轴拉伸（支持）；`[2,6,8]` 对 `[2,6,1]` 是末维为 1
// （支持）。而 `[2,6,8]` 对 `[1,1,8]` 要求**同时**拉伸两个前导轴且尾部只对上末维
// —— 走"除以行宽"会读错行，走"按元素数取模"会读错头。任选一种都是静默算错。
module {
  tt.func @unsupported_broadcast(%a: tensor<2x6x8xf16>, %b: tensor<1x1x8xf16>) {
    // expected-error @below {{only broadcasts a scalar, a trailing-1 rhs, or a leading axis whose trailing shape matches}}
    %y = pim.eltwise %a, %b {kind = #pim.eltwise<mul>, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, phases = [#pim.phase_spec<index = 0, bytes = 192, unit = kantor>]} : tensor<2x6x8xf16>, tensor<1x1x8xf16> -> tensor<2x6x8xf16>
    tt.return
  }
}

// -----

// 三槽及以上：降级只读前两槽，第三槽会被丢在地上——生成的 C 会把三个输入里的
// 两个组合起来，而且看起来完全正常。必须报错。
module {
  tt.func @three_slot_eltwise_has_no_lowering(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>, %c: tensor<4x8xf16>) {
    // expected-error @below {{lowers the two-slot form; this op has 3 slots}}
    %y = pim.eltwise %a, %b, %c {kind = #pim.eltwise<add>, datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = kantor>]} : tensor<4x8xf16>, tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}

// -----

// A per-slot datapath that is ignored would scale both slots the same way.
// Must fail loudly rather than emit C that looks well-formed.
module {
  tt.func @per_slot_datapath_is_not_silently_dropped(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>) {
    %y = pim.eltwise %a, %b
    // expected-error @-1 {{does not apply per-slot datapaths yet}}
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        perSlotDatapath = [#pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
                           #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>],
        phases = [#pim.phase_spec<index = 0, bytes = 64, unit = kantor>]}
       : tensor<4x8xf16>, tensor<4x8xf16> -> tensor<4x8xf16>
    tt.return
  }
}
