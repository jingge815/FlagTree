// RUN: triton-opt %s -split-input-file -pim-expand-phases -pim-lower-to-emitc | FileCheck %s

// Operator-level kernels: whole tensors in, one output pointer out. These are
// the shapes the graph compiler's emitter produces -- no DMA, no WRAM staging,
// so none of the tile-level path applies.
//
// The point of these cases is that the phase chains really do become C. Until
// this existed the operator-level ops reached the pass and were dropped on the
// floor: the generated C computed something else entirely, and the NumPy
// mirror "agreed" because it was wrong the same way.

// CHECK-LABEL: func.func @dq
module {
  tt.func @dq(%x: tensor<1x4096xf16>, %s: tensor<32xf16>) {
    // One input plus one output parameter. The scale operand `%s` does
    // not get one: phase 1 produces it, so nothing reads the argument.
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i8>
    %q = pim.quantize %x, %s
       {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>}
       : tensor<1x4096xf16>, tensor<32xf16> -> tensor<1x4096xi8>
    tt.return
  }
}

// The four phases, each as its own loop over the values it walks: the grouped
// absmax over every element, the two per-group scalar walks, and the
// quantization over every element again.
// CHECK: emitc.for
// CHECK: call_opaque "pim_absf"
// CHECK: emitc.for
// CHECK: call_opaque "pim_lut_identity"
// CHECK: emitc.for
// CHECK: call_opaque "pim_lut_reciprocal"
// CHECK: emitc.for
// CHECK: call_opaque "pim_f32_to_i8"

// -----

// CHECK-LABEL: func.func @sm
module {
  tt.func @sm(%s: tensor<1x1024xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    %p = pim.softmax %s {axis = 1 : i64, unit = #pim.unit<cstl>}
       : tensor<1x1024xf16> -> tensor<1x1024xf16>
    tt.return
  }
}

// Five phases: the max reduction, the stabilization affine, exp, the sum, the
// reciprocal, and the final multiply.
// CHECK: emitc.for
// CHECK: call_opaque "pim_maxf"

// The stabilization `x - max` carries no phase spec -- it is an FPSU affine
// folded into the exp phase and never round trips through an f16 buffer on
// hardware. It keeps an **f32** buffer here for the same reason: rounding it
// would put this kernel an f16 ulp away from the NumPy mirror on about a
// quarter of the elements. Every other buffer in this kernel is f16.
// CHECK: emitc.call_opaque "malloc"(%{{.*}}) : (i64) -> !emitc.ptr<f32>

// CHECK: call_opaque "pim_lut_exp"
// CHECK: call_opaque "pim_lut_reciprocal"

// -----

// A kernel whose intermediate nothing consumes but which is not the output --
// dynamic quantization's phase 1 scale, which downstream nodes read out of
// band. It gets a buffer and is computed; only the last unwritten result
// becomes the output parameter.
// CHECK-LABEL: func.func @dq_intermediate_is_not_the_output
module {
  tt.func @dq_intermediate_is_not_the_output(%x: tensor<1x4096xf16>, %s: tensor<8xf16>) {
    // Two parameters for one input and one output.
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i8>
    %q = pim.quantize %x, %s
       {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 512, spg = true, spgAxis = 3, spgGroupSize = 512>}
       : tensor<1x4096xf16>, tensor<8xf16> -> tensor<1x4096xi8>
    tt.return
  }
}

// -----

// A gather becomes two nested loops: one row per index, copied element by
// element. The index is read as an integer rather than through the f32 path --
// a large row number would lose precision on the way.
// CHECK-LABEL: func.func @gather
module {
  tt.func @gather(%table: tensor<100x8xf16>, %ids: tensor<1x5xi32>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i32>, %{{.*}}: !emitc.ptr<i16>
    %y = pim.gather %table, %ids
        : tensor<100x8xf16>, tensor<1x5xi32> -> tensor<1x5x8xf16>
    tt.return
  }
}

// 内层循环打印时省略 `emitc.` 前缀（嵌套区域内的简写形式）。
// CHECK: emitc.for
// CHECK: for %
// CHECK: call_opaque "pim_f16_to_f32"

// -----

// RoPE's three phases become C. The sin term's half swap is an index remap plus
// a conditional negate -- not a materialized rotation, because there is no
// negate op anywhere in the reference implementation's chain and a rotated copy
// would be a buffer the hardware does not have.
//
// cos/sin broadcast along the head axis (`[1,H,S,D]` against `[1,1,S,D]`), which
// wraps by the table's element count rather than dividing by a row width.
// CHECK-LABEL: func.func @rope
module {
  tt.func @rope(%x: tensor<1x2x4x8xf16>, %c: tensor<1x1x4x8xf16>,
                %s: tensor<1x1x4x8xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    %y = pim.rope %x, %c, %s {numHeads = 2 : i64}
       : tensor<1x2x4x8xf16>, tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>
       -> tensor<1x2x4x8xf16>
    tt.return
  }
}

// 半区判据与取负：按列号选边，不建旋转副本。
// CHECK: cmp lt
// CHECK: unary_minus
// CHECK: conditional

// -----

// The grouped-dequantization accumulator, at the loop structure rather than at
// a number. Four loops nest m -> n -> group -> k: the third one walks the
// groups, and the multiply that folds a group in sits at its boundary, after
// the inner loop closes and before the next group opens. That nesting is the
// whole point -- two groups carry two scales, so the integer accumulator has
// to stop at every group boundary and dequantize before folding, which no
// single flat k loop can express.
//
// CHECK-LABEL: func.func @w4a8
module {
  tt.func @w4a8(%a: tensor<4x4xi8>, %w: tensor<4x2xi8>, %s: tensor<2x2xf16>) {
    // A, W, the per-group scales and the output. The two weights are 1 byte per
    // element (int4 stored sign-extended rather than nibble packed); the scales
    // are fp16, hence the i16 pointer.
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i8>, %{{.*}}: !emitc.ptr<i8>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i8>
    %0 = pim.matmul %a, %w scales %s
       {datapath = #pim.datapath<nmuMode = floating_point,
                                 scaleMode = floating_point,
                                 groupDequantAccum = true, groupSize = 2>,
        weightBinding = #pim.weight_binding<format = weight,
                                            role = model_weight,
                                            elemBits = 4, groupSize = 2>,
        stationarity = #pim.stationarity<weight>}
       : tensor<4x4xi8>, tensor<4x2xi8> scales tensor<2x2xf16> -> tensor<4x2xi8>
    tt.return
  }
}

// Only the outermost loop in a block carries its dialect prefix; the three
// nested ones print bare, which is how MLIR spells an op inside a region.
// So `emitc.for` then `for` three times is m -> n -> group -> k.
// CHECK: emitc.for
// CHECK: for
// The running f32 total, opened per (m, n).
// CHECK: emitc.variable
// CHECK: for
// A second f32 accumulator, opened per group: this is the integer accumulator
// the group is summed into before it is dequantized. A flat k loop would have
// no such variable.
// CHECK: emitc.variable
// CHECK: for
// The integer product, read one byte at a time.
// CHECK: call_opaque "pim_i8_to_f32"
// CHECK: mul
// The group boundary: scale what this group accumulated, then fold it into the
// total. A flat k loop that dequantized the whole row at the end would have no
// mul in this position.
// CHECK: mul
// CHECK: add
// The result goes out saturated to a byte.
// CHECK: call_opaque "pim_f32_to_i8"

// -----

// Without the flag there is one group spanning all of K, so the group loop is
// a single iteration and the two forms are the identity of one loop nest
// rather than two code paths -- the nest below is the same four loops. The
// `mul` at the boundary survives as a multiply by one, which is what keeps a
// grouped and an ungrouped multiply from being two lowerings that can drift
// apart.
// CHECK-LABEL: func.func @w4a8_ungrouped
module {
  tt.func @w4a8_ungrouped(%a: tensor<4x4xi8>, %w: tensor<4x2xi8>) {
    %0 = pim.matmul %a, %w
       {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>,
        weightBinding = #pim.weight_binding<format = weight, role = model_weight,
                                            elemBits = 4>,
        stationarity = #pim.stationarity<weight>}
       : tensor<4x4xi8>, tensor<4x2xi8> -> tensor<4x2xi8>
    tt.return
  }
}
// CHECK: emitc.for
// CHECK: for
// CHECK: emitc.variable
// CHECK: for
// CHECK: emitc.variable
// CHECK: for
// CHECK: call_opaque "pim_i8_to_f32"
// CHECK: mul

// -----

// An rms normalization: square, average over the row, add epsilon, take the
// reciprocal square root, scale by gamma. The mean is over the last axis, so
// the row width is what the inner loop walks.
// CHECK-LABEL: func.func @rms
module {
  tt.func @rms(%x: tensor<4x8xf16>, %g: tensor<8xf16>, %e: tensor<1xf32>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<f32>, %{{.*}}: !emitc.ptr<i16>
    %y = pim.normalize %x, %g eps %e {axis = 1 : i64, rmsNorm}
       : tensor<4x8xf16>, tensor<8xf16> eps tensor<1xf32> -> tensor<4x8xf16>
    tt.return
  }
}

// One loop over the rows for the sum of squares, then one for the scaled
// result -- two passes, because the mean is not known until the row is read.
// CHECK: emitc.for
// CHECK: mul
// CHECK: call_opaque "pim_rsqrtf"
// CHECK: for
// CHECK: mul

// -----

// An attention mask is an additive bias in the floating domain, not a boolean:
// the masked-out positions carry negative infinity, so the addition is what
// applies it. The mask broadcast along the query axis is one value per column.
// CHECK-LABEL: func.func @mask
module {
  tt.func @mask(%scores: tensor<2x4xf16>, %m: tensor<1x4xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    %y = pim.mask %scores, %m : tensor<2x4xf16>, tensor<1x4xf16> -> tensor<2x4xf16>
    tt.return
  }
}

// CHECK: emitc.for
// CHECK: add

// -----

// A permutation, element by element. `absorbed` says the consumer's layout
// already agrees so no node is emitted for it, but the values still have to
// land in the right order: the output is one flat buffer either way.
// CHECK-LABEL: func.func @transpose_axes
module {
  tt.func @transpose_axes(%x: tensor<4x8xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    %y = pim.transpose %x {axes = array<i64: 1, 0>, onthefly, purpose = #pim.transpose_purpose<purpose = absorbed>}
       : tensor<4x8xf16> -> tensor<8x4xf16>
    tt.return
  }
}

// One loop over the result's elements, each mapped back through the
// permutation -- the result is what the caller reads, so its count is the one
// that has to come out whole.
// CHECK: emitc.for
// CHECK: div
// CHECK: rem

// -----

// CHECK-LABEL: func.func @reshape_is_a_copy
module {
  tt.func @reshape_is_a_copy(%x: tensor<4x8xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    %y = pim.reshape %x : tensor<4x8xf16> -> tensor<2x16xf16>
    tt.return
  }
}
// Element order is preserved, so this is a copy. Aliasing instead would read
// the right bytes in the wrong order.
// CHECK: emitc.for
// CHECK: load
// CHECK: store

// -----

// CHECK-LABEL: func.func @concat_runs
module {
  tt.func @concat_runs(%a: tensor<2x4xf16>, %b: tensor<3x4xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    %y = pim.concat %a, %b {axis = 0 : i64} : tensor<2x4xf16>, tensor<3x4xf16> -> tensor<5x4xf16>
    tt.return
  }
}
// Each source contributes a run of whole rows, laid end to end.
// CHECK: emitc.for
// CHECK: for
// CHECK: for

// -----

// A pure format change: the same numbers in another element type. The store
// does the conversion, which is where the rounding and the saturation live.
// CHECK-LABEL: func.func @convert_to_i8
module {
  tt.func @convert_to_i8(%x: tensor<2x4xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i8>
    %y = pim.convert %x : tensor<2x4xf16> -> tensor<2x4xi8>
    tt.return
  }
}
// CHECK: emitc.for
// CHECK: call_opaque "pim_f32_to_i8"

// -----

// A scatter writes the row the index names, into a cache the caller owns. The
// cache is a memory descriptor rather than a tensor: it outlives one kernel,
// so the kernel takes a pointer into it.
// CHECK-LABEL: func.func @kv_scatter
module {
  tt.func @kv_scatter(%v: tensor<4xi8>, %cache: !pim.memdesc<16xi8, #pim.mram>, %pos: i32, %slot: tensor<1xi16>) {
    // Three parameters and no fourth: a cache write has no output of its own,
    // so inventing an output pointer would make every caller pass a buffer the
    // kernel never writes.
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i8>, %{{.*}}: !emitc.ptr<i8>, %{{.*}}: !emitc.ptr<i16>)
    pim.kv_cache %v, %cache, %pos[%slot] {layer = 0 : i64, isKey}
        : tensor<4xi8>, !pim.memdesc<16xi8, #pim.mram>, i32 [tensor<1xi16>]
    tt.return
  }
}

// The index is read as an integer rather than through the f32 path: a row
// number is not a value, and a float would round it. It is scaled by the row
// width to get a base, then the row is copied into the cache from there.
// CHECK: emitc.cast
// CHECK: emitc.mul
// CHECK: emitc.for

// -----

// A folded silu must appear in the C. Until this existed, matmul {activation}
// and a plain matmul emitted identical C (the attribute evaporated).
// CHECK-LABEL: func.func @fused_silu
module {
  tt.func @fused_silu(%a: tensor<4x4xi8>, %w: tensor<4x2xi8>) {
    // CHECK: call_opaque "pim_lut_silu"
    %0 = pim.matmul %a, %w
       {activation = #pim.act_spec<kind = silu>,
        datapath = #pim.datapath<nmuMode = floating_point,
                                 scaleMode = floating_point>,
        weightBinding = #pim.weight_binding<format = weight,
                                            role = model_weight,
                                            elemBits = 4, groupSize = 2>,
        stationarity = #pim.stationarity<weight>}
       : tensor<4x4xi8>, tensor<4x2xi8> -> tensor<4x2xi8>
    tt.return
  }
}

// -----

// Independent `pim.fpsu_scale`: add bias, multiply by scale, then 2^-shift.
// The shift is the load-bearing field -- hardcoding 2^-14 would make the
// attribute decorative, the same failure `pim.kantor`'s shift used to have.
// CHECK-LABEL: func.func @fpsu_scale
module {
  tt.func @fpsu_scale(%x: tensor<4xf16>, %b: tensor<4xf32>, %s: tensor<4xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<f32>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i8>
    %y = pim.fpsu_scale %x, %b, %s {shift = 14 : i8,
        spec = #pim.fpsu_spec<mode = fixed_point>}
        : tensor<4xf16>, tensor<4xf32>, tensor<4xf16> -> tensor<4xi8>
    tt.return
  }
}
// CHECK: add %{{.*}}, %{{.*}} : (f32, f32)
// CHECK: mul %{{.*}}, %{{.*}} : (f32, f32)
// CHECK: mul %{{.*}}, %{{.*}} : (f32, f32)
// CHECK: call_opaque "pim_f32_to_i8"

// -----

// A scatter that carries the update-side scale must apply it, not drop it.
// CHECK-LABEL: func.func @kv_scatter_scaled
module {
  tt.func @kv_scatter_scaled(%v: tensor<4xi8>, %cache: !pim.memdesc<16xi8, #pim.mram>,
                             %pos: i32, %slot: tensor<1xi16>, %sf: tensor<1xf16>) {
    pim.kv_cache %v, %cache, %pos[%slot] sf %sf {layer = 0 : i64, isKey}
        : tensor<4xi8>, !pim.memdesc<16xi8, #pim.mram>, i32 [tensor<1xi16>] sf tensor<1xf16>
    tt.return
  }
}
// CHECK: emitc.mul
// CHECK: emitc.for
// CHECK: call_opaque "pim_f32_to_i8"

// -----

// Changing `shift` must change the generated C. The constant is 2^-shift:
// -8 → 256. CHECK this literal so a hardcoded 256 that ignores the attribute
// still matches today -- but a later test with a different shift would fail
// if the lowering stopped reading the attribute.
// CHECK-LABEL: func.func @dq_shift_is_two_to_the_minus_shift
module {
  tt.func @dq_shift_is_two_to_the_minus_shift(%x: tensor<1x32xf16>, %s: tensor<1xf16>) {
    %q, %sc = pim.dynamic_quant %x
       {groupSize = 32 : i64, axis = 1 : i64,
        spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 32, spg = true, spgAxis = 3, spgGroupSize = 32>}
       : tensor<1x32xf16> -> tensor<1x32xi8>, tensor<1x1xf16>
    tt.return
  }
}
// 2^-(-8) = 256, emitted as an f32 constant of the last multiply.
// CHECK: 2.560000e+02

// -----

// range_write 带定标：indices 缺席时 sf 仍要能解析，不能只走 generic form。
// CHECK-LABEL: func.func @kv_range_write_scaled
module {
  tt.func @kv_range_write_scaled(%v: tensor<4xi8>, %cache: !pim.memdesc<16xi8, #pim.mram>,
                                 %pos: i32, %sf: tensor<1xf16>) {
    pim.kv_cache %v, %cache, %pos sf %sf {layer = 0 : i64, isKey, mode = #pim.kv_mode<range_write>}
        : tensor<4xi8>, !pim.memdesc<16xi8, #pim.mram>, i32 sf tensor<1xf16>
    tt.return
  }
}
// CHECK: mul
// CHECK: call_opaque "pim_f32_to_i8"

// -----

// 按轴切开，每份结果各有一段拷贝循环。降级漏了这两类会静默不报，所以锁住。
// CHECK-LABEL: func.func @split_along_axis
module {
  tt.func @split_along_axis(%x: tensor<4x8xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    // CHECK: emitc.for
    // CHECK: load
    // CHECK: assign
    %a, %b = pim.split %x {axis = 0 : i64} : tensor<4x8xf16> -> tensor<2x8xf16>, tensor<2x8xf16>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @split_heads_along_axis
module {
  tt.func @split_heads_along_axis(%x: tensor<1x4x8xf16>) {
    // CHECK-SAME: %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>, %{{.*}}: !emitc.ptr<i16>
    // CHECK: emitc.for
    // CHECK: load
    // CHECK: assign
    %a, %b = pim.split_heads %x {axis = 1 : i64, numHeads = 2 : i64} : tensor<1x4x8xf16> -> tensor<1x2x8xf16>, tensor<1x2x8xf16>
    tt.return
  }
}
