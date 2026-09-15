#ifndef TRITONPIM_OPS
#define TRITONPIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonPIM/IR/PIMDialect.td"
include "triton/Dialect/TritonPIM/IR/PIMAttrDefs.td"
include "triton/Dialect/TritonPIM/IR/PIMTypes.td"

//===----------------------------------------------------------------------===//
// Memory resources
//===----------------------------------------------------------------------===//

def TTPIM_WRAM : Resource<"::mlir::triton::pim::WRAM">;
def TTPIM_MRAM : Resource<"::mlir::triton::pim::MRAM">;

//===----------------------------------------------------------------------===//
// Execution hierarchy queries
//===----------------------------------------------------------------------===//

def TTPIM_TaskletIdOp : TTPIM_Op<"tasklet_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "index of the current tasklet within its DPU";

  let description = [{
    Returns the id of the tasklet executing this code, in `[0, num_tasklets)`
    where `num_tasklets` is the `pim.num-tasklets` module attribute. This is the
    PIM analogue of a thread id; it is how a kernel body distinguishes the
    tasklets that share a DPU.

    Example:

    ```mlir
    %tid = pim.tasklet_id : i32
    ```
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTPIM_DpuIdOp : TTPIM_Op<"dpu_id", [
    Pure,
    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let summary = "index of the current DPU within the device";

  let description = [{
    Returns the id of the DPU executing this code, in `[0, num_dpus)` where
    `num_dpus` is the `pim.num-dpus` module attribute.

    A kernel lowered from TTIR is single-DPU and has no reason to query this;
    the op exists for hand-written kernels and for a future graph-level
    compiler that maps shards onto DPUs.
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Scratchpad allocation
//===----------------------------------------------------------------------===//

def TTPIM_WRAMAllocOp : TTPIM_Op<"wram_alloc", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate a staging buffer in WRAM";

  let description = [{
    Reserves a buffer in the DPU's WRAM and returns a descriptor for it. WRAM is
    a small budget -- the `pim.wram-bytes` module attribute records how small --
    so allocations are expected to be hoisted out of loops and reused across
    iterations rather than made per-tile.

    Deallocation is implicit: a kernel's WRAM is reclaimed when the kernel ends.

    Example:

    ```mlir
    %buf = pim.wram_alloc : !pim.memdesc<64x32xf16, #pim.wram>
    ```
  }];

  let arguments = (ins OptionalAttr<I32Attr>:$alignment);
  let results = (outs TTPIM_WRAMMemDesc:$result);

  let builders = [
    OpBuilder<(ins "Type":$result),
              [{ build($_builder, $_state, result, IntegerAttr()); }]>
  ];

  let extraClassDeclaration = [{
    MemDescType getType() {
      return ::llvm::cast<MemDescType>(getResult().getType());
    }
    int32_t getAlignmentOrDefault();
  }];

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Explicit data movement
//===----------------------------------------------------------------------===//

def TTPIM_DmaLoadOp : TTPIM_Op<"dma_load", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    // `mask` and `other` are elided from the assembly format, so their types
    // have to be derivable from `ptr` for the generated parser and builders.
    TypesMatchWith<"mask type matches ptr type", "ptr", "mask",
                   "getI1SameShape($_self)",
                   "($_op.getOperands().size() <= 2) || std::equal_to<>()">,
    TypesMatchWith<"other type matches ptr type", "ptr", "other",
                   "getPointeeType($_self)",
                   "($_op.getOperands().size() <= 3) || std::equal_to<>()">]> {
  let summary = "DMA a tile from MRAM into a WRAM buffer";

  let description = [{
    Transfers the elements addressed by `ptr` from MRAM into `result`, a WRAM
    buffer. Where a GPU would let the cache hierarchy handle this implicitly,
    a PIM device requires the transfer to be spelled out -- this op is that
    transfer.

    `ptr` is a *tensor of pointers*, carried over unchanged from TTIR's
    `tt.load`. Recovering a base address and stride from it, which a DMA engine
    ultimately needs, is a separate analysis; when that analysis succeeds it
    records what it proved in these optional attributes:

      - `contiguous_dim`: the dimension along which addresses advance by
        `elem_stride` elements per index step;
      - `elem_stride`: that step, in elements;
      - `base_arg`: index of the `tt.func` argument the base address traces
        back to.

    Their *absence carries meaning*: it says the transfer has not been proven
    expressible as a strided DMA, not that it is contiguous by default. A
    backend must not assume a layout these attributes do not assert.

    `mask` and `other` have `tt.load` semantics: lanes whose mask is false are
    filled from `other` rather than transferred.

    This op does not synchronize. A tasklet that reads the buffer afterwards
    must be separated from the transfer by a `pim.barrier`.

    Example:

    ```mlir
    pim.dma_load %a_ptrs, %mask, %fill -> %buf
        {contiguous_dim = 1 : i64, elem_stride = 1 : i64}
        : tensor<64x32x!tt.ptr<f16>> -> !pim.memdesc<64x32xf16, #pim.wram>
    ```
  }];

  // Effects are declared in C++ (see getEffects), not via Arg<> here, so that
  // the MRAM read and WRAM write are attached to the right operands.
  let arguments = (ins
    TT_PtrTensor:$ptr,
    Optional<TT_BoolLike>:$mask,
    Optional<TT_Type>:$other,
    TTPIM_WRAMMemDesc:$result,
    OptionalAttr<I64Attr>:$contiguous_dim,
    OptionalAttr<I64Attr>:$elem_stride,
    OptionalAttr<I64Attr>:$base_arg
  );

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other,
                   "Value":$result),
              [{ build($_builder, $_state, ptr, mask, other, result,
                       IntegerAttr(), IntegerAttr(), IntegerAttr()); }]>
  ];

  let extraClassDeclaration = [{
    MemDescType getMemDescType() {
      return ::llvm::cast<MemDescType>(getResult().getType());
    }
    // True when the pointer analysis proved a strided access pattern.
    bool hasProvenLayout() { return getContiguousDim().has_value(); }
  }];

  let assemblyFormat = [{
    $ptr (`,` $mask^)? (`,` $other^)? `->` $result attr-dict
    `:` type($ptr) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def TTPIM_DmaStoreOp : TTPIM_Op<"dma_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"mask type matches ptr type", "ptr", "mask",
                   "getI1SameShape($_self)",
                   "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
  let summary = "DMA a tile from a WRAM buffer back to MRAM";

  let description = [{
    The reverse of `pim.dma_load`: transfers `src`, a WRAM buffer, out to the
    MRAM addresses given by `ptr`. Lanes whose `mask` is false are not written.

    The optional `contiguous_dim` / `elem_stride` / `base_arg` attributes mean
    exactly what they mean on `pim.dma_load`, including that their absence is
    an assertion of ignorance rather than a default.

    Writes to `src` must be separated from this op by a `pim.barrier`.

    Example:

    ```mlir
    pim.dma_store %buf -> %c_ptrs, %mask
        : !pim.memdesc<64x128xf16, #pim.wram> -> tensor<64x128x!tt.ptr<f16>>
    ```
  }];

  let arguments = (ins
    TTPIM_WRAMMemDesc:$src,
    TT_PtrTensor:$ptr,
    Optional<TT_BoolLike>:$mask,
    OptionalAttr<I64Attr>:$contiguous_dim,
    OptionalAttr<I64Attr>:$elem_stride,
    OptionalAttr<I64Attr>:$base_arg
  );

  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$ptr, "Value":$mask),
              [{ build($_builder, $_state, src, ptr, mask,
                       IntegerAttr(), IntegerAttr(), IntegerAttr()); }]>
  ];

  let extraClassDeclaration = [{
    MemDescType getMemDescType() {
      return ::llvm::cast<MemDescType>(getSrc().getType());
    }
    bool hasProvenLayout() { return getContiguousDim().has_value(); }
  }];

  let assemblyFormat = [{
    $src `->` $ptr (`,` $mask^)? attr-dict
    `:` qualified(type($src)) `->` type($ptr)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// WRAM access
//===----------------------------------------------------------------------===//

def TTPIM_WRAMLoadOp : TTPIM_Op<"wram_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "read a WRAM buffer into a tasklet-tiled tensor";

  let description = [{
    Materializes the contents of a WRAM buffer as a tensor value, so that the
    arithmetic ops carried over from TTIR can consume it. The result's
    `#pim.tasklet_tiled` encoding says which tasklet owns which element.

    Example:

    ```mlir
    %a = pim.wram_load %buf
        : !pim.memdesc<64x32xf16, #pim.wram> -> tensor<64x32xf16, #tiled>
    ```
  }];

  let arguments = (ins TTPIM_WRAMMemDesc:$src);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_WRAMStoreOp : TTPIM_Op<"wram_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "write a tasklet-tiled tensor into a WRAM buffer";

  let description = [{
    The reverse of `pim.wram_load`: stages a computed tensor into WRAM so it can
    subsequently be DMA'd out to MRAM.

    Example:

    ```mlir
    pim.wram_store %c, %buf
        : tensor<64x128xf16, #tiled> -> !pim.memdesc<64x128xf16, #pim.wram>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    TTPIM_WRAMMemDesc:$dst
  );

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `->` qualified(type($dst))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def TTPIM_BarrierOp : TTPIM_Op<"barrier"> {
  let summary = "barrier across the tasklets of a DPU";

  let description = [{
    Blocks until every tasklet of the DPU has arrived. This is the only
    cross-tasklet synchronization primitive PIM offers: there is no warp-level
    shuffle and no atomic, so a reduction that a GPU kernel would express with
    a shuffle becomes a barrier plus explicit passes over WRAM.

    Placed between a DMA and the accesses that depend on it, and between writes
    to a buffer and the DMA that moves it out.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Layout conversion
//===----------------------------------------------------------------------===//

def TTPIM_ConvertLayoutOp : TTPIM_Op<"convert_layout", [
    Pure,
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType]> {
  let summary = "convert between two tasklet-tiled layouts";

  let description = [{
    Redistributes a tensor from one `#pim.tasklet_tiled` layout to another,
    changing which tasklet owns which element while leaving the values alone.

    Analogous to `ttg.convert_layout`. Beyond appearing in hand-written IR, it
    is what the TTIR->PIM type converter inserts wherever a value's layout does
    not match what its consumer requires; partial conversion depends on having
    such an op available.
  }];

  let arguments = (ins TT_Tensor:$src);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator-level primitives
//===----------------------------------------------------------------------===//
//
// Everything above this point is tile-level: it describes how one DPU stages
// and moves bytes. The ops below sit one level up and name a whole quantized
// operator -- the granularity the downstream graph format and the fixed-function
// units actually work in.
//
// Two properties are shared by all of them and are the reason they are ops
// rather than attributes on `tt.dot`:
//
//   - the scale / zero-point / bias tensors a quantized operator reads are
//     *operands*, so they stay ordinary SSA values that dataflow passes can see;
//   - the functional unit and datapath configuration are attributes, because
//     they describe the machine rather than the data.
//
// `unit` records which functional unit runs the operator. It is optional; when
// absent, assignment has not been made yet, which is a different statement from
// "runs on the default unit".

// Common to every operator-level op: the unit assignment and the fixed-function
// datapath configuration.
class TTPIM_OperatorOp<string mnemonic, list<Trait> traits = []>
    : TTPIM_Op<mnemonic, traits>;

def TTPIM_QuantizeOp : TTPIM_OperatorOp<"quantize"> {
  let summary = "convert a float tensor to fixed point";

  let description = [{
    Quantizes `src` using `scale` and, when the format is asymmetric,
    `zeroPoint`. `spec` says how those parameters are laid out over the tensor
    -- one for the whole tensor, one per channel, or one per group.

    A separate op rather than an attribute on the producer because dynamic
    quantization computes its scale at run time from the tensor itself, so the
    scale is a real value with a real definition, not a compile-time constant.
    `dynamic` marks that case.

    Example:

    ```mlir
    %q = pim.quantize %x, %scale, %zp
        {spec = #pim.quant_spec<granularity = per_channel, axis = 1>}
        : tensor<128x256xf32>, tensor<256xf32>, tensor<256xi8> -> tensor<128x256xi8>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    TT_Tensor:$scale,
    Optional<TT_Tensor>:$zeroPoint,
    TTPIM_QuantSpecAttr:$spec,
    UnitAttr:$dynamic,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `,` $scale (`,` $zeroPoint^)? attr-dict
    `:` type($src) `,` type($scale) (`,` type($zeroPoint)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_DequantizeOp : TTPIM_OperatorOp<"dequantize"> {
  let summary = "convert a fixed-point tensor back to float";

  let description = [{
    The inverse of `pim.quantize`: scales `src` back into floating point.
    `spec` describes the same parameter layout, and must match the spec the
    values were quantized with.

    Example:

    ```mlir
    %f = pim.dequantize %q, %scale
        {spec = #pim.quant_spec<granularity = per_tensor>}
        : tensor<128x256xi32>, tensor<1xf32> -> tensor<128x256xf32>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    TT_Tensor:$scale,
    Optional<TT_Tensor>:$zeroPoint,
    TTPIM_QuantSpecAttr:$spec,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `,` $scale (`,` $zeroPoint^)? attr-dict
    `:` type($src) `,` type($scale) (`,` type($zeroPoint)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_MatmulOp : TTPIM_OperatorOp<"matmul"> {
  let summary = "quantized matrix multiply on the matrix unit";

  let description = [{
    Computes `a * b`, optionally adding `bias`, and pushes the accumulator
    through the datapath described by `datapath`.

    This is not a replacement for `tt.dot`. `tt.dot` is the tile-level multiply
    a kernel body performs on values already staged in WRAM; this op is the
    whole operator, with its quantization parameters attached, at the
    granularity the fixed-function unit is configured in. A kernel lowered from
    TTIR keeps using `tt.dot`; this op is what a graph-level operator becomes.

    `transposeB` marks that `b` is supplied transposed, which the matrix unit
    handles natively rather than needing a separate transpose. `bIsActivation`
    marks that `b` is another operator's output rather than a weight -- the two
    are staged differently, and the distinction is not recoverable from types.

    Example:

    ```mlir
    %c = pim.matmul %a, %w, %bias
        {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                  kantorMode = scalar>,
         unit = #pim.unit<nmu>}
        : tensor<128x512xi8>, tensor<512x256xi8>, tensor<256xi32> -> tensor<128x256xi8>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$a,
    TT_Tensor:$b,
    Optional<TT_Tensor>:$bias,
    TTPIM_DatapathAttr:$datapath,
    OptionalAttr<TTPIM_ActSpecAttr>:$activation,
    OptionalAttr<TTPIM_PoolSpecAttr>:$fusedPool,
    UnitAttr:$transposeB,
    UnitAttr:$bIsActivation,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $a `,` $b (`,` $bias^)? attr-dict
    `:` type($a) `,` type($b) (`,` type($bias)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_ConvOp : TTPIM_OperatorOp<"conv"> {
  let summary = "quantized convolution on the matrix unit";

  let description = [{
    Convolves `input` with `weight`, optionally adding `bias`, over the geometry
    in `window`, then applies `datapath` to the accumulator.

    Layout is NCHW for `input` and OIHW for `weight`, matching the downstream
    graph format so the hand-off needs no transposition.

    Example:

    ```mlir
    %y = pim.conv %x, %w, %b
        {window = #pim.window<kernel = [3, 3], strides = [1, 1],
                              pads = [1, 1, 1, 1], dilations = [1, 1], group = 1>,
         datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                  kantorMode = off>,
         unit = #pim.unit<nmu>}
        : tensor<1x64x56x56xi8>, tensor<64x64x3x3xi8>, tensor<64xi32>
        -> tensor<1x64x56x56xi8>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$input,
    TT_Tensor:$weight,
    Optional<TT_Tensor>:$bias,
    TTPIM_WindowAttr:$window,
    TTPIM_DatapathAttr:$datapath,
    OptionalAttr<TTPIM_ActSpecAttr>:$activation,
    OptionalAttr<TTPIM_PoolSpecAttr>:$fusedPool,
    UnitAttr:$transposed,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
    `:` type($input) `,` type($weight) (`,` type($bias)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_LutOp : TTPIM_OperatorOp<"lut"> {
  let summary = "activation evaluated through a lookup table";

  let description = [{
    Applies an activation to `src` by table lookup. `table`, when present, is
    the table contents as an operand; when absent, the table is to be generated
    from `kind` by a later stage.

    The target has no transcendental arithmetic unit -- `silu` and `sigmoid` are
    the same instruction over different table bytes -- so `kind` is what carries
    the mathematical meaning. A reader cannot recover `silu` from the table
    bytes, and neither can a cost model, which is why the kind is kept even once
    a table exists.

    `mode` selects regular or interleaved (even/odd) addressing, which is how
    two operators share one physical table.

    Example:

    ```mlir
    %y = pim.lut %x {kind = #pim.activation<silu>, mode = #pim.lut_mode<regular>,
                     unit = #pim.unit<cstl>}
        : tensor<128x256xi16> -> tensor<128x256xi8>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    Optional<TT_Tensor>:$table,
    TTPIM_ActivationKindAttr:$kind,
    DefaultValuedAttr<TTPIM_LutModeAttr, "LutMode::Regular">:$mode,
    OptionalAttr<F64Attr>:$alpha,
    OptionalAttr<F64Attr>:$clipMin,
    OptionalAttr<F64Attr>:$clipMax,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src (`,` $table^)? attr-dict
    `:` type($src) (`,` type($table)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_EltwiseOp : TTPIM_OperatorOp<"eltwise"> {
  let summary = "binary elementwise operation with per-input scales";

  let description = [{
    Combines `lhs` and `rhs` under `kind`. Unlike `arith.addi`, the two inputs
    may carry *different* quantization scales, so the op needs the datapath
    configuration to know how to align them before combining -- which is why
    this is not simply an `arith` op on quantized tensors.

    Each operand is rescaled by its *own* datapath before they meet: `datapath`
    covers `lhs`, and `rhsDatapath` covers `rhs`. When `rhsDatapath` is absent
    both operands share `datapath`, which is the common case of two inputs that
    were quantized alike.

    `rhs` may be a smaller tensor broadcast against `lhs`, as for a bias or a
    per-channel factor.

    Example:

    ```mlir
    %z = pim.eltwise %x, %y
        {kind = #pim.eltwise<add>,
         datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                  scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1>>,
         rhsDatapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                                     scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1>>,
         unit = #pim.unit<vpu>}
        : tensor<128x256xi8>, tensor<128x256xi8> -> tensor<128x256xi8>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$lhs,
    TT_Tensor:$rhs,
    TTPIM_EltwiseKindAttr:$kind,
    TTPIM_DatapathAttr:$datapath,
    OptionalAttr<TTPIM_DatapathAttr>:$rhsDatapath,
    OptionalAttr<TTPIM_ActSpecAttr>:$activation,
    OptionalAttr<TTPIM_PoolSpecAttr>:$fusedPool,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_PoolOp : TTPIM_OperatorOp<"pool"> {
  let summary = "pooling reduction over a sliding window";

  let description = [{
    Reduces `input` over the window in `window` under `kind`. For
    `global_average` the window covers the whole spatial extent and `window` is
    ignored.

    Example:

    ```mlir
    %y = pim.pool %x
        {kind = #pim.pool<max>,
         window = #pim.window<kernel = [3, 3], strides = [2, 2],
                              pads = [1, 1, 1, 1], dilations = [1, 1], group = 1>,
         unit = #pim.unit<vpu>}
        : tensor<1x64x112x112xi8> -> tensor<1x64x56x56xi8>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$input,
    TTPIM_PoolKindAttr:$kind,
    OptionalAttr<TTPIM_WindowAttr>:$window,
    OptionalAttr<TTPIM_NumericModeAttr>:$mode,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def TTPIM_ReduceAxisOp : TTPIM_OperatorOp<"reduce_axis"> {
  let summary = "reduction along one axis on the vector unit";

  let description = [{
    Reduces `src` along `axis` under `kind`, keeping the rank (the reduced axis
    becomes 1). This is the multi-pass reduction the vector unit performs; it is
    distinct from `tt.reduce`, whose region form assumes cross-lane primitives
    the target does not have.

    Softmax and normalization are built from this plus `pim.lut` and
    `pim.eltwise`, rather than being single ops, because that is how they map
    onto the units.

    Example:

    ```mlir
    %m = pim.reduce_axis %x {kind = #pim.eltwise<max>, axis = 1 : i64,
                             unit = #pim.unit<vpu>}
        : tensor<128x256xf32> -> tensor<128x1xf32>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    TTPIM_EltwiseKindAttr:$kind,
    I64Attr:$axis,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def TTPIM_RopeOp : TTPIM_OperatorOp<"rope"> {
  let summary = "rotary position embedding on the activation unit";

  let description = [{
    Applies rotary position embedding to `src` using the precomputed `cos` and
    `sin` tables.

    This is one op rather than a transpose/split/multiply/add sequence because
    the target implements it as a single fused activation-unit operation, and
    the downstream graph format has a matching single node for it. Splitting it
    would mean re-fusing it later from a pattern match.

    `numHeads` is needed because the rotation pairs elements within a head, so
    the head boundary is part of the operation's meaning rather than just a
    shape detail. `transposeInput` marks the variant that begins with a
    transpose, which the unit folds in.

    Example:

    ```mlir
    %y = pim.rope %x, %cos, %sin {numHeads = 32 : i64, unit = #pim.unit<cstl>}
        : tensor<1x32x128x128xf16>, tensor<128x128xf16>, tensor<128x128xf16>
        -> tensor<1x32x128x128xf16>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    TT_Tensor:$cos,
    TT_Tensor:$sin,
    I64Attr:$numHeads,
    UnitAttr:$transposeInput,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `,` $cos `,` $sin attr-dict
    `:` type($src) `,` type($cos) `,` type($sin) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_NormalizeOp : TTPIM_OperatorOp<"normalize", [
    AttrSizedOperandSegments]> {
  let summary = "layer or RMS normalization on the vector unit";

  let description = [{
    Normalizes `src` along `axis`, then optionally scales by `weight` and shifts
    by `bias`.

    `rmsNorm` selects RMS normalization -- divide by the root mean square, with
    no mean subtraction -- which is what the transformer models this compiler
    targets use. Plain layer normalization is the default.

    Kept as one op rather than a reduce/rsqrt/multiply sequence because the
    downstream graph format has a single node for it, and because the
    accumulation needs a wider intermediate format than the surrounding
    quantized tensors; splitting it would lose that.

    Example:

    ```mlir
    %y = pim.normalize %x, %w {axis = -1 : i64, epsilon = 1.0e-5 : f64,
                               rmsNorm, unit = #pim.unit<vpu>}
        : tensor<128x4096xf16>, tensor<4096xf16> -> tensor<128x4096xf16>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    Optional<TT_Tensor>:$weight,
    Optional<TT_Tensor>:$bias,
    I64Attr:$axis,
    DefaultValuedAttr<F64Attr, "1.0e-5">:$epsilon,
    UnitAttr:$rmsNorm,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src (`,` $weight^)? (`,` $bias^)? attr-dict
    `:` type($src) (`,` type($weight)^)? (`,` type($bias)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_SoftmaxOp : TTPIM_OperatorOp<"softmax"> {
  let summary = "softmax along one axis";

  let description = [{
    Computes softmax over `axis` of `src`.

    One op rather than max/subtract/exp/sum/divide because the target runs it as
    a staged sequence on the activation unit with its own internal precision,
    and the downstream graph format has a single node for it. The staging is a
    property of the unit, not something the graph compiler should be choosing.

    Example:

    ```mlir
    %p = pim.softmax %s {axis = -1 : i64, unit = #pim.unit<cstl>}
        : tensor<32x128x128xf16> -> tensor<32x128x128xf16>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    I64Attr:$axis,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// On-chip staging
//===----------------------------------------------------------------------===//
//
// `pim.wram_alloc` and the DMA pair above describe a two-level hierarchy
// (MRAM <-> WRAM). The target has a third level: each functional unit has a
// private L1, and the units of one NPU share an L2 that sits between them and
// main memory. An operand therefore travels MRAM -> L2 -> L1, and a result the
// other way, with the L2 hop being how two units hand data to each other.

def TTPIM_BufferAllocOp : TTPIM_Op<"buffer_alloc", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate a buffer in an on-chip memory";

  let description = [{
    Reserves a buffer in L1 or L2 and returns a descriptor for it. The
    generalization of `pim.wram_alloc`, which stays as the name for the
    WRAM-specific case; this op is what the L1/L2 levels use.

    `unit` names the owning functional unit for an L1 allocation, since each unit
    has its own. It is meaningless for L2, which is shared, and must be omitted
    there.

    As with WRAM, deallocation is implicit at the end of the kernel.

    Example:

    ```mlir
    %stage = pim.buffer_alloc : !pim.memdesc<128x256xi8, #pim.l2>
    %operand = pim.buffer_alloc {unit = #pim.unit<nmu>}
        : !pim.memdesc<64x256xi8, #pim.l1>
    ```
  }];

  let arguments = (ins
    OptionalAttr<I32Attr>:$alignment,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TTPIM_OnChipMemDesc:$result);

  let extraClassDeclaration = [{
    MemDescType getType() {
      return ::llvm::cast<MemDescType>(getResult().getType());
    }
  }];

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

def TTPIM_BufferCopyOp : TTPIM_Op<"buffer_copy", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy between two on-chip buffers";

  let description = [{
    Moves the contents of `src` into `dst`, both on-chip. This is the L2<->L1
    hop, and the L2 staging step by which one unit's output becomes another
    unit's input.

    Distinct from `pim.dma_load` / `pim.dma_store`, which cross the MRAM
    boundary and are addressed by a tensor of pointers. This transfer is
    buffer-to-buffer within the chip and needs no address computation.

    Both buffers must be on-chip and must not be the same memory space -- a copy
    within one level would be a no-op the allocator should have avoided.

    Does not synchronize; a reader of `dst` must be separated from this op by a
    `pim.barrier`.

    Example:

    ```mlir
    pim.buffer_copy %l2_stage -> %l1_operand
        : !pim.memdesc<64x256xi8, #pim.l2> -> !pim.memdesc<64x256xi8, #pim.l1>
    ```
  }];

  let arguments = (ins
    TTPIM_OnChipMemDesc:$src,
    TTPIM_OnChipMemDesc:$dst
  );

  let assemblyFormat = [{
    $src `->` $dst attr-dict
    `:` qualified(type($src)) `->` qualified(type($dst))
  }];
  let hasVerifier = 1;
}

def TTPIM_DecompressWeightOp : TTPIM_Op<"decompress_weight", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "expand a compressed weight buffer on the way in";

  let description = [{
    Reads the compressed weight block in `src` and writes the expanded weights
    into `dst`.

    Weights are stored compressed in main memory and expanded by dedicated
    hardware as they are pulled on-chip, so the compressed and expanded sizes
    differ and only the expanded one matches the operator's logical shape. That
    makes this a real operation with two distinct buffer types rather than a
    layout annotation.

    `ratio`, when known, is the expansion factor, for the cost model's benefit;
    its absence means the factor is data-dependent, not that it is 1.

    Example:

    ```mlir
    pim.decompress_weight %packed -> %weights {ratio = 2.0 : f64}
        : !pim.memdesc<131072xi8, #pim.mram> -> !pim.memdesc<512x512xi8, #pim.l2>
    ```
  }];

  let arguments = (ins
    TTPIM_MemDescType:$src,
    TTPIM_OnChipMemDesc:$dst,
    OptionalAttr<F64Attr>:$ratio
  );

  let assemblyFormat = [{
    $src `->` $dst attr-dict
    `:` qualified(type($src)) `->` qualified(type($dst))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Parameters
//===----------------------------------------------------------------------===//

def TTPIM_ParamOp : TTPIM_Op<"param", [Pure]> {
  let summary = "reference a named constant parameter";

  let description = [{
    Yields the parameter called `name` -- a weight, a bias, a quantization scale,
    or an activation table.

    Operands of the operator-level ops are otherwise anonymous SSA values, but
    every parameter has to be matched with the buffer file that holds its bytes.
    Naming it here is what lets the two sides agree: the graph compiler writes
    the buffer under this name, and the lowering to the target graph format
    refers to it by the same name. Nothing else in the IR carries that link.

    Pure and with no operands: the value is a compile-time constant, so it can be
    freely duplicated or hoisted.

    Example:

    ```mlir
    %w = pim.param {name = "layers.0.self_attn.q_proj.weight"} : tensor<4096x4096xi8>
    ```
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape manipulation
//===----------------------------------------------------------------------===//
//
// The target graph format has its own nodes for these, and attention needs them
// between the matmuls. Triton's `tt.trans` / `tt.split` / `tt.cat` cover some of
// the same ground, but they carry tile-level layout encodings and assume a
// kernel body; these are the operator-level counterparts that survive to the
// graph hand-off.

def TTPIM_TransposeOp : TTPIM_Op<"transpose", [Pure]> {
  let summary = "permute a tensor's dimensions";

  let description = [{
    Permutes `src` according to `axes`, which must be a permutation of
    `[0, rank)`.

    Example:

    ```mlir
    %t = pim.transpose %x {axes = [0, 2, 1, 3]}
        : tensor<1x128x32x128xf16> -> tensor<1x32x128x128xf16>
    ```
  }];

  let arguments = (ins TT_Tensor:$src, DenseI64ArrayAttr:$axes);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def TTPIM_ReshapeOp : TTPIM_Op<"reshape", [Pure]> {
  let summary = "reinterpret a tensor with a new shape";

  let description = [{
    Reinterprets `src` with the result's shape, keeping the element order. The
    element counts must agree, so this moves no data.

    Example:

    ```mlir
    %r = pim.reshape %x : tensor<1x128x4096xf16> -> tensor<128x4096xf16>
    ```
  }];

  let arguments = (ins TT_Tensor:$src);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasVerifier = 1;
}

def TTPIM_SplitOp : TTPIM_Op<"split", [Pure]> {
  let summary = "split a tensor along one axis";

  let description = [{
    Splits `src` into several tensors along `axis`. The result types give the
    sizes, which must sum to the source's extent along that axis.

    Used for the fused qkv projection, whose single matmul output is split three
    ways.

    Example:

    ```mlir
    %q, %k, %v = pim.split %qkv {axis = 1 : i64}
        : tensor<128x12288xf16> -> tensor<128x4096xf16>, tensor<128x4096xf16>, tensor<128x4096xf16>
    ```
  }];

  let arguments = (ins TT_Tensor:$src, I64Attr:$axis);
  let results = (outs Variadic<TT_Tensor>:$results);

  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($results)
  }];
  let hasVerifier = 1;
}

def TTPIM_ConcatOp : TTPIM_Op<"concat", [Pure]> {
  let summary = "concatenate tensors along one axis";

  let description = [{
    Joins `srcs` along `axis`. Every input must agree on all other dimensions.

    This is how the KV cache grows during decoding: the new key or value is
    concatenated onto the cached ones.

    Example:

    ```mlir
    %kv = pim.concat %cached, %new {axis = 2 : i64}
        : tensor<1x32x127x128xf16>, tensor<1x32x1x128xf16> -> tensor<1x32x128x128xf16>
    ```
  }];

  let arguments = (ins Variadic<TT_Tensor>:$srcs, I64Attr:$axis);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $srcs attr-dict `:` type($srcs) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Attention and cache
//===----------------------------------------------------------------------===//
//
// The target's graph format models these as nodes of their own rather than as
// arithmetic, so they need primitives here too. Verified against a real llama2
// W4A8 decode block: `Mask` appears 32 times (one per head) and `KV_Cache_DMA`
// twice (the key and value writebacks).

def TTPIM_MaskOp : TTPIM_Op<"mask"> {
  let summary = "apply an additive attention mask";

  let description = [{
    Adds `mask` to `scores`. On the target this is a node of its own rather than
    an `arith.addf`, because the mask is applied inside the attention datapath
    with its own operand staging.

    `transposeInput` marks the variant that begins with a transpose, which the
    unit folds in -- the same convention `pim.rope` uses.

    Example:

    ```mlir
    %masked = pim.mask %scores, %m {unit = #pim.unit<vpu>}
        : tensor<32x1x128xf16>, tensor<1x1x128xf16> -> tensor<32x1x128xf16>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$scores,
    TT_Tensor:$mask,
    UnitAttr:$transposeInput,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $scores `,` $mask attr-dict
    `:` type($scores) `,` type($mask) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTPIM_KvCacheOp : TTPIM_Op<"kv_cache", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "read or write the KV cache through the DMA engine";

  let description = [{
    Moves `value` into the cache buffer `cache` at position `pos`, or reads the
    accumulated cache back out.

    The KV cache is the one piece of state that outlives a single decode step, so
    it is a DMA node in the target's graph rather than an ordinary tensor: the
    cache lives in main memory and the engine copies one step's worth in or out.

    `layer` and `isKey` identify which of the model's caches this touches --
    two per layer, and they are not interchangeable.

    Example:

    ```mlir
    pim.kv_cache write %k, %cache, %pos {layer = 0 : i64, isKey}
        : tensor<1x8x128xf16>, !pim.memdesc<2048x8x128xf16, #pim.mram>, i32
    ```
  }];

  let arguments = (ins
    TT_Tensor:$value,
    TTPIM_MemDescType:$cache,
    I32:$pos,
    I64Attr:$layer,
    UnitAttr:$isKey,
    UnitAttr:$isRead,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );

  let assemblyFormat = [{
    (`read` $isRead^)? $value `,` $cache `,` $pos attr-dict
    `:` type($value) `,` qualified(type($cache)) `,` type($pos)
  }];
  let hasVerifier = 1;
}

def TTPIM_SplitHeadsOp : TTPIM_Op<"split_heads", [Pure]> {
  let summary = "split a packed projection into per-head tensors";

  let description = [{
    Splits `src` along `axis` into `numHeads` equal pieces, one per attention
    head.

    Distinct from `pim.split`, whose result sizes are given individually: here
    every piece is the same size and the count is the head count, which is how
    the target's graph format writes it (one `Split` node with `num_heads`,
    feeding a fan-out of per-head nodes).

    Example:

    ```mlir
    %heads:32 = pim.split_heads %qkv {axis = 1 : i64, numHeads = 32 : i64}
        : tensor<1x4096xf16> -> tensor<1x128xf16>, ...
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    I64Attr:$axis,
    I64Attr:$numHeads,
    OptionalAttr<TTPIM_FunctionalUnitAttr>:$unit
  );
  let results = (outs Variadic<TT_Tensor>:$results);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($results)";
  let hasVerifier = 1;
}

#endif // TRITONPIM_OPS
