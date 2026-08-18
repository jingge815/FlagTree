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

#endif // TRITONPIM_OPS
