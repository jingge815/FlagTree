#ifndef TRITONPIM_TYPES
#define TRITONPIM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "triton/Dialect/TritonPIM/IR/PIMDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

class TTPIM_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonPIM_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def TTPIM_MemDescType : TTPIM_TypeDef<"MemDesc", "memdesc", [ShapedTypeInterface]> {
  let summary = "descriptor of a buffer in a DPU-local memory";

  let description = [{
    Describes a buffer resident in one of a DPU's memories, identified by
    `memorySpace` (`#pim.wram` or `#pim.mram`).

    This is a deliberately trimmed-down analogue of `!ttg.memdesc`. It carries
    no layout encoding, because a scratchpad buffer on a PIM device is just a
    contiguous span of bytes -- there is no swizzling to describe. It also
    carries no separate `allocShape`, because PIM buffers are not sub-viewed:
    a tile is staged in and out whole.

    Example:

    ```mlir
    !pim.memdesc<64x32xf16, #pim.wram>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Attribute":$memorySpace,
    "bool":$mutableMemory
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "llvm::ArrayRef<int64_t>":$shape,
        "Type":$elementType,
        "Attribute":$memorySpace
    ), [{
      return $_get(elementType.getContext(), shape, elementType, memorySpace,
                   /*mutableMemory=*/true);
    }]>,
    TypeBuilderWithInferredContext<(ins
        "llvm::ArrayRef<int64_t>":$shape,
        "Type":$elementType,
        "Attribute":$memorySpace,
        "bool":$mutableMemory
    ), [{
      return $_get(elementType.getContext(), shape, elementType, memorySpace,
                   mutableMemory);
    }]>
  ];

  let extraClassDeclaration = [{
    bool hasRank() const { return true; }

    MemDescType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                          Type elementType) const {
      return MemDescType::get(shape.value_or(getShape()), elementType,
                              getMemorySpace(), getMutableMemory());
    }

    bool isWRAM() const;
    bool isMRAM() const;
    bool isL1() const;
    bool isL2() const;

    // Bytes this buffer occupies. Returns std::nullopt when the element type
    // has no fixed size or the shape is dynamic.
    std::optional<int64_t> getSizeInBytes() const;
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// A memdesc known to live in WRAM.
def TTPIM_WRAMMemDesc : Type<
  And<[TTPIM_MemDescType.predicate,
       CPred<"::llvm::cast<::mlir::triton::pim::MemDescType>($_self).isWRAM()">]>,
  "a memory descriptor in WRAM", "::mlir::triton::pim::MemDescType">;

// A memdesc known to live in L2, the NPU-shared buffer.
def TTPIM_L2MemDesc : Type<
  And<[TTPIM_MemDescType.predicate,
       CPred<"::llvm::cast<::mlir::triton::pim::MemDescType>($_self).isL2()">]>,
  "a memory descriptor in L2", "::mlir::triton::pim::MemDescType">;

// A memdesc in any of the on-chip staging spaces, i.e. anywhere but MRAM.
def TTPIM_OnChipMemDesc : Type<
  And<[TTPIM_MemDescType.predicate,
       CPred<"!::llvm::cast<::mlir::triton::pim::MemDescType>($_self).isMRAM()">]>,
  "a memory descriptor in an on-chip memory", "::mlir::triton::pim::MemDescType">;

#endif // TRITONPIM_TYPES
