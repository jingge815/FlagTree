#ifndef TRITONPIM_ATTRDEFS
#define TRITONPIM_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/TritonPIM/IR/PIMDialect.td"

class TritonPIM_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<TritonPIM_Dialect, name, traits> {
  let attrName = "triton.pim." # attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Tensor layout encoding
//===----------------------------------------------------------------------===//

def TTPIM_TaskletTiledEncodingAttr
    : TritonPIM_Attr<"TaskletTiledEncoding", "tasklet_tiled"> {
  let mnemonic = "tasklet_tiled";

  let description = [{
    Describes how a tensor is decomposed across the PIM execution hierarchy.
    This is the PIM counterpart of `#ttg.blocked`, with the GPU's
    (thread, warp, CTA) hierarchy replaced by (tasklet, DPU).

    Three per-dimension tuples, from innermost outwards:

      - `sizePerTasklet`: how many elements along each dimension a single
        tasklet owns;
      - `taskletsPerDpu`: how the tasklets of one DPU are arranged over the
        dimensions;
      - `dpusPerDevice`: how the DPUs of the device are arranged over the
        dimensions.

    `order` lists dimensions fastest-changing first, as in `#ttg.blocked`.

    The product `sizePerTasklet[i] * taskletsPerDpu[i] * dpusPerDevice[i]`
    describes one pass over dimension `i`; when the tensor is larger the layout
    is replicated over it, again matching `#ttg.blocked` semantics.

    Note on `dpusPerDevice`: an operator kernel only ever sees the address space
    of its *own* DPU, so lowering from TTIR keeps this at all-ones. The field
    exists so that a graph-level compiler, which is what decides cross-DPU
    sharding, has somewhere to record its decision without a layout change.

    Example: a 64x32 tile over 16 tasklets of one DPU, row-major:

    ```mlir
    #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1],
                        dpusPerDevice = [1, 1], order = [1, 0]}>
    ```
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerTasklet,
    ArrayRefParameter<"unsigned">:$taskletsPerDpu,
    ArrayRefParameter<"unsigned">:$dpusPerDevice,
    ArrayRefParameter<"unsigned">:$order // fastest-changing axis first
  );

  let builders = [
    // Distribute `numTasklets` tasklets over `shape`, starting from the
    // fastest-changing dimension, giving each tasklet `sizePerTasklet`
    // elements. Any tasklets left over after every dimension is covered are
    // folded into the slowest-changing dimension so the product stays exact.
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerTasklet,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numTasklets,
                     "unsigned":$numDpus), [{
      unsigned rank = sizePerTasklet.size();
      SmallVector<unsigned, 4> taskletsPerDpu(rank, 1);
      SmallVector<unsigned, 4> dpusPerDevice(rank, 1);

      unsigned remaining = numTasklets;
      for (unsigned d = 0; d < rank && remaining > 1; ++d) {
        unsigned i = order[d];
        unsigned avail = std::max<unsigned>(
            1, (unsigned)shape[i] / std::max<unsigned>(1, sizePerTasklet[i]));
        taskletsPerDpu[i] = std::clamp<unsigned>(remaining, 1, avail);
        remaining /= taskletsPerDpu[i];
      }
      // Whatever did not fit wraps at the slowest-changing dimension, so that
      // product(taskletsPerDpu) == numTasklets always holds.
      if (remaining > 1)
        taskletsPerDpu[order[rank - 1]] *= remaining;

      // Kernels are single-DPU by construction; see the note above.
      (void)numDpus;

      return $_get(context, sizePerTasklet, taskletsPerDpu, dpusPerDevice,
                   order);
    }]>
  ];

  let extraClassDeclaration = [{
    // Number of tasklets this layout spreads a tile over.
    unsigned getTotalTaskletsPerDpu() const;
    // Number of DPUs this layout spreads a tensor over.
    unsigned getTotalDpusPerDevice() const;
    // Elements owned by a single tasklet.
    unsigned getTotalSizePerTasklet() const;
  }];

  // Hand-written, like #ttg.blocked: the generic parser cannot read an
  // `ArrayRef<unsigned>` from bracket syntax.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Memory spaces
//===----------------------------------------------------------------------===//

def TTPIM_WRAMSpace : TritonPIM_Attr<"WRAMSpace", "wram"> {
  let mnemonic = "wram";
  let description = [{
    Indicates that a memory descriptor points into WRAM: the small, fast,
    software-managed scratchpad private to a DPU. Data must be moved here
    explicitly before it can be computed on -- there is no cache that would do
    it implicitly.
  }];
}

def TTPIM_MRAMSpace : TritonPIM_Attr<"MRAMSpace", "mram"> {
  let mnemonic = "mram";
  let description = [{
    Indicates that a memory descriptor points into MRAM: the large main memory
    of a DPU. Addressable only by its own DPU; neither the host nor any other
    DPU shares this address space.
  }];
}

def TTPIM_MemSpace : AnyAttrOf<[TTPIM_WRAMSpace, TTPIM_MRAMSpace]>;

//===----------------------------------------------------------------------===//
// DMA direction
//===----------------------------------------------------------------------===//

def TTPIM_DmaMRAMToWRAM : I32EnumAttrCase<"MRAMToWRAM", 0, "mram_to_wram">;
def TTPIM_DmaWRAMToMRAM : I32EnumAttrCase<"WRAMToMRAM", 1, "wram_to_mram">;

def TTPIM_DmaDir : I32EnumAttr<"DmaDir",
  "Direction of an MRAM<->WRAM transfer", [
    TTPIM_DmaMRAMToWRAM,
    TTPIM_DmaWRAMToMRAM
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_DmaDirAttr : EnumAttr<TritonPIM_Dialect, TTPIM_DmaDir, "dma_dir">;

#endif // TRITONPIM_ATTRDEFS
