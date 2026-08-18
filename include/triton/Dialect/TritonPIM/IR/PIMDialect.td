#ifndef TRITONPIM_DIALECT
#define TRITONPIM_DIALECT

include "mlir/IR/OpBase.td"

def TritonPIM_Dialect : Dialect {
  let name = "pim";

  let cppNamespace = "::mlir::triton::pim";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton PIM (processing-in-memory) dialect.

    This dialect sits directly below TTIR, in parallel with TTGIR. Where TTGIR
    models a GPU (warps, shared memory, unified address space), TritonPIM models
    a near-memory accelerator in the style of UPMEM:

      - a device is a flat array of DPUs, each with a *private* address space;
      - each DPU runs several `tasklet`s over a small, explicitly managed
        scratchpad (WRAM) backed by a large main memory (MRAM);
      - there is no cache: every MRAM<->WRAM transfer is an explicit DMA;
      - there are no warp-level primitives, so cross-tasklet coordination is a
        barrier rather than a shuffle.

    The mathematical IR is deliberately left untouched: `tt.dot`, `arith` and
    `scf` ops survive lowering unchanged. What PIM adds is a tensor layout
    encoding describing the tasklet decomposition (`#pim.tasklet_tiled`), a
    memory descriptor for scratchpad buffers (`!pim.memdesc`), and explicit
    data-movement and synchronization primitives.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

class TTPIM_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonPIM_Dialect, mnemonic, traits>;

#endif // TRITONPIM_DIALECT
