#ifndef TRITONPIM_PASSES
#define TRITONPIM_PASSES

include "mlir/Pass/PassBase.td"

def TritonPIMExplicitDMA : Pass<"pim-explicit-dma", "mlir::ModuleOp"> {
  let summary = "Lower tt.load/tt.store into explicit WRAM staging plus DMA";

  let description = [{
    Replaces the implicit global accesses carried over from TTIR with the
    explicit two-level data movement a PIM device requires:

      `tt.load`  -> `pim.wram_alloc` + `pim.dma_load`  + `pim.barrier` + `pim.wram_load`
      `tt.store` -> `pim.wram_alloc` + `pim.wram_store` + `pim.barrier` + `pim.dma_store`

    `pim.wram_alloc` is hoisted out of enclosing loops so that a staging buffer
    is reserved once and reused every iteration, rather than reserved per tile.

    The pass also runs a deliberately narrow pointer analysis over each
    transfer's pointer operand, walking back through `tt.addptr` / `tt.splat` /
    `tt.broadcast` / `tt.expand_dims` / `tt.make_range`. When it can prove a
    strided access pattern it records that on the DMA op as `contiguous_dim` /
    `elem_stride` / `base_arg`; when it cannot, it records nothing. The absence
    of those attributes is meaningful -- it marks a transfer not yet proven
    expressible as a strided DMA, for a later, stronger analysis to revisit.

    Finally it sums the WRAM claimed by the allocations into the
    `pim.wram-bytes-used` module attribute, and warns if that exceeds the
    `pim.wram-bytes` budget. Note that it does not re-tile to fit the budget;
    that would mean restructuring loops and is left to a separate pass.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];
}

#endif // TRITONPIM_PASSES
