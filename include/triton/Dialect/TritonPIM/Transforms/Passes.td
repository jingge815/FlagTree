#ifndef TRITONPIM_PASSES
#define TRITONPIM_PASSES

include "mlir/Pass/PassBase.td"

def TritonPIMLowerSingleTasklet : Pass<"pim-lower-single-tasklet", "mlir::ModuleOp"> {
  let summary = "Lower a single-DPU, single-tasklet PIM kernel to EmitC";

  let description = [{
    Bridges the flagos-pim-compiler graph compiler to a numpy-executable
    artifact: lowers the pim-explicit-dma output (`pim.wram_alloc` /
    `pim.dma_load` / `pim.barrier` / `pim.wram_load` / `tt.dot` /
    `pim.wram_store` / `pim.dma_store`) into `emitc.*` ops that
    `-convert-func-to-emitc` and `mlir-translate --mlir-to-cpp` can turn into
    plain C.

    Scoped deliberately narrowly: `pim.num-tasklets` must be 1 and every shape
    involved must be static. Under that restriction the two-level WRAM staging
    a real PIM device needs has no numeric effect on a single tasklet reading
    directly-addressable memory, so this pass elides it rather than modeling a
    WRAM buffer that nothing downstream needs: `pim.wram_alloc` disappears,
    `pim.dma_load`/`pim.wram_load` collapse to the original MRAM pointer
    (recovered from `base_arg`), `pim.barrier` is dropped, and the mirror image
    happens for stores. `pim.tasklet_id`/`pim.dpu_id` become the constant 0.

    `tt.dot` has no lowering anywhere in Triton or FlagTree below the tensor
    level -- every existing path either keeps its layout encoding for a tensor
    core or (on PIM) leaves it untouched. This pass is what actually expands
    it, into a triple-nested `emitc.for` reduction over `emitc.subscript`/
    `emitc.load`/`emitc.assign`.

    Function arguments are rewritten from `!tt.ptr<T>` to `!emitc.ptr<T>`
    one-for-one -- deliberately not a single MRAM base plus per-argument byte
    offsets, so the emitted C carries no address arithmetic; the caller
    (`opcompiler_bridge/driver.py`) is expected to pass each tensor argument's
    already-offset pointer directly (e.g. `ctypes.data_as`). `tt.func` /
    `tt.return` are rewritten to `func.func`/`func.return` in the same pass,
    since `-convert-func-to-emitc` only recognizes the latter.

    A kernel outside this pass's scope (multiple tasklets, or a DMA transfer
    `pim-explicit-dma` could not prove strided) fails the pass rather than
    silently emitting something wrong.
  }];

  let dependentDialects = [
    "mlir::emitc::EmitCDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];
}

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
