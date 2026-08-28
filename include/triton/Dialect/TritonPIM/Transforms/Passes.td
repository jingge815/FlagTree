#ifndef TRITONPIM_PASSES
#define TRITONPIM_PASSES

include "mlir/Pass/PassBase.td"

def TritonPIMLowerToEmitC : Pass<"pim-lower-to-emitc", "mlir::ModuleOp"> {
  let summary = "Lower a single-DPU PIM kernel to EmitC, for any pim.num-tasklets >= 1";

  let description = [{
    Bridges the flagos-pim-compiler graph compiler to a numpy-executable
    artifact: lowers the pim-explicit-dma output (`pim.wram_alloc` /
    `pim.dma_load` / `pim.barrier` / `pim.wram_load` / `tt.dot` /
    `pim.wram_store` / `pim.dma_store`) into `emitc.*` ops that
    `-convert-func-to-emitc` and `mlir-translate --mlir-to-cpp` can turn into
    plain C.

    Scoped to a single DPU; every shape involved must be static.
    `pim.num-tasklets` (>= 1, missing attribute defaults to 1) is the degree
    to which the M dimension of a `tt.dot` is split: rather than one loop over
    the full `[0, M)` row range, the pass emits `num-tasklets` separate,
    sequential, statically-bounded blocks, each owning the row range
    `[tid*ceil(M/num-tasklets), min((tid+1)*ceil(M/num-tasklets), M))` for
    `tid` in `[0, num-tasklets)` -- mirroring the UPMEM convention of one
    tasklet per contiguous row block (see
    `runtime/kernels.py::tasklet_linear_kernel` in flagos-pim-compiler for the
    NumPy-side mirror of the same split). M and num-tasklets are both known
    statically at pass-compile time, so the `tid` blocks are unrolled here in
    the pass rather than emitted as a runtime loop; there is no runtime `tid`
    value anywhere in the generated C.

    `w` is snapshotted from MRAM into a heap buffer once, shared read-only by
    every block (it is never written, so it cannot alias `out`). Each block
    additionally snapshots only *its own* row range of `a` -- a per-tasklet
    WRAM tile, not the whole operand -- before writing that same row range of
    `out`. This ordering (read a block's `a` rows, then write that block's
    `out` rows, before moving to the next block) is what keeps `out` aliasing
    `a`'s MRAM bytes safe: `mem_planner` can reuse a dead activation's exact
    address for this op's output, so `a` and `out` can be the very same bytes,
    but only at matching row offsets -- output row `m` only ever depends on
    input row `m` of `a`, never another block's rows, so each block's
    snapshot-before-write is self-contained and there is no cross-block
    hazard. `pim.barrier` is not lowered to a runtime primitive because the
    blocks are emitted strictly in program order -- block `tid`'s snapshot and
    write are fully complete before block `tid+1`'s code executes, which is a
    stronger guarantee than a barrier provides. `num-tasklets == 1` is simply
    the degenerate case of a single block covering the whole M range -- there
    is no separate single-tasklet path. `pim.tasklet_id`/`pim.dpu_id` become
    the constant 0, as before -- no kernel this pass currently lowers reads
    either op, so there is no meaningful non-zero value to give them yet.

    `tt.dot` has no lowering anywhere in Triton or FlagTree below the tensor
    level -- every existing path either keeps its layout encoding for a tensor
    core or (on PIM) leaves it untouched. This pass is what actually expands
    it, into a `tid` loop wrapping a triple-nested `emitc.for` reduction over
    `emitc.subscript`/`emitc.load`/`emitc.assign`.

    Function arguments are rewritten from `!tt.ptr<T>` to `!emitc.ptr<T>`
    one-for-one -- deliberately not a single MRAM base plus per-argument byte
    offsets, so the emitted C carries no address arithmetic; the caller
    (`opcompiler_bridge/driver.py`) is expected to pass each tensor argument's
    already-offset pointer directly (e.g. `ctypes.data_as`). `tt.func` /
    `tt.return` are rewritten to `func.func`/`func.return` in the same pass,
    since `-convert-func-to-emitc` only recognizes the latter.

    A kernel outside this pass's scope (a DMA transfer `pim-explicit-dma`
    could not prove strided, or a dynamically shaped buffer) fails the pass
    rather than silently emitting something wrong.
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
    `pim.wram-bytes-used` module attribute, and fails if that exceeds the
    `pim.wram-bytes` budget.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];
}

def TritonPIMTileToBudget : Pass<"pim-tile-to-budget", "mlir::ModuleOp"> {
  let summary = "Infer PIM tile dimensions that fit the WRAM budget";

  let description = [{
    Prepares budget-aware PIM tiling. This pass validates the linear PIM IR
    tile against the module's hardware contract and writes `pim.tile-*`
    attributes for downstream lowering and cost extraction.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];
}

#endif // TRITONPIM_PASSES
