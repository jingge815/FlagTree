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

    The untiled (full) M/N/K of the linear operator is normally recovered by
    walking the `tt.dot`'s enclosing `scf.for` loops -- but real autotuned
    kernels (e.g. FlagGems `linear_kernel`) often split M/N across a 2-D
    launch grid instead of a loop, and pass M/N/K in as runtime scalar
    kernel arguments rather than IR-visible constants, so that structural
    walk cannot recover them. The `full-m`/`full-n`/`full-k` options let the
    caller (which already has the real values -- captured from the launch
    site in Python) supply them directly; when any is >= 0 it overrides
    the corresponding structural inference for every `tt.dot` in the
    module. A negative value (the default) means "not supplied", falling
    back to structural inference as before.

    When the module has no `tt.dot`, whole-operator `pim.matmul` ops (the
    B path) are tiled first: a matmul that does not fit `pim.wram-bytes`
    gets a `tile = [M, N, K]` attribute from the same search as the
    `tt.dot` path, restricted to tiles dividing M, N and the group width;
    the B path has no DMA, so `pim.dma-align` takes no part in choosing
    the tile. `-pim-lower-to-emitc` turns it into block loops. The module head carries
    one `pim.tile-m/n/k` and `pim.tile-wram-bytes`, written only when every
    matmul in the module is tiled with the same tile and the same bytes;
    otherwise each matmul keeps its own `tile` and the head is left out,
    with a remark.

    The pass then tiles `scf.for` loops whose bounds and step are constant
    and whose body carries tensor ops. A loop is tiled when the whole loop
    (bytes moved per iteration times the trip count) exceeds
    `pim.wram-bytes`; the rebuilt step is the original step times the
    largest power-of-two tile whose block still fits, falling back to
    smaller powers of two until one divides the extent. Only `pim.tile-m`
    is written -- a single loop extent has no N or K -- and a loop whose
    tile differs from a tile already in the module head is left unchanged.
    A loop is left unchanged, with a remark, when its step is not 1, when
    no searched tile divides the extent, when a load or store carries a
    mask, when its body contains a nested `scf.for`, when an offset already
    covers more than one row, when one iteration exceeds the budget, or
    when one iteration fits but no larger tile does. A module with nothing
    tileable is left unchanged. The pass requires the module to carry
    `pim.wram-bytes`, `pim.mram-bytes` and `pim.dma-align`, and fails
    otherwise.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];

  let options = [
    Option<"fullM", "full-m",
           "int64_t", /*default*/"-1",
           "the operator's untiled M, from the real launch args; -1 means"
           " infer from IR structure">,
    Option<"fullN", "full-n",
           "int64_t", /*default*/"-1",
           "the operator's untiled N, from the real launch args; -1 means"
           " infer from IR structure">,
    Option<"fullK", "full-k",
           "int64_t", /*default*/"-1",
           "the operator's untiled K, from the real launch args; -1 means"
           " infer from IR structure">,
  ];
}

def TritonPIMConstantFold : Pass<"pim-constant-fold", "mlir::ModuleOp"> {
  let summary = "Fold PIM and arith ops whose operands are all constants";

  let description = [{
    Evaluates `pim.eltwise`, `pim.lut` and `arith.addi`, `arith.subi` and
    `arith.muli` whose operands are all `arith.constant`, and replaces them
    with the result. The integer ops cover both scalars and dense i32
    tensors, which is the form address offsets take after an A-path kernel
    is converted to PIM. `pim.param`
    carries only a name and no value, so it is not foldable here.

    The upstream canonicalizer cannot do this: the PIM ops have no `hasFolder`
    and are not marked `Pure`. `pim.matmul` is deliberately left alone -- one
    operand is a whole weight tensor, and folding it would move a matrix
    multiply into compile time.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];
}

def TritonPIMCSE : Pass<"pim-cse", "mlir::ModuleOp"> {
  let summary = "Merge identical pure PIM ops inside one basic block";

  let description = [{
    Keeps the first of two `pim.*` ops that share an opcode, attributes and
    operands, and rewires the second's users to it.

    The upstream CSE only touches ops marked `Pure`, which `matmul`, `eltwise`
    and `lut` are not. Ops with memory effects (`dma_load`, `kv_cache`,
    `wram_alloc`) are skipped, as are ops in different blocks.
  }];

  let dependentDialects = ["mlir::triton::pim::TritonPIMDialect"];
}

def TritonPIMDCE : Pass<"pim-dce", "mlir::ModuleOp"> {
  let summary = "Erase pure PIM ops whose results have no users";

  let description = [{
    Erases a pure op whose every result has no user. In an operator-level
    function (first argument is a whole tensor or an on-chip buffer) the kernel
    output is the last batch of unused tensor results, the same rule
    `pim-lower-to-emitc` uses; the op producing it is kept whichever dialect it
    belongs to. Unused tensor results of `pim.*` ops are kept as well because
    they can be read out of band. A-path functions take pointers and hand their
    results out with `tt.store` or `tt.return`, so there every pure op is
    erased even when it produces tensors. `dma_store` and `kv_cache` survive
    because they have memory effects, not because their names are on an
    exclusion list.

    The upstream `symbol-dce` only removes unused symbols, so it cannot clean up
    dead ops inside a function body.
  }];

  let dependentDialects = ["mlir::triton::pim::TritonPIMDialect"];
}

def TritonPIMFuseActivation : Pass<"pim-fuse-activation", "mlir::ModuleOp"> {
  let summary = "Fold activations and trailing pools into their producing operator";

  let description = [{
    Rewrites `pim.matmul`/`pim.conv`/`pim.eltwise` followed by `pim.lut`, and
    optionally then by `pim.pool`, into a single operator carrying `activation`
    and `fusedPool` attributes.

    This is not an optimization. The target's graph format stores an activation
    *inside* its producer's node (its `contraction` block) and has no way to
    represent a standalone one, so the fusion has to happen before the graph
    hand-off or the result is unrepresentable. A trailing pool folds into the
    same node, after the activation.

    Only fuses when the intermediate value has exactly one use: with two or more
    consumers the pre-activation result is itself needed, and folding would
    delete a value someone reads.

    Operators that already carry an `activation` are left alone -- a second
    activation would have nowhere to go, since the graph format allows one per
    node.

    Example:

    ```mlir
    %m = pim.matmul %a, %w {datapath = ...} : ...
    %y = pim.lut %m {kind = #pim.activation<silu>} : ...
    ```

    becomes

    ```mlir
    %y = pim.matmul %a, %w {datapath = ..., activation = #pim.act_spec<kind = silu>} : ...
    ```
  }];

  let dependentDialects = ["mlir::triton::pim::TritonPIMDialect"];
}

def TritonPIMExpandPhases : Pass<"pim-expand-phases", "mlir::ModuleOp"> {
  let summary = "Expand opaque operators into hardware phase chains";

  let description = [{
    Rewrites three opaque operator-level ops into the phase SSA the target
    actually executes, one engine traversal per phase:

      pim.quantize {dynamic}  ->  4 phases (absmax, identity-scale, reciprocal, fp2int)
      pim.softmax             ->  5 phases (max, exp, sum, reciprocal, mul)
      pim.rope                ->  3 phases (mul-cos, mul-sin, add)

    The templates are op-local and hardcoded: every DQ node in the llama2
    reference artifact is 4 phases, every Softmax node is 5, with no
    exceptions. Other operators (`pim.matmul`, `pim.normalize`, …) are left
    alone.

    Each produced op carries `#pim.phase_spec`, whose `bytes` field is the
    logical buffer size, so a later orchestrator can allocate L2 without
    re-deriving shapes. It does not assign L2 offsets, Layer IDs, or Task IDs.

    DQ phases 1 and 2 both read phase 0 (a fan-out, not a chain). Softmax
    phase 4 reads both the exp and the reciprocal. Getting either dependency
    backwards is a silent 256x scale error.

    Example:

    ```mlir
    %q = pim.quantize %x, %s {dynamic, spec = #pim.quant_spec<
           granularity = per_group, axis = 1, groupSize = 128>}
         : tensor<1x4096xf16>, tensor<32xf16> -> tensor<1x4096xi8>
    ```

    becomes a four-op chain ending in a non-dynamic `pim.quantize` whose
    scale is the reciprocal of the per-group absmax.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];
}

def TritonPIMVerifyGmlContract : Pass<"pim-verify-gml-contract", "mlir::ModuleOp"> {
  let summary = "Check the cross-attribute invariants the graph hand-off relies on";

  let description = [{
    Reads the module and fails when an invariant that spans more than one
    attribute does not hold. It writes nothing: the graph hand-off is already
    described by the IR, and a pass that "fixed up" a phase count would be
    hiding the very disagreement a consumer needs to see.

    What it checks:

      - phase indices inside one function are unique and run `0..n-1` with no
        gaps, so a consumer that counts phases by distinct index gets the number
        the expansion meant;
      - every `reads` entry names a phase that exists in the same function. The
        attribute's own verifier already rejects a self- or forward reference --
        that it can do without seeing the rest of the chain -- but only here is
        the whole chain visible;
      - dynamic quantization's phases 1 and 2 both read phase 0. They are a
        fan-out, not a chain, and serializing them puts the reciprocal off by
        256x: a wrong answer with nothing anywhere to report it;
      - `pim.normalize` carries none of the fixed-function datapath fields. Its
        empty field set *is* the semantics -- it runs on the vector unit -- and
        filling in a default would emit keys the reference does not have.

    It does not look at the tile-level symbols (`tt.dot`, `pim.dma_*`): those
    belong to the other pipeline and are guarded by their own tests.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::pim::TritonPIMDialect"
  ];
}

#endif // TRITONPIM_PASSES
