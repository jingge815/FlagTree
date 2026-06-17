#ifndef TLE_RAW_OPS
#define TLE_RAW_OPS

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "tle/dialect/include/IR/TleDialect.td"
include "tle/dialect/include/IR/TleAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class Tle_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tle_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

// Begin TLE-Lite
def Tle_ExtractTileOp : Tle_Op<"extract_tile", [Pure]> {
  let summary = "Extract a tile from a tensor with static offsets";

  let description = [{
    Extract a contiguous tile from a tensor based on static offsets.
    This is a zero-cost operation at runtime - it only reorganizes
    register values within each thread, with no inter-thread communication.

    Requirements:
      1. Offsets must be multiples of CTA tile sizes
      2. Tile shape must be multiples of CTA tile sizes
      3. Source and destination layouts must match at CTA tile level
  }];

  let arguments = (ins TT_Tensor:$src, TT_IntLike:$index);

  let results = (outs TT_Tensor:$result);
  let builders = [OpBuilder<(ins "Value":$src, "Value":$index,
      "ArrayRef<int64_t>":$tileShape)>];
  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def Tle_InsertTileOp
    : Tle_Op<"insert_tile", [Pure,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

// End TLE-Lite

// Begin TLE-Raw

def Tle_TensorType : AnyTypeOf<[TT_Type, TTG_MemDescType]>;
def Tle_ArgType : AnyTypeOf<[Tle_TensorType, LLVMPointerType, LLVMStructType]>;
def Tle_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def Tle_LocalPointerIndexType : AnyTypeOf<[TT_Tensor, TT_Int]>;

def Tle_LocalPointersOp : Tle_Op<"local_pointers", [Pure]> {
  let arguments = (ins TTG_MemDescType:$src,
      Variadic<Tle_LocalPointerIndexType>:$indices);
  let results = (outs Tle_LocalPointerResultType:$result);
  let hasVerifier = 1;
}

def Tle_MemDescWGMMAViewOp : Tle_Op<"memdesc_wgmma_view", [Pure,
                                                           MemDescViewTrait,
                                                           SameOperandsAndResultElementType]> {
  let summary = "create a WGMMA descriptor view of a shared memory descriptor";

  let description = [{
    This operation returns a descriptor-only view of an existing shared memory
    allocation for WGMMA consumers. It does not allocate or move shared memory.

    The result type carries the logical matrix shape and WGMMA shared encoding
    required by the consumer. The source descriptor provides the underlying
    storage. The order attribute describes how logical offsets are mapped from
    the source view to the result view.
  }];

  let arguments = (
    ins TTG_MemDescType:$src,
    DenseI32ArrayAttr:$order
  );

  let results = (outs TTG_MemDescType:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let hasVerifier = 1;
}

def Tle_PipeCreateOp : Tle_Op<"pipe.create"> {
  let summary = "create a typed TLE GPU pipe";
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<ArrayAttr>:$readers,
    OptionalAttr<BoolAttr>:$one_shot
  );
  let assemblyFormat = "$fields attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_PipeWriterAcquireOp : Tle_Op<"pipe.writer_acquire"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields `[` $stage `,` $phase `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_PipeWriterCommitOp : Tle_Op<"pipe.writer_commit"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields `[` $stage `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_PipeWriterCloseOp : Tle_Op<"pipe.writer_close"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields `[` $stage `,` $phase `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_PipeReaderWaitOp : Tle_Op<"pipe.reader_wait"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<StrAttr>:$reader_name
  );
  let results = (outs I1:$is_closed);
  let assemblyFormat = "$fields `[` $stage `,` $phase `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_PipeReaderReleaseOp : Tle_Op<"pipe.reader_release", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<StrAttr>:$reader_name
  );
  let assemblyFormat = "$fields `[` $stage `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_TaskGridCreateOp : Tle_Op<"task_grid.create"> {
  let summary = "create a typed TLE task-grid dependency edge";
  let arguments = (ins
    Variadic<Tle_TensorType>:$fields,
    StrAttr:$scope,
    StrAttr:$grid_name,
    ArrayAttr:$field_names,
    OptionalAttr<DenseI64ArrayAttr>:$shape
  );
  let assemblyFormat = "$fields attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_TaskGridTileIdOp : Tle_Op<"task_grid.tile_id"> {
  let arguments = (ins
    Variadic<Tle_TensorType>:$fields,
    StrAttr:$scope,
    StrAttr:$grid_name,
    ArrayAttr:$field_names,
    OptionalAttr<DenseI64ArrayAttr>:$shape
  );
  let results = (outs Variadic<I32>:$ids);
  let hasVerifier = 1;
}

def Tle_TaskGridCommitOp : Tle_Op<"task_grid.commit"> {
  let arguments = (ins
    Variadic<Tle_TensorType>:$fields,
    StrAttr:$scope,
    StrAttr:$grid_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_TaskDeclareOp : Tle_Op<"task.declare"> {
  let summary = "declare a tile task domain and task-grid dependency maps";
  let description = [{
    Marker operation that attaches task metadata to the enclosing tt.func.
    The task body remains the enclosing function body for this milestone. The
    reads and writes arrays contain dictionaries with:

      - grid: string task-grid name
      - map: affine map from task domain coordinates to non-wildcard grid dims
      - wildcard_dims: optional i64 dense array for read-all grid dimensions

    Wildcard dimensions are allowed only on reads.
  }];
  let arguments = (ins
    StrAttr:$task_name,
    OptionalAttr<FlatSymbolRefAttr>:$callee,
    DenseI64ArrayAttr:$domain_shape,
    ArrayAttr:$reads,
    ArrayAttr:$writes
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tle_TaskGraphSchedulerOp : Tle_Op<"task_graph.scheduler"> {
  let summary = "materialized TLE mega task scheduler plan";
  let description = [{
    Compiler-generated scheduler boundary for a verified static task graph.
    This op records the tables needed by scheduler runtime codegen:

      - task_names: stable task-kind dispatch order
      - dispatch: mapping from task name to task-kind integer id
      - instances: task instances with dependency counts and produced tiles
      - edges: reverse dependency edges from produced tiles to consumers
      - initial_ready: initial ready queue contents

    It is intentionally a plan op, not an executable synchronization
    primitive. Lowering this op to device queues, counters, and dispatch is the
    next runtime codegen step.
  }];
  let arguments = (ins
    I64Attr:$num_tasks,
    I64Attr:$num_instances,
    I64Attr:$num_edges,
    I64Attr:$queue_capacity,
    StrAttr:$counter_type,
    ArrayAttr:$task_names,
    ArrayAttr:$dispatch,
    ArrayAttr:$instances,
    ArrayAttr:$edges,
    ArrayAttr:$initial_ready,
    DenseI32ArrayAttr:$task_domain_ranks,
    DenseI32ArrayAttr:$instance_task_ids,
    DenseI32ArrayAttr:$instance_dep_counts,
    DenseI32ArrayAttr:$instance_coord_offsets,
    DenseI32ArrayAttr:$instance_coords,
    DenseI32ArrayAttr:$initial_ready_ids,
    DenseI32ArrayAttr:$producer_edge_offsets,
    DenseI32ArrayAttr:$edge_consumer_ids
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tle_TaskGraphRuntimeStateOp : Tle_Op<"task_graph.runtime_state"> {
  let summary = "materialized TLE mega scheduler runtime state layout";
  let description = [{
    Compiler-generated layout contract for the persistent scheduler runtime
    state. This op does not execute scheduling by itself. It records the byte
    offsets and sizes required to initialize and operate the device-scope task
    queue and dependency counters for a ``tle.task_graph.scheduler`` plan.

    The MVP state model is single-kernel and i32-based:

      - init flag
      - queue lock
      - queue head and tail
      - completed task count
      - per-instance remaining dependency counters
      - ready queue storage

    Lowering must fail if it cannot map this layout to a device-scope storage
    allocation with acquire/release ordering.
  }];
  let arguments = (ins
    I64Attr:$num_instances,
    I64Attr:$queue_capacity,
    I64Attr:$state_size_bytes,
    I64Attr:$alignment_bytes,
    StrAttr:$counter_type,
    I64Attr:$counter_bytes,
    I64Attr:$queue_element_bytes,
    I64Attr:$init_flag_offset_bytes,
    I64Attr:$queue_lock_offset_bytes,
    I64Attr:$queue_head_offset_bytes,
    I64Attr:$queue_tail_offset_bytes,
    I64Attr:$completed_count_offset_bytes,
    I64Attr:$dep_counters_offset_bytes,
    I64Attr:$queue_storage_offset_bytes
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tle_ExclusiveCumsumOp : Tle_Op<"exclusive_cumsum", [Pure, SameOperandsAndResultEncoding]> {
  let arguments = (ins TT_Tensor:$src, I32Attr:$axis, BoolAttr:$reverse);
  let results = (outs TT_Tensor:$exclusive, TT_Type:$total);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($exclusive) `,` type($total)";
  let hasVerifier = 1;
}

def Tle_WGMMASharedOperandFenceOp : Tle_Op<"wgmma_shared_operand_fence"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$deps,
    BoolAttr:$bCluster
  );

  let summary = "fence shared-memory operands before WGMMA async-proxy reads";

  let description = [{
    Orders prior generic-proxy shared-memory writes to the listed WGMMA shared
    operands before a following WGMMA reads them through the async proxy.
  }];

  let assemblyFormat = "$deps attr-dict `:` type($deps)";
  let hasVerifier = 1;
}

def Tle_TMAStoreCommitGroupOp
    : Tle_Op<"tma_store.commit_group", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "commit the current TMA store bulk group";

  let description = [{
    Commits outstanding shared-to-global TMA store messages issued by this
    agent into the current bulk group.  This op makes TMA store group
    boundaries explicit in TLE IR so scheduling passes can group adjacent
    stores and place waits at the latest safe point.
  }];

  let assemblyFormat = "attr-dict";
}

def Tle_DistributedBarrierOp : Tle_Op<"distributed_barrier",
                                      [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$group_kind,
    OptionalAttr<I32Attr>:$group_rank,
    OptionalAttr<DenseI32ArrayAttr>:$group_shape,
    OptionalAttr<DenseI32ArrayAttr>:$group_axes,
    OptionalAttr<DenseI32ArrayAttr>:$group_mask
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tle_ClusterCTAIdOp : Tle_Op<"cluster_cta_id", [Pure]> {
  let summary = "return the hardware CTA rank inside the current cluster";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def Tle_RemotePointersOp : Tle_Op<"remote_pointers", [Pure]> {
  let arguments = (ins Tle_LocalPointerResultType:$src, TT_Int:$shard_id);
  let results = (outs Tle_LocalPointerResultType:$result);
  let hasVerifier = 1;
}

def Tle_DSLRegionOp : Tle_Op<"dsl_region", [IsolatedFromAbove, MemDescViewTrait,
                                            RecursiveMemoryEffects]> {
  let arguments = (ins
    Variadic<Tle_ArgType>:$inputs
  );
  let results = (outs Variadic<Tle_ArgType>:$outputs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Tle_YieldOp : Tle_Op<"yield", [Pure, Terminator, ReturnLike,
                                   HasParent<"DSLRegionOp">]> {
  let arguments = (ins Variadic<Tle_ArgType>:$inputs);
  let assemblyFormat = "attr-dict ($inputs^ `:` type($inputs))?";
}

def Tle_ExtractAllocatedPtrOp
    : Tle_Op<"extract_allocated_ptr", [Pure]> {
  let arguments = (ins Tle_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractAlignedPtrOp
    : Tle_Op<"extract_aligned_ptr", [Pure]> {
  let arguments = (ins Tle_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractOffsetOp
    : Tle_Op<"extract_offset", [Pure]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractSizesOp
    : Tle_Op<"extract_sizes", [Pure]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs Variadic<I64>:$sizes);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($sizes)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def Tle_ExtractStridesOp
    : Tle_Op<"extract_strides", [Pure]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs Variadic<I64>:$strides);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($strides)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def Tle_ExtractPtrOp : Tle_Op<"extract_ptr", [Pure]> {
  let arguments = (ins TT_Ptr:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_PackOp
    : Tle_Op<"pack", [MemDescViewTrait, Pure]> {
  let arguments = (ins LLVMStructType:$input);
  let results = (outs Tle_TensorType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

#endif
