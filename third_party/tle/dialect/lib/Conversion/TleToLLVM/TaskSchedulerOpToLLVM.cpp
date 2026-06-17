#include "tle/dialect/include/Conversion/TleToLLVM/TaskSchedulerOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace {

using namespace mlir;
namespace tle = mlir::triton::tle;

constexpr llvm::StringLiteral kTTGSharedAttr = "ttg.shared";
constexpr llvm::StringLiteral kTTGGlobalScratchSizeAttr =
    "ttg.global_scratch_memory_size";
constexpr llvm::StringLiteral kTTGGlobalScratchAlignAttr =
    "ttg.global_scratch_memory_alignment";
constexpr llvm::StringLiteral kTaskSchedulerScratchOffsetAttr =
    "tle.task_scheduler_scratch_offset";
constexpr llvm::StringLiteral kTaskSchedulerSharedOffsetAttr =
    "tle.task_scheduler_shared_offset";
constexpr int32_t kI32Bytes = 4;

struct RuntimeStateLayout {
  int64_t numInstances = 0;
  int64_t queueCapacity = 0;
  int64_t stateSizeBytes = 0;
  int64_t alignmentBytes = 0;
  int64_t initFlagOffsetBytes = 0;
  int64_t queueLockOffsetBytes = 0;
  int64_t queueHeadOffsetBytes = 0;
  int64_t queueTailOffsetBytes = 0;
  int64_t completedCountOffsetBytes = 0;
  int64_t depCountersOffsetBytes = 0;
  int64_t queueStorageOffsetBytes = 0;
};

struct RuntimeStatePointers {
  Value initFlag;
  Value queueLock;
  Value queueHead;
  Value queueTail;
  Value completedCount;
  Value depCounters;
  Value queueStorage;
  Value sharedTask;
};

static FailureOr<int64_t> requireI64(Operation *op, StringRef attrName,
                                     bool requirePositive) {
  auto attr = op->getAttrOfType<IntegerAttr>(attrName);
  if (!attr)
    return op->emitOpError("requires ") << attrName << " attribute";
  int64_t value = attr.getInt();
  if (requirePositive && value <= 0)
    return op->emitOpError("expects positive ") << attrName;
  if (!requirePositive && value < 0)
    return op->emitOpError("expects non-negative ") << attrName;
  return value;
}

static FailureOr<DenseI32ArrayAttr> requireI32Array(Operation *op,
                                                    StringRef attrName) {
  auto attr = op->getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!attr)
    return op->emitOpError("requires ") << attrName << " attribute";
  return attr;
}

static FailureOr<ArrayAttr> requireArray(Operation *op, StringRef attrName) {
  auto attr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!attr)
    return op->emitOpError("requires ") << attrName << " attribute";
  return attr;
}

static FailureOr<RuntimeStateLayout>
parseRuntimeState(tle::TaskGraphRuntimeStateOp op) {
  RuntimeStateLayout layout;
  FailureOr<int64_t> numInstances =
      requireI64(op, "num_instances", /*requirePositive=*/true);
  FailureOr<int64_t> queueCapacity =
      requireI64(op, "queue_capacity", /*requirePositive=*/true);
  FailureOr<int64_t> stateSize =
      requireI64(op, "state_size_bytes", /*requirePositive=*/true);
  FailureOr<int64_t> alignment =
      requireI64(op, "alignment_bytes", /*requirePositive=*/true);
  FailureOr<int64_t> initFlag =
      requireI64(op, "init_flag_offset_bytes", /*requirePositive=*/false);
  FailureOr<int64_t> queueLock =
      requireI64(op, "queue_lock_offset_bytes", /*requirePositive=*/false);
  FailureOr<int64_t> queueHead =
      requireI64(op, "queue_head_offset_bytes", /*requirePositive=*/false);
  FailureOr<int64_t> queueTail =
      requireI64(op, "queue_tail_offset_bytes", /*requirePositive=*/false);
  FailureOr<int64_t> completed =
      requireI64(op, "completed_count_offset_bytes",
                 /*requirePositive=*/false);
  FailureOr<int64_t> depCounters =
      requireI64(op, "dep_counters_offset_bytes", /*requirePositive=*/false);
  FailureOr<int64_t> queueStorage =
      requireI64(op, "queue_storage_offset_bytes", /*requirePositive=*/false);
  if (failed(numInstances) || failed(queueCapacity) || failed(stateSize) ||
      failed(alignment) || failed(initFlag) || failed(queueLock) ||
      failed(queueHead) || failed(queueTail) || failed(completed) ||
      failed(depCounters) || failed(queueStorage))
    return failure();
  layout.numInstances = *numInstances;
  layout.queueCapacity = *queueCapacity;
  layout.stateSizeBytes = *stateSize;
  layout.alignmentBytes = *alignment;
  layout.initFlagOffsetBytes = *initFlag;
  layout.queueLockOffsetBytes = *queueLock;
  layout.queueHeadOffsetBytes = *queueHead;
  layout.queueTailOffsetBytes = *queueTail;
  layout.completedCountOffsetBytes = *completed;
  layout.depCountersOffsetBytes = *depCounters;
  layout.queueStorageOffsetBytes = *queueStorage;
  return layout;
}

static FailureOr<int32_t> checkedI32(Operation *op, int64_t value,
                                     StringRef what) {
  if (value < 0 || value > std::numeric_limits<int32_t>::max())
    return op->emitOpError() << what << " exceeds i32 range";
  return static_cast<int32_t>(value);
}

static FailureOr<int32_t> reserveGlobalScratch(ModuleOp mod, int64_t size,
                                               int64_t alignment) {
  if (auto existing =
          mod->getAttrOfType<IntegerAttr>(kTaskSchedulerScratchOffsetAttr)) {
    int64_t value = existing.getInt();
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
      return failure();
    return static_cast<int32_t>(value);
  }

  auto *ctx = mod.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);

  int64_t currentSize = 0;
  if (auto attr =
          mod->getAttrOfType<IntegerAttr>(kTTGGlobalScratchSizeAttr)) {
    currentSize = attr.getInt();
    if (currentSize < 0)
      return failure();
  } else {
    mod->setAttr(kTTGGlobalScratchSizeAttr, IntegerAttr::get(i32Ty, 0));
  }

  int64_t currentAlign = 1;
  if (auto attr =
          mod->getAttrOfType<IntegerAttr>(kTTGGlobalScratchAlignAttr)) {
    currentAlign = attr.getInt();
    if (currentAlign <= 0)
      return failure();
  } else {
    mod->setAttr(kTTGGlobalScratchAlignAttr, IntegerAttr::get(i32Ty, 1));
  }

  int64_t offset = llvm::alignTo(currentSize, alignment);
  int64_t newSize = offset + size;
  if (newSize > std::numeric_limits<int32_t>::max())
    return failure();
  int64_t newAlign = std::max(currentAlign, alignment);
  mod->setAttr(kTTGGlobalScratchSizeAttr, IntegerAttr::get(i32Ty, newSize));
  mod->setAttr(kTTGGlobalScratchAlignAttr, IntegerAttr::get(i32Ty, newAlign));
  mod->setAttr(kTaskSchedulerScratchOffsetAttr,
               IntegerAttr::get(i32Ty, offset));
  return static_cast<int32_t>(offset);
}

static FailureOr<int32_t> reserveSharedScratch(ModuleOp mod) {
  if (auto existing =
          mod->getAttrOfType<IntegerAttr>(kTaskSchedulerSharedOffsetAttr)) {
    int64_t value = existing.getInt();
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
      return failure();
    return static_cast<int32_t>(value);
  }

  auto *ctx = mod.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  int64_t currentShared = 0;
  if (auto attr = mod->getAttrOfType<IntegerAttr>(kTTGSharedAttr)) {
    currentShared = attr.getInt();
    if (currentShared < 0)
      return failure();
  } else {
    mod->setAttr(kTTGSharedAttr, IntegerAttr::get(i32Ty, 0));
  }

  int64_t offset = llvm::alignTo(currentShared, int64_t{kI32Bytes});
  int64_t newShared = offset + kI32Bytes;
  if (newShared > std::numeric_limits<int32_t>::max())
    return failure();
  mod->setAttr(kTTGSharedAttr, IntegerAttr::get(i32Ty, newShared));
  mod->setAttr(kTaskSchedulerSharedOffsetAttr,
               IntegerAttr::get(i32Ty, offset));
  return static_cast<int32_t>(offset);
}

static Value gepI32Ptr(ConversionPatternRewriter &rewriter, Location loc,
                       Value base, int64_t byteOffset) {
  mlir::triton::TritonLLVMOpBuilder b(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(base.getType());
  auto i8Ty = IntegerType::get(ctx, 8);
  auto i32PtrTy = LLVM::LLVMPointerType::get(ctx, ptrTy.getAddressSpace());
  Value bytePtr = b.gep(ptrTy, i8Ty, base, b.i32_val(byteOffset));
  return b.bitcast(bytePtr, i32PtrTy);
}

static RuntimeStatePointers makeRuntimePointers(
    ConversionPatternRewriter &rewriter, Location loc, ModuleOp mod,
    LLVM::LLVMFuncOp func, const RuntimeStateLayout &layout,
    int32_t globalOffset, int32_t sharedOffset) {
  mlir::triton::TritonLLVMOpBuilder b(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto i8Ty = IntegerType::get(ctx, 8);

  Value globalBase = func.getArgument(func.getNumArguments() +
                                      mlir::kGlobalScratchBufferOffset);
  auto globalPtrTy = cast<LLVM::LLVMPointerType>(globalBase.getType());
  Value stateBase =
      b.gep(globalPtrTy, i8Ty, globalBase, b.i32_val(globalOffset));

  auto globalSmem = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
  Value sharedBase = LLVM::AddressOfOp::create(rewriter, loc, globalSmem);
  auto sharedPtrTy =
      LLVM::LLVMPointerType::get(ctx, static_cast<unsigned>(
                                          NVVM::NVVMMemorySpace::Shared));
  sharedBase = b.bitcast(sharedBase, sharedPtrTy);

  return RuntimeStatePointers{
      gepI32Ptr(rewriter, loc, stateBase, layout.initFlagOffsetBytes),
      gepI32Ptr(rewriter, loc, stateBase, layout.queueLockOffsetBytes),
      gepI32Ptr(rewriter, loc, stateBase, layout.queueHeadOffsetBytes),
      gepI32Ptr(rewriter, loc, stateBase, layout.queueTailOffsetBytes),
      gepI32Ptr(rewriter, loc, stateBase, layout.completedCountOffsetBytes),
      gepI32Ptr(rewriter, loc, stateBase, layout.depCountersOffsetBytes),
      gepI32Ptr(rewriter, loc, stateBase, layout.queueStorageOffsetBytes),
      gepI32Ptr(rewriter, loc, sharedBase, sharedOffset)};
}

static Value gepI32Element(ConversionPatternRewriter &rewriter, Location loc,
                           Value base, Value index) {
  mlir::triton::TritonLLVMOpBuilder b(loc, rewriter);
  auto ptrTy = cast<LLVM::LLVMPointerType>(base.getType());
  auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
  return b.gep(ptrTy, i32Ty, base, index);
}

static void storeI32(ConversionPatternRewriter &rewriter, Location loc,
                     Value ptr, Value value) {
  LLVM::StoreOp::create(rewriter, loc, value, ptr);
}

static Value loadI32(ConversionPatternRewriter &rewriter, Location loc,
                     Value ptr) {
  auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
  return LLVM::LoadOp::create(rewriter, loc, i32Ty, ptr);
}

static void emitDeviceMembar(ConversionPatternRewriter &rewriter,
                             Location loc) {
  mlir::triton::PTXBuilder ptxBuilder;
  auto &membar = *ptxBuilder.create("membar.gl;");
  membar({}, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static Value atomicLoadAcquire(ConversionPatternRewriter &rewriter,
                               Location loc, Value ptr) {
  mlir::triton::TritonLLVMOpBuilder b(loc, rewriter);
  return LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::add, ptr,
                                   b.i32_val(0),
                                   LLVM::AtomicOrdering::acquire,
                                   StringRef("device"))
      .getResult();
}

static Value atomicAdd(ConversionPatternRewriter &rewriter, Location loc,
                       Value ptr, Value value,
                       LLVM::AtomicOrdering ordering) {
  return LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::add, ptr,
                                   value, ordering, StringRef("device"))
      .getResult();
}

static Value atomicXchg(ConversionPatternRewriter &rewriter, Location loc,
                        Value ptr, Value value,
                        LLVM::AtomicOrdering ordering) {
  return LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::xchg, ptr,
                                   value, ordering, StringRef("device"))
      .getResult();
}

static LogicalResult collectCallees(tle::TaskGraphSchedulerOp scheduler,
                                    SmallVectorImpl<FlatSymbolRefAttr> &callees) {
  FailureOr<int64_t> numTasks =
      requireI64(scheduler, "num_tasks", /*requirePositive=*/true);
  FailureOr<ArrayAttr> dispatch = requireArray(scheduler, "dispatch");
  if (failed(numTasks) || failed(dispatch))
    return failure();
  callees.assign(*numTasks, FlatSymbolRefAttr());
  for (Attribute attr : *dispatch) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return scheduler.emitOpError()
             << "expects dispatch entries to be dictionaries";
    auto taskId = dyn_cast_or_null<IntegerAttr>(dict.get("task_id"));
    auto callee = dyn_cast_or_null<FlatSymbolRefAttr>(dict.get("callee"));
    if (!taskId || taskId.getInt() < 0 || taskId.getInt() >= *numTasks)
      return scheduler.emitOpError()
             << "expects dispatch task_id to be in task range";
    if (!callee)
      return scheduler.emitOpError()
             << "requires dispatch callee for persistent scheduler lowering";
    callees[taskId.getInt()] = callee;
  }
  return success();
}

static LogicalResult rejectGlobalScratchCallees(
    tle::TaskGraphSchedulerOp scheduler, ModuleOp module,
    ArrayRef<FlatSymbolRefAttr> callees) {
  for (FlatSymbolRefAttr callee : callees) {
    auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(callee.getValue());
    if (!func)
      return scheduler.emitOpError()
             << "cannot resolve task callee @" << callee.getValue()
             << " after function conversion";
    auto scratchSize =
        func->getAttrOfType<IntegerAttr>(kTTGGlobalScratchSizeAttr);
    if (scratchSize && scratchSize.getInt() != 0)
      return scheduler.emitOpError()
             << "persistent scheduler MVP does not support task callee @"
             << callee.getValue()
             << " requiring global scratch memory; scheduler runtime state "
                "currently occupies the raw kernel global scratch base";
  }
  return success();
}

struct TaskSchedulerOpConversion
    : public ConvertOpToLLVMPattern<tle::TaskGraphSchedulerOp> {
  using ConvertOpToLLVMPattern<
      tle::TaskGraphSchedulerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tle::TaskGraphSchedulerOp scheduler, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = scheduler.getLoc();
    auto *ctx = rewriter.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    mlir::triton::TritonLLVMOpBuilder b(loc, rewriter);

    auto func = scheduler->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func)
      return scheduler.emitOpError(
          "persistent scheduler lowering requires LLVM function context");
    if (!func->hasAttr(NVVM::NVVMDialect::getKernelFuncAttrName()))
      return scheduler.emitOpError(
          "persistent scheduler lowering requires a kernel function");
    if (func.getNumArguments() < 2)
      return scheduler.emitOpError(
          "persistent scheduler lowering requires global scratch argument");

    auto module = scheduler->getParentOfType<ModuleOp>();
    if (!module)
      return scheduler.emitOpError(
          "persistent scheduler lowering requires parent module");
    if (!module->getAttrOfType<IntegerAttr>("tle.requires_cooperative_grid"))
      return scheduler.emitOpError(
          "persistent scheduler lowering requires cooperative-grid launch "
          "metadata from runtime state materialization");
    if (!module.lookupSymbol<LLVM::GlobalOp>("global_smem"))
      return scheduler.emitOpError(
          "persistent scheduler lowering requires global_smem symbol");

    SmallVector<tle::TaskGraphRuntimeStateOp> runtimeStates;
    func.walk([&](tle::TaskGraphRuntimeStateOp op) {
      runtimeStates.push_back(op);
    });
    if (runtimeStates.size() != 1)
      return scheduler.emitOpError(
          "persistent scheduler lowering requires exactly one runtime state");
    FailureOr<RuntimeStateLayout> layout =
        parseRuntimeState(runtimeStates.front());
    if (failed(layout))
      return failure();

    FailureOr<int64_t> numInstances =
        requireI64(scheduler, "num_instances", /*requirePositive=*/true);
    FailureOr<int64_t> queueCapacity =
        requireI64(scheduler, "queue_capacity", /*requirePositive=*/true);
    FailureOr<DenseI32ArrayAttr> instanceTaskIds =
        requireI32Array(scheduler, "instance_task_ids");
    FailureOr<DenseI32ArrayAttr> instanceDepCounts =
        requireI32Array(scheduler, "instance_dep_counts");
    FailureOr<DenseI32ArrayAttr> coordOffsets =
        requireI32Array(scheduler, "instance_coord_offsets");
    FailureOr<DenseI32ArrayAttr> coords =
        requireI32Array(scheduler, "instance_coords");
    FailureOr<DenseI32ArrayAttr> initialReady =
        requireI32Array(scheduler, "initial_ready_ids");
    FailureOr<DenseI32ArrayAttr> edgeOffsets =
        requireI32Array(scheduler, "producer_edge_offsets");
    FailureOr<DenseI32ArrayAttr> edgeConsumers =
        requireI32Array(scheduler, "edge_consumer_ids");
    if (failed(numInstances) || failed(queueCapacity) ||
        failed(instanceTaskIds) || failed(instanceDepCounts) ||
        failed(coordOffsets) || failed(coords) || failed(initialReady) ||
        failed(edgeOffsets) || failed(edgeConsumers))
      return failure();
    if (*numInstances != layout->numInstances ||
        *queueCapacity != layout->queueCapacity)
      return scheduler.emitOpError()
             << "scheduler and runtime state disagree on queue dimensions";

    SmallVector<FlatSymbolRefAttr> callees;
    if (failed(collectCallees(scheduler, callees)))
      return failure();
    if (failed(rejectGlobalScratchCallees(scheduler, module, callees)))
      return failure();

    FailureOr<int32_t> globalOffset = reserveGlobalScratch(
        module, layout->stateSizeBytes, layout->alignmentBytes);
    FailureOr<int32_t> sharedOffset = reserveSharedScratch(module);
    if (failed(globalOffset) || failed(sharedOffset))
      return scheduler.emitOpError(
          "failed to reserve scheduler global/shared scratch");

    RuntimeStatePointers ptrs =
        makeRuntimePointers(rewriter, loc, module, func, *layout,
                            *globalOffset, *sharedOffset);

    unsigned graphArgCount = func.getNumArguments() - 2;
    SmallVector<Value> graphArgs;
    graphArgs.reserve(graphArgCount);
    for (unsigned i = 0; i < graphArgCount; ++i)
      graphArgs.push_back(func.getArgument(i));

    ArrayRef<int32_t> taskIds = (*instanceTaskIds).asArrayRef();
    ArrayRef<int32_t> depCounts = (*instanceDepCounts).asArrayRef();
    ArrayRef<int32_t> coordOffsetValues = (*coordOffsets).asArrayRef();
    ArrayRef<int32_t> coordValues = (*coords).asArrayRef();
    ArrayRef<int32_t> initialReadyIds = (*initialReady).asArrayRef();
    ArrayRef<int32_t> producerEdgeOffsets = (*edgeOffsets).asArrayRef();
    ArrayRef<int32_t> consumerIds = (*edgeConsumers).asArrayRef();

    if (producerEdgeOffsets.size() != static_cast<size_t>(*numInstances + 1))
      return scheduler.emitOpError("producer edge offsets size mismatch");

    SmallVector<Operation *> metadata;
    func.walk([&](Operation *op) {
      if (isa<tle::TaskDeclareOp, tle::TaskGridCreateOp,
              tle::TaskGridTileIdOp, tle::TaskGridCommitOp,
              tle::TaskGraphRuntimeStateOp>(op))
        metadata.push_back(op);
    });

    Block *entryBlock = rewriter.getInsertionBlock();
    Block *tailBlock = entryBlock->splitBlock(rewriter.getInsertionPoint());
    Block *initBlock = rewriter.createBlock(tailBlock);
    Block *waitInitBlock = rewriter.createBlock(tailBlock);
    Block *loopBlock = rewriter.createBlock(tailBlock);
    Block *lockBlock = rewriter.createBlock(tailBlock);
    Block *popBlock = rewriter.createBlock(tailBlock);
    Block *popHasWorkBlock = rewriter.createBlock(tailBlock);
    Block *popNoWorkBlock = rewriter.createBlock(tailBlock);
    Block *popReleaseBlock = rewriter.createBlock(tailBlock);
    popReleaseBlock->addArgument(i32Ty, loc);
    Block *waitPopBlock = rewriter.createBlock(tailBlock);
    Block *retryCheckBlock = rewriter.createBlock(tailBlock);
    retryCheckBlock->addArgument(i32Ty, loc);
    Block *dispatchBlock = rewriter.createBlock(tailBlock);
    dispatchBlock->addArgument(i32Ty, loc);
    Block *invalidTaskBlock = rewriter.createBlock(tailBlock);

    SmallVector<Block *> checkBlocks;
    SmallVector<Block *> taskBlocks;
    checkBlocks.reserve(*numInstances);
    taskBlocks.reserve(*numInstances);
    for (int64_t i = 0; i < *numInstances; ++i) {
      Block *check = rewriter.createBlock(tailBlock);
      check->addArgument(i32Ty, loc);
      checkBlocks.push_back(check);
      taskBlocks.push_back(rewriter.createBlock(tailBlock));
    }

    rewriter.setInsertionPointToEnd(entryBlock);
    Value threadId = getThreadId(rewriter, loc);
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Value blockIdX = rewriter.create<NVVM::BlockIdXOp>(loc, i32Ty);
    Value blockIdY = rewriter.create<NVVM::BlockIdYOp>(loc, i32Ty);
    Value blockIdZ = rewriter.create<NVVM::BlockIdZOp>(loc, i32Ty);
    Value isBlock0 =
        b.and_(b.and_(b.icmp_eq(blockIdX, b.i32_val(0)),
                      b.icmp_eq(blockIdY, b.i32_val(0))),
               b.icmp_eq(blockIdZ, b.i32_val(0)));
    Value isInitializer = b.and_(isThread0, isBlock0);

    rewriter.create<LLVM::CondBrOp>(loc, isInitializer, initBlock,
                                    ValueRange{}, waitInitBlock,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(initBlock);
    storeI32(rewriter, loc, ptrs.initFlag, b.i32_val(0));
    storeI32(rewriter, loc, ptrs.queueLock, b.i32_val(0));
    storeI32(rewriter, loc, ptrs.queueHead, b.i32_val(0));
    storeI32(rewriter, loc, ptrs.queueTail,
             b.i32_val(initialReadyIds.size()));
    storeI32(rewriter, loc, ptrs.completedCount, b.i32_val(0));
    for (auto [index, depCount] : llvm::enumerate(depCounts)) {
      Value depPtr =
          gepI32Element(rewriter, loc, ptrs.depCounters, b.i32_val(index));
      storeI32(rewriter, loc, depPtr, b.i32_val(depCount));
    }
    for (auto [index, readyId] : llvm::enumerate(initialReadyIds)) {
      Value queuePtr =
          gepI32Element(rewriter, loc, ptrs.queueStorage, b.i32_val(index));
      storeI32(rewriter, loc, queuePtr, b.i32_val(readyId));
    }
    atomicXchg(rewriter, loc, ptrs.initFlag, b.i32_val(1),
               LLVM::AtomicOrdering::release);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, waitInitBlock);

    rewriter.setInsertionPointToEnd(waitInitBlock);
    Value initFlag = atomicLoadAcquire(rewriter, loc, ptrs.initFlag);
    Value initialized = b.icmp_eq(initFlag, b.i32_val(1));
    rewriter.create<LLVM::CondBrOp>(loc, initialized, loopBlock,
                                    ValueRange{}, waitInitBlock,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(loopBlock);
    rewriter.create<LLVM::CondBrOp>(loc, isThread0, lockBlock, ValueRange{},
                                    waitPopBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(lockBlock);
    Value oldLock = atomicXchg(rewriter, loc, ptrs.queueLock, b.i32_val(1),
                               LLVM::AtomicOrdering::acq_rel);
    Value lockAcquired = b.icmp_eq(oldLock, b.i32_val(0));
    rewriter.create<LLVM::CondBrOp>(loc, lockAcquired, popBlock,
                                    ValueRange{}, lockBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(popBlock);
    Value head = atomicLoadAcquire(rewriter, loc, ptrs.queueHead);
    Value tail = atomicLoadAcquire(rewriter, loc, ptrs.queueTail);
    Value hasWork = b.icmp_slt(head, tail);
    rewriter.create<LLVM::CondBrOp>(loc, hasWork, popHasWorkBlock,
                                    ValueRange{}, popNoWorkBlock,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(popHasWorkBlock);
    Value queueIndex = b.urem(head, b.i32_val(*queueCapacity));
    Value queuePtr = gepI32Element(rewriter, loc, ptrs.queueStorage, queueIndex);
    Value selectedTask = loadI32(rewriter, loc, queuePtr);
    storeI32(rewriter, loc, ptrs.queueHead, b.add(head, b.i32_val(1)));
    rewriter.create<LLVM::BrOp>(loc, ValueRange{selectedTask},
                                popReleaseBlock);

    rewriter.setInsertionPointToEnd(popNoWorkBlock);
    Value completed = atomicLoadAcquire(rewriter, loc, ptrs.completedCount);
    Value isDone = b.icmp_sge(completed, b.i32_val(*numInstances));
    Value noTask = b.select(isDone, b.i32_val(-1), b.i32_val(-2));
    rewriter.create<LLVM::BrOp>(loc, ValueRange{noTask}, popReleaseBlock);

    rewriter.setInsertionPointToEnd(popReleaseBlock);
    Value poppedTask = popReleaseBlock->getArgument(0);
    storeI32(rewriter, loc, ptrs.sharedTask, poppedTask);
    atomicXchg(rewriter, loc, ptrs.queueLock, b.i32_val(0),
               LLVM::AtomicOrdering::release);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, waitPopBlock);

    rewriter.setInsertionPointToEnd(waitPopBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    Value ctaTask = loadI32(rewriter, loc, ptrs.sharedTask);
    Value shouldExit = b.icmp_eq(ctaTask, b.i32_val(-1));
    rewriter.create<LLVM::CondBrOp>(loc, shouldExit, tailBlock, ValueRange{},
                                    retryCheckBlock, ValueRange{ctaTask});

    rewriter.setInsertionPointToEnd(retryCheckBlock);
    Value retryTask = retryCheckBlock->getArgument(0);
    Value shouldRetry = b.icmp_eq(retryTask, b.i32_val(-2));
    rewriter.create<LLVM::CondBrOp>(loc, shouldRetry, loopBlock,
                                    ValueRange{}, dispatchBlock,
                                    ValueRange{retryTask});

    rewriter.setInsertionPointToEnd(dispatchBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{dispatchBlock->getArgument(0)},
                                checkBlocks.front());

    for (int64_t i = 0; i < *numInstances; ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      Value id = checkBlocks[i]->getArgument(0);
      Value isInstance = b.icmp_eq(id, b.i32_val(i));
      Block *nextBlock =
          (i + 1 == *numInstances) ? invalidTaskBlock : checkBlocks[i + 1];
      rewriter.create<LLVM::CondBrOp>(
          loc, isInstance, taskBlocks[i], ValueRange{}, nextBlock,
          (nextBlock == invalidTaskBlock) ? ValueRange{} : ValueRange{id});
    }

    rewriter.setInsertionPointToEnd(invalidTaskBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, loopBlock);

    auto emitEnqueue = [&](Block *currentBlock, int32_t consumerId,
                           Block *continuation) {
      Block *enqueueLock = rewriter.createBlock(tailBlock);
      Block *enqueueHaveLock = rewriter.createBlock(tailBlock);

      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, enqueueLock);

      rewriter.setInsertionPointToEnd(enqueueLock);
      Value enqueueOldLock =
          atomicXchg(rewriter, loc, ptrs.queueLock, b.i32_val(1),
                     LLVM::AtomicOrdering::acq_rel);
      Value enqueueLockAcquired = b.icmp_eq(enqueueOldLock, b.i32_val(0));
      rewriter.create<LLVM::CondBrOp>(loc, enqueueLockAcquired,
                                      enqueueHaveLock, ValueRange{},
                                      enqueueLock, ValueRange{});

      rewriter.setInsertionPointToEnd(enqueueHaveLock);
      Value enqueueTail = atomicLoadAcquire(rewriter, loc, ptrs.queueTail);
      Value enqueueIndex = b.urem(enqueueTail, b.i32_val(*queueCapacity));
      Value enqueuePtr =
          gepI32Element(rewriter, loc, ptrs.queueStorage, enqueueIndex);
      storeI32(rewriter, loc, enqueuePtr, b.i32_val(consumerId));
      storeI32(rewriter, loc, ptrs.queueTail,
               b.add(enqueueTail, b.i32_val(1)));
      atomicXchg(rewriter, loc, ptrs.queueLock, b.i32_val(0),
                 LLVM::AtomicOrdering::release);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuation);
    };

    for (int64_t instanceId = 0; instanceId < *numInstances; ++instanceId) {
      Block *commitBlock = rewriter.createBlock(tailBlock);
      Block *commitDoneBlock = rewriter.createBlock(tailBlock);

      rewriter.setInsertionPointToEnd(taskBlocks[instanceId]);
      int32_t taskId = taskIds[instanceId];
      if (taskId < 0 || taskId >= static_cast<int32_t>(callees.size()))
        return scheduler.emitOpError("task id is outside dispatch range");
      int32_t coordBegin = coordOffsetValues[instanceId];
      int32_t coordEnd = coordOffsetValues[instanceId + 1];
      if (coordBegin < 0 || coordEnd < coordBegin ||
          coordEnd > static_cast<int32_t>(coordValues.size()))
        return scheduler.emitOpError("instance coordinate offsets mismatch");

      SmallVector<Value> operands;
      operands.reserve((coordEnd - coordBegin) + graphArgs.size());
      for (int32_t idx = coordBegin; idx < coordEnd; ++idx) {
        operands.push_back(arith::ConstantIntOp::create(rewriter, loc,
                                                        coordValues[idx], 32));
      }
      operands.append(graphArgs.begin(), graphArgs.end());
      mlir::triton::CallOp::create(rewriter, loc, callees[taskId],
                                   TypeRange(), operands);
      // All CTA lanes may participate in a task body and write global memory.
      // Publish those writes before thread 0 commits dependency counters that
      // can wake consumers on other CTAs.
      emitDeviceMembar(rewriter, loc);
      rewriter.create<mlir::gpu::BarrierOp>(loc);
      rewriter.create<LLVM::CondBrOp>(loc, isThread0, commitBlock,
                                      ValueRange{}, commitDoneBlock,
                                      ValueRange{});

      Block *currentCommitBlock = commitBlock;
      int32_t edgeBegin = producerEdgeOffsets[instanceId];
      int32_t edgeEnd = producerEdgeOffsets[instanceId + 1];
      if (edgeBegin < 0 || edgeEnd < edgeBegin ||
          edgeEnd > static_cast<int32_t>(consumerIds.size()))
        return scheduler.emitOpError("producer edge offsets mismatch");
      for (int32_t edge = edgeBegin; edge < edgeEnd; ++edge) {
        int32_t consumerId = consumerIds[edge];
        if (consumerId < 0 || consumerId >= *numInstances)
          return scheduler.emitOpError("edge consumer id is out of range");
        Block *readyBlock = rewriter.createBlock(tailBlock);
        Block *nextCommitBlock = rewriter.createBlock(tailBlock);

        rewriter.setInsertionPointToEnd(currentCommitBlock);
        Value depPtr = gepI32Element(rewriter, loc, ptrs.depCounters,
                                     b.i32_val(consumerId));
        Value oldDep = atomicAdd(rewriter, loc, depPtr, b.i32_val(-1),
                                 LLVM::AtomicOrdering::acq_rel);
        Value nowReady = b.icmp_eq(oldDep, b.i32_val(1));
        rewriter.create<LLVM::CondBrOp>(loc, nowReady, readyBlock,
                                        ValueRange{}, nextCommitBlock,
                                        ValueRange{});
        emitEnqueue(readyBlock, consumerId, nextCommitBlock);
        currentCommitBlock = nextCommitBlock;
      }

      rewriter.setInsertionPointToEnd(currentCommitBlock);
      atomicAdd(rewriter, loc, ptrs.completedCount, b.i32_val(1),
                LLVM::AtomicOrdering::acq_rel);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, commitDoneBlock);

      rewriter.setInsertionPointToEnd(commitDoneBlock);
      rewriter.create<mlir::gpu::BarrierOp>(loc);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, loopBlock);
    }

    scheduler.erase();
    for (Operation *op : metadata) {
      if (!op->use_empty())
        return op->emitOpError("task graph metadata unexpectedly has uses");
      op->erase();
    }
    return success();
  }
};

} // namespace

void tle::populateTaskSchedulerOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<TaskSchedulerOpConversion>(typeConverter, benefit);
}
