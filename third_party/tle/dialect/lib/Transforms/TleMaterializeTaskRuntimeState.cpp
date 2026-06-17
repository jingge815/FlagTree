// flagtree tle
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include <limits>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEMATERIALIZETASKRUNTIMESTATE
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kI32Bytes = 4;
constexpr int64_t kHeaderWords = 5;
constexpr int64_t kRuntimeStateAlignment = 4;

static FailureOr<int64_t> requireI64(TaskGraphSchedulerOp scheduler,
                                     StringRef attrName,
                                     bool requirePositive) {
  auto attr = scheduler->getAttrOfType<IntegerAttr>(attrName);
  if (!attr)
    return scheduler.emitOpError("requires ") << attrName << " attribute";
  int64_t value = attr.getInt();
  if (requirePositive && value <= 0)
    return scheduler.emitOpError("expects positive ") << attrName;
  if (!requirePositive && value < 0)
    return scheduler.emitOpError("expects non-negative ") << attrName;
  return value;
}

static FailureOr<int64_t> checkedStateSize(TaskGraphSchedulerOp scheduler,
                                           int64_t numInstances,
                                           int64_t queueCapacity) {
  int64_t max = std::numeric_limits<int64_t>::max();
  if (numInstances > max / kI32Bytes ||
      queueCapacity > max / kI32Bytes)
    return scheduler.emitOpError("runtime state byte size overflows i64");

  int64_t depBytes = numInstances * kI32Bytes;
  int64_t queueBytes = queueCapacity * kI32Bytes;
  int64_t headerBytes = kHeaderWords * kI32Bytes;
  if (depBytes > max - headerBytes ||
      queueBytes > max - (headerBytes + depBytes))
    return scheduler.emitOpError("runtime state byte size overflows i64");
  return headerBytes + depBytes + queueBytes;
}

static LogicalResult
materializeRuntimeState(FunctionOpInterface func,
                        TaskGraphSchedulerOp scheduler) {
  auto counterType = scheduler->getAttrOfType<StringAttr>("counter_type");
  if (!counterType)
    return scheduler.emitOpError("requires counter_type attribute");
  if (counterType.getValue() != "i32")
    return scheduler.emitOpError("MVP supports only counter_type = \"i32\"");

  FailureOr<int64_t> numInstances =
      requireI64(scheduler, "num_instances", /*requirePositive=*/true);
  FailureOr<int64_t> queueCapacity =
      requireI64(scheduler, "queue_capacity", /*requirePositive=*/true);
  if (failed(numInstances) || failed(queueCapacity))
    return failure();
  if (*queueCapacity < *numInstances)
    return scheduler.emitOpError()
           << "expects queue_capacity to cover all task instances";

  FailureOr<int64_t> stateSize =
      checkedStateSize(scheduler, *numInstances, *queueCapacity);
  if (failed(stateSize))
    return failure();

  int64_t initFlagOffset = 0;
  int64_t queueLockOffset = initFlagOffset + kI32Bytes;
  int64_t queueHeadOffset = queueLockOffset + kI32Bytes;
  int64_t queueTailOffset = queueHeadOffset + kI32Bytes;
  int64_t completedCountOffset = queueTailOffset + kI32Bytes;
  int64_t depCountersOffset = completedCountOffset + kI32Bytes;
  int64_t queueStorageOffset =
      depCountersOffset + (*numInstances * kI32Bytes);

  OpBuilder builder(scheduler);
  builder.setInsertionPointAfter(scheduler);
  OperationState state(scheduler.getLoc(),
                       TaskGraphRuntimeStateOp::getOperationName());
  state.addAttribute("num_instances",
                     builder.getI64IntegerAttr(*numInstances));
  state.addAttribute("queue_capacity",
                     builder.getI64IntegerAttr(*queueCapacity));
  state.addAttribute("state_size_bytes",
                     builder.getI64IntegerAttr(*stateSize));
  state.addAttribute("alignment_bytes",
                     builder.getI64IntegerAttr(kRuntimeStateAlignment));
  state.addAttribute("counter_type", counterType);
  state.addAttribute("counter_bytes", builder.getI64IntegerAttr(kI32Bytes));
  state.addAttribute("queue_element_bytes",
                     builder.getI64IntegerAttr(kI32Bytes));
  state.addAttribute("init_flag_offset_bytes",
                     builder.getI64IntegerAttr(initFlagOffset));
  state.addAttribute("queue_lock_offset_bytes",
                     builder.getI64IntegerAttr(queueLockOffset));
  state.addAttribute("queue_head_offset_bytes",
                     builder.getI64IntegerAttr(queueHeadOffset));
  state.addAttribute("queue_tail_offset_bytes",
                     builder.getI64IntegerAttr(queueTailOffset));
  state.addAttribute("completed_count_offset_bytes",
                     builder.getI64IntegerAttr(completedCountOffset));
  state.addAttribute("dep_counters_offset_bytes",
                     builder.getI64IntegerAttr(depCountersOffset));
  state.addAttribute("queue_storage_offset_bytes",
                     builder.getI64IntegerAttr(queueStorageOffset));
  builder.create(state);

  auto module = func->getParentOfType<ModuleOp>();
  if (!module)
    return func->emitOpError()
           << "runtime state materialization requires parent module";
  module->setAttr("tle.requires_cooperative_grid",
                  builder.getI32IntegerAttr(1));
  return success();
}

class TritonTleMaterializeTaskRuntimeStatePass
    : public impl::TritonTleMaterializeTaskRuntimeStateBase<
          TritonTleMaterializeTaskRuntimeStatePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool sawError = false;

    module.walk([&](FunctionOpInterface func) {
      SmallVector<TaskGraphSchedulerOp> schedulers;
      SmallVector<TaskGraphRuntimeStateOp> runtimeStates;
      bool hasTaskDeclare = false;
      func->walk([&](Operation *op) {
        if (auto scheduler = dyn_cast<TaskGraphSchedulerOp>(op))
          schedulers.push_back(scheduler);
        if (auto runtimeState = dyn_cast<TaskGraphRuntimeStateOp>(op))
          runtimeStates.push_back(runtimeState);
        if (isa<TaskDeclareOp>(op))
          hasTaskDeclare = true;
      });

      if (runtimeStates.size() > 1) {
        func->emitOpError()
            << "contains multiple task graph runtime states";
        sawError = true;
        return;
      }
      if (!runtimeStates.empty() && !schedulers.empty()) {
        func->emitOpError()
            << "already contains a materialized task runtime state";
        sawError = true;
        return;
      }
      if (schedulers.size() > 1) {
        func->emitOpError()
            << "contains multiple task graph schedulers";
        sawError = true;
        return;
      }
      if (schedulers.empty()) {
        if (hasTaskDeclare) {
          func->emitOpError()
              << "requires tle.task_graph.scheduler before runtime state "
                 "materialization; run triton-tle-materialize-task-scheduler "
                 "first";
          sawError = true;
        }
        return;
      }
      if (failed(materializeRuntimeState(func, schedulers.front())))
        sawError = true;
    });

    if (sawError)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
