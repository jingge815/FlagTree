// flagtree tle
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>
#include <limits>
#include <map>
#include <string>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWERTASKSCHEDULER
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static FailureOr<DenseI32ArrayAttr>
requireI32Array(TaskGraphSchedulerOp scheduler, StringRef attrName) {
  auto attr = scheduler->getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!attr)
    return scheduler.emitOpError("requires ") << attrName << " attribute";
  return attr;
}

static FailureOr<ArrayAttr> requireArray(TaskGraphSchedulerOp scheduler,
                                         StringRef attrName) {
  auto attr = scheduler->getAttrOfType<ArrayAttr>(attrName);
  if (!attr)
    return scheduler.emitOpError("requires ") << attrName << " attribute";
  return attr;
}

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

static FailureOr<SmallVector<FlatSymbolRefAttr>>
collectTaskCallees(TaskGraphSchedulerOp scheduler) {
  FailureOr<ArrayAttr> dispatch = requireArray(scheduler, "dispatch");
  FailureOr<int64_t> numTasks =
      requireI64(scheduler, "num_tasks", /*requirePositive=*/true);
  if (failed(dispatch) || failed(numTasks))
    return failure();
  if (static_cast<int64_t>((*dispatch).size()) != *numTasks)
    return scheduler.emitOpError()
           << "expects dispatch size to match num_tasks";

  SmallVector<FlatSymbolRefAttr> callees(*numTasks);
  for (Attribute attr : *dispatch) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return scheduler.emitOpError()
             << "expects dispatch entries to be dictionaries";
    auto taskId = dyn_cast_or_null<IntegerAttr>(dict.get("task_id"));
    if (!taskId || taskId.getInt() < 0 || taskId.getInt() >= *numTasks)
      return scheduler.emitOpError()
             << "expects dispatch task_id to be in task range";
    auto callee = dyn_cast_or_null<FlatSymbolRefAttr>(dict.get("callee"));
    if (!callee)
      return scheduler.emitOpError()
             << "requires dispatch callee for task scheduler lowering";
    callees[taskId.getInt()] = callee;
  }
  return callees;
}

static LogicalResult lowerSchedulerInFunction(FunctionOpInterface func,
                                              TaskGraphSchedulerOp scheduler) {
  FailureOr<int64_t> numInstances =
      requireI64(scheduler, "num_instances", /*requirePositive=*/true);
  if (failed(numInstances))
    return failure();
  if (*numInstances > std::numeric_limits<int32_t>::max())
    return scheduler.emitOpError()
           << "num_instances exceeds scheduler i32 range";

  FailureOr<SmallVector<FlatSymbolRefAttr>> callees =
      collectTaskCallees(scheduler);
  FailureOr<DenseI32ArrayAttr> instanceTaskIds =
      requireI32Array(scheduler, "instance_task_ids");
  FailureOr<DenseI32ArrayAttr> instanceDepCounts =
      requireI32Array(scheduler, "instance_dep_counts");
  FailureOr<DenseI32ArrayAttr> instanceCoordOffsets =
      requireI32Array(scheduler, "instance_coord_offsets");
  FailureOr<DenseI32ArrayAttr> instanceCoords =
      requireI32Array(scheduler, "instance_coords");
  FailureOr<DenseI32ArrayAttr> initialReadyIds =
      requireI32Array(scheduler, "initial_ready_ids");
  FailureOr<DenseI32ArrayAttr> producerEdgeOffsets =
      requireI32Array(scheduler, "producer_edge_offsets");
  FailureOr<DenseI32ArrayAttr> edgeConsumerIds =
      requireI32Array(scheduler, "edge_consumer_ids");
  if (failed(callees) || failed(instanceTaskIds) ||
      failed(instanceDepCounts) || failed(instanceCoordOffsets) ||
      failed(instanceCoords) || failed(initialReadyIds) ||
      failed(producerEdgeOffsets) || failed(edgeConsumerIds))
    return failure();

  ArrayRef<int32_t> taskIds = (*instanceTaskIds).asArrayRef();
  ArrayRef<int32_t> depCountsAttr = (*instanceDepCounts).asArrayRef();
  ArrayRef<int32_t> coordOffsets = (*instanceCoordOffsets).asArrayRef();
  ArrayRef<int32_t> coords = (*instanceCoords).asArrayRef();
  ArrayRef<int32_t> readyIds = (*initialReadyIds).asArrayRef();
  ArrayRef<int32_t> edgeOffsets = (*producerEdgeOffsets).asArrayRef();
  ArrayRef<int32_t> consumers = (*edgeConsumerIds).asArrayRef();

  if (taskIds.size() != static_cast<size_t>(*numInstances) ||
      depCountsAttr.size() != static_cast<size_t>(*numInstances) ||
      coordOffsets.size() != static_cast<size_t>(*numInstances + 1) ||
      edgeOffsets.size() != static_cast<size_t>(*numInstances + 1))
    return scheduler.emitOpError()
           << "has inconsistent runtime table sizes";
  if (readyIds.size() != 1)
    return scheduler.emitOpError()
           << "restricted task scheduler lowering requires exactly one "
              "initial ready task";

  SmallVector<int32_t> remainingDeps(depCountsAttr.begin(),
                                     depCountsAttr.end());
  std::deque<int32_t> ready;
  ready.push_back(readyIds.front());
  SmallVector<int32_t> schedule;
  schedule.reserve(*numInstances);

  while (!ready.empty()) {
    if (ready.size() != 1)
      return scheduler.emitOpError()
             << "restricted task scheduler lowering refuses to serialize a "
                "parallel ready set";
    int32_t instanceId = ready.front();
    ready.pop_front();
    if (instanceId < 0 || instanceId >= *numInstances)
      return scheduler.emitOpError()
             << "ready task id is outside task instance range";
    schedule.push_back(instanceId);

    int32_t begin = edgeOffsets[instanceId];
    int32_t end = edgeOffsets[instanceId + 1];
    if (begin < 0 || end < begin ||
        end > static_cast<int32_t>(consumers.size()))
      return scheduler.emitOpError()
             << "producer edge offsets are inconsistent";
    for (int32_t edge = begin; edge < end; ++edge) {
      int32_t consumer = consumers[edge];
      if (consumer < 0 || consumer >= *numInstances)
        return scheduler.emitOpError()
               << "edge consumer id is outside task instance range";
      if (remainingDeps[consumer] <= 0)
        return scheduler.emitOpError()
               << "edge consumes an already-ready task instance";
      remainingDeps[consumer] -= 1;
      if (remainingDeps[consumer] == 0)
        ready.push_back(consumer);
    }
  }

  if (schedule.size() != static_cast<size_t>(*numInstances))
    return scheduler.emitOpError()
           << "restricted task scheduler lowering could not cover all task "
              "instances";

  OpBuilder builder(scheduler);
  Location loc = scheduler.getLoc();
  SmallVector<Value> graphArgs(func.getArguments().begin(),
                               func.getArguments().end());
  for (int32_t instanceId : schedule) {
    int32_t taskId = taskIds[instanceId];
    if (taskId < 0 || taskId >= static_cast<int32_t>((*callees).size()))
      return scheduler.emitOpError()
             << "task id is outside dispatch range";
    int32_t coordBegin = coordOffsets[instanceId];
    int32_t coordEnd = coordOffsets[instanceId + 1];
    if (coordBegin < 0 || coordEnd < coordBegin ||
        coordEnd > static_cast<int32_t>(coords.size()))
      return scheduler.emitOpError()
             << "instance coordinate offsets are inconsistent";

    SmallVector<Value> operands;
    operands.reserve((coordEnd - coordBegin) + graphArgs.size());
    for (int32_t idx = coordBegin; idx < coordEnd; ++idx) {
      operands.push_back(arith::ConstantIntOp::create(builder, loc, coords[idx],
                                                      32));
    }
    operands.append(graphArgs.begin(), graphArgs.end());
    mlir::triton::CallOp::create(builder, loc, (*callees)[taskId],
                                 TypeRange(), operands);
  }

  SmallVector<Operation *> metadata;
  func->walk([&](TaskDeclareOp op) { metadata.push_back(op); });
  func->walk([&](TaskGridCreateOp op) { metadata.push_back(op); });
  scheduler.erase();
  for (Operation *op : metadata)
    op->erase();
  return success();
}

class TritonTleLowerTaskSchedulerPass
    : public impl::TritonTleLowerTaskSchedulerBase<
          TritonTleLowerTaskSchedulerPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool sawError = false;
    SmallVector<std::pair<FunctionOpInterface, TaskGraphSchedulerOp>>
        schedulers;
    module.walk([&](FunctionOpInterface func) {
      SmallVector<TaskGraphSchedulerOp> funcSchedulers;
      func->walk([&](TaskGraphSchedulerOp scheduler) {
        funcSchedulers.push_back(scheduler);
      });
      if (funcSchedulers.size() > 1) {
        func->emitOpError()
            << "contains multiple task graph schedulers";
        sawError = true;
        return;
      }
      if (funcSchedulers.size() == 1)
        schedulers.push_back({func, funcSchedulers.front()});
    });
    if (sawError)
      return signalPassFailure();

    for (auto [func, scheduler] : schedulers) {
      if (failed(lowerSchedulerInFunction(func, scheduler)))
        sawError = true;
    }
    if (sawError)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
