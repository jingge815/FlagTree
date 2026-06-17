// flagtree tle
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <map>
#include <set>
#include <string>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEMATERIALIZETASKSCHEDULER
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static FailureOr<IntegerAttr> requireIntegerAttr(Operation *op,
                                                 DictionaryAttr dict,
                                                 StringRef attrName) {
  auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(attrName));
  if (!attr)
    return op->emitOpError("requires task graph analysis attribute ")
           << attrName;
  return attr;
}

static FailureOr<ArrayAttr> requireArrayAttr(Operation *op,
                                             DictionaryAttr dict,
                                             StringRef attrName) {
  auto attr = dyn_cast_or_null<ArrayAttr>(dict.get(attrName));
  if (!attr)
    return op->emitOpError("requires task graph analysis attribute ")
           << attrName;
  return attr;
}

static FailureOr<int32_t> checkedI32(Operation *op, int64_t value,
                                     StringRef what) {
  if (value < 0 || value > std::numeric_limits<int32_t>::max())
    return op->emitOpError() << what << " exceeds scheduler i32 range";
  return static_cast<int32_t>(value);
}

static FailureOr<DictionaryAttr>
requireDictionaryEntry(Operation *op, Attribute attr, StringRef attrName) {
  auto dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict)
    return op->emitOpError()
           << "expects task graph analysis " << attrName
           << " entries to be dictionaries";
  return dict;
}

static FailureOr<StringAttr> requireStringEntry(Operation *op,
                                                DictionaryAttr dict,
                                                StringRef dictName,
                                                StringRef key) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(key));
  if (!attr || attr.getValue().empty())
    return op->emitOpError()
           << "expects task graph analysis " << dictName << " entry " << key
           << " to be a non-empty string";
  return attr;
}

static FailureOr<IntegerAttr> requireIntegerEntry(Operation *op,
                                                  DictionaryAttr dict,
                                                  StringRef dictName,
                                                  StringRef key) {
  auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(key));
  if (!attr)
    return op->emitOpError()
           << "expects task graph analysis " << dictName << " entry " << key
           << " to be an integer";
  return attr;
}

static FailureOr<DenseI64ArrayAttr>
requireDenseI64ArrayEntry(Operation *op, DictionaryAttr dict,
                          StringRef dictName, StringRef key) {
  auto attr = dyn_cast_or_null<DenseI64ArrayAttr>(dict.get(key));
  if (!attr)
    return op->emitOpError()
           << "expects task graph analysis " << dictName << " entry " << key
           << " to be an i64 dense array";
  return attr;
}

static LogicalResult materializeFunctionScheduler(FunctionOpInterface func) {
  SmallVector<TaskDeclareOp> tasks;
  bool hasScheduler = false;
  func->walk([&](Operation *op) {
    if (auto task = dyn_cast<TaskDeclareOp>(op))
      tasks.push_back(task);
    if (isa<TaskGraphSchedulerOp>(op))
      hasScheduler = true;
  });

  if (tasks.empty())
    return success();
  if (hasScheduler)
    return func->emitOpError()
           << "already contains a materialized task graph scheduler";

  auto analysis =
      func->getAttrOfType<DictionaryAttr>("tle.task_graph.analysis");
  if (!analysis)
    return func->emitOpError()
           << "requires tle.task_graph.analysis before scheduler "
              "materialization; run triton-tle-analyze-task-graph first";

  FailureOr<IntegerAttr> analysisVersion =
      requireIntegerAttr(func.getOperation(), analysis, "analysis_version");
  if (failed(analysisVersion))
    return failure();
  if ((*analysisVersion).getInt() != 2)
    return func->emitOpError()
           << "requires task graph analysis_version = 2 for scheduler "
              "materialization";

  FailureOr<IntegerAttr> numInstances =
      requireIntegerAttr(func.getOperation(), analysis, "num_instances");
  FailureOr<IntegerAttr> numEdges =
      requireIntegerAttr(func.getOperation(), analysis, "num_edges");
  FailureOr<ArrayAttr> instances =
      requireArrayAttr(func.getOperation(), analysis, "instances");
  FailureOr<ArrayAttr> edges =
      requireArrayAttr(func.getOperation(), analysis, "edges");
  FailureOr<ArrayAttr> initialReady =
      requireArrayAttr(func.getOperation(), analysis, "initial_ready");
  if (failed(numInstances) || failed(numEdges) || failed(instances) ||
      failed(edges) || failed(initialReady))
    return failure();

  if ((*numInstances).getInt() <= 0)
    return func->emitOpError()
           << "scheduler materialization requires at least one task instance";
  if ((*numEdges).getInt() < 0)
    return func->emitOpError()
           << "scheduler materialization requires non-negative edge count";
  if ((*numInstances).getInt() != static_cast<int64_t>((*instances).size()))
    return func->emitOpError()
           << "task graph analysis num_instances does not match instances";
  if ((*numEdges).getInt() != static_cast<int64_t>((*edges).size()))
    return func->emitOpError()
           << "task graph analysis num_edges does not match edges";
  if ((*initialReady).empty())
    return func->emitOpError()
           << "scheduler materialization requires an initial ready queue";
  FailureOr<int32_t> numTasksI32 =
      checkedI32(func.getOperation(), tasks.size(), "task count");
  FailureOr<int32_t> numInstancesI32 = checkedI32(
      func.getOperation(), (*numInstances).getInt(), "task instance count");
  FailureOr<int32_t> numEdgesI32 =
      checkedI32(func.getOperation(), (*numEdges).getInt(), "task edge count");
  if (failed(numTasksI32) || failed(numInstancesI32) || failed(numEdgesI32))
    return failure();

  OpBuilder builder(func.getContext());
  builder.setInsertionPoint(tasks.front());

  std::set<std::string> seenTaskNames;
  std::map<std::string, int32_t> taskIdByName;
  SmallVector<int32_t> taskDomainRanks;
  SmallVector<Attribute> taskNameAttrs;
  SmallVector<Attribute> dispatchAttrs;
  taskNameAttrs.reserve(tasks.size());
  dispatchAttrs.reserve(tasks.size());
  for (auto [index, task] : llvm::enumerate(tasks)) {
    std::string taskName = task.getTaskName().str();
    if (!seenTaskNames.insert(taskName).second)
      return task.emitOpError()
             << "duplicates task name before scheduler materialization";
    FailureOr<int32_t> taskId =
        checkedI32(task, index, "task id");
    FailureOr<int32_t> taskRank =
        checkedI32(task, task.getDomainShape().size(), "task domain rank");
    if (failed(taskId) || failed(taskRank))
      return failure();
    taskIdByName[taskName] = *taskId;
    taskDomainRanks.push_back(*taskRank);
    auto taskNameAttr = builder.getStringAttr(taskName);
    taskNameAttrs.push_back(taskNameAttr);
    SmallVector<NamedAttribute> dispatchEntry{
        builder.getNamedAttr("task", taskNameAttr),
        builder.getNamedAttr("task_id",
                             builder.getI64IntegerAttr(index)),
    };
    if (auto callee =
            task->getAttrOfType<FlatSymbolRefAttr>("callee")) {
      dispatchEntry.push_back(builder.getNamedAttr("callee", callee));
    }
    dispatchAttrs.push_back(builder.getDictionaryAttr(dispatchEntry));
  }

  std::map<std::string, int32_t> instanceIdByName;
  SmallVector<int32_t> instanceTaskIds;
  SmallVector<int32_t> instanceDepCounts;
  SmallVector<int32_t> instanceCoordOffsets;
  SmallVector<int32_t> instanceCoords;
  instanceTaskIds.reserve(*numInstancesI32);
  instanceDepCounts.reserve(*numInstancesI32);
  instanceCoordOffsets.reserve(*numInstancesI32 + 1);
  instanceCoordOffsets.push_back(0);

  for (auto [index, attr] : llvm::enumerate(*instances)) {
    FailureOr<DictionaryAttr> dict =
        requireDictionaryEntry(func.getOperation(), attr, "instances");
    if (failed(dict))
      return failure();
    FailureOr<StringAttr> task =
        requireStringEntry(func.getOperation(), *dict, "instances", "task");
    FailureOr<StringAttr> instance =
        requireStringEntry(func.getOperation(), *dict, "instances", "instance");
    FailureOr<IntegerAttr> depCount = requireIntegerEntry(
        func.getOperation(), *dict, "instances", "dep_count");
    FailureOr<DenseI64ArrayAttr> coord =
        requireDenseI64ArrayEntry(func.getOperation(), *dict, "instances",
                                  "coord");
    if (failed(task) || failed(instance) || failed(depCount) ||
        failed(coord))
      return failure();

    auto taskIdIt = taskIdByName.find((*task).getValue().str());
    if (taskIdIt == taskIdByName.end())
      return func->emitOpError()
             << "task graph analysis instance references unknown task "
             << (*task).getValue();
    int32_t taskId = taskIdIt->second;
    if ((*coord).size() != static_cast<size_t>(taskDomainRanks[taskId]))
      return func->emitOpError()
             << "task graph analysis instance coord rank does not match task "
                "domain rank";

    FailureOr<int32_t> instanceId =
        checkedI32(func.getOperation(), index, "task instance id");
    FailureOr<int32_t> depCountI32 =
        checkedI32(func.getOperation(), (*depCount).getInt(),
                   "task dependency count");
    if (failed(instanceId) || failed(depCountI32))
      return failure();
    if (!instanceIdByName.emplace((*instance).getValue().str(), *instanceId)
             .second)
      return func->emitOpError()
             << "task graph analysis contains duplicate task instance "
             << (*instance).getValue();
    instanceTaskIds.push_back(taskId);
    instanceDepCounts.push_back(*depCountI32);
    for (int64_t coordValue : (*coord).asArrayRef()) {
      FailureOr<int32_t> coordI32 =
          checkedI32(func.getOperation(), coordValue, "task instance coord");
      if (failed(coordI32))
        return failure();
      instanceCoords.push_back(*coordI32);
    }
    FailureOr<int32_t> coordOffset =
        checkedI32(func.getOperation(), instanceCoords.size(),
                   "task instance coord offset");
    if (failed(coordOffset))
      return failure();
    instanceCoordOffsets.push_back(*coordOffset);
  }

  SmallVector<SmallVector<int32_t>> consumersByProducer(*numInstancesI32);
  for (Attribute attr : *edges) {
    FailureOr<DictionaryAttr> dict =
        requireDictionaryEntry(func.getOperation(), attr, "edges");
    if (failed(dict))
      return failure();
    FailureOr<StringAttr> producer =
        requireStringEntry(func.getOperation(), *dict, "edges", "producer");
    FailureOr<StringAttr> consumer =
        requireStringEntry(func.getOperation(), *dict, "edges", "consumer");
    if (failed(producer) || failed(consumer))
      return failure();
    auto producerIt = instanceIdByName.find((*producer).getValue().str());
    if (producerIt == instanceIdByName.end())
      return func->emitOpError()
             << "task graph analysis edge references unknown producer "
             << (*producer).getValue();
    auto consumerIt = instanceIdByName.find((*consumer).getValue().str());
    if (consumerIt == instanceIdByName.end())
      return func->emitOpError()
             << "task graph analysis edge references unknown consumer "
             << (*consumer).getValue();
    consumersByProducer[producerIt->second].push_back(consumerIt->second);
  }

  SmallVector<int32_t> producerEdgeOffsets;
  SmallVector<int32_t> edgeConsumerIds;
  producerEdgeOffsets.reserve(*numInstancesI32 + 1);
  edgeConsumerIds.reserve(*numEdgesI32);
  producerEdgeOffsets.push_back(0);
  for (ArrayRef<int32_t> consumers : consumersByProducer) {
    edgeConsumerIds.append(consumers.begin(), consumers.end());
    FailureOr<int32_t> offset =
        checkedI32(func.getOperation(), edgeConsumerIds.size(),
                   "producer edge offset");
    if (failed(offset))
      return failure();
    producerEdgeOffsets.push_back(*offset);
  }
  if (edgeConsumerIds.size() != static_cast<size_t>(*numEdgesI32))
    return func->emitOpError()
           << "task graph analysis edge table size does not match num_edges";

  SmallVector<int32_t> initialReadyIds;
  initialReadyIds.reserve((*initialReady).size());
  for (Attribute attr : *initialReady) {
    auto ready = dyn_cast<StringAttr>(attr);
    if (!ready || ready.getValue().empty())
      return func->emitOpError()
             << "task graph analysis initial_ready entries must be strings";
    auto instanceIt = instanceIdByName.find(ready.getValue().str());
    if (instanceIt == instanceIdByName.end())
      return func->emitOpError()
             << "task graph analysis initial_ready references unknown task "
                "instance "
             << ready.getValue();
    if (instanceDepCounts[instanceIt->second] != 0)
      return func->emitOpError()
             << "task graph analysis initial_ready references non-ready task "
                "instance "
             << ready.getValue();
    initialReadyIds.push_back(instanceIt->second);
  }

  OperationState state(tasks.front().getLoc(),
                       TaskGraphSchedulerOp::getOperationName());
  state.addAttribute("num_tasks", builder.getI64IntegerAttr(tasks.size()));
  state.addAttribute("num_instances", *numInstances);
  state.addAttribute("num_edges", *numEdges);
  state.addAttribute("queue_capacity",
                     builder.getI64IntegerAttr((*numInstances).getInt()));
  state.addAttribute("counter_type", builder.getStringAttr("i32"));
  state.addAttribute("task_names", builder.getArrayAttr(taskNameAttrs));
  state.addAttribute("dispatch", builder.getArrayAttr(dispatchAttrs));
  state.addAttribute("instances", *instances);
  state.addAttribute("edges", *edges);
  state.addAttribute("initial_ready", *initialReady);
  state.addAttribute("task_domain_ranks",
                     builder.getDenseI32ArrayAttr(taskDomainRanks));
  state.addAttribute("instance_task_ids",
                     builder.getDenseI32ArrayAttr(instanceTaskIds));
  state.addAttribute("instance_dep_counts",
                     builder.getDenseI32ArrayAttr(instanceDepCounts));
  state.addAttribute("instance_coord_offsets",
                     builder.getDenseI32ArrayAttr(instanceCoordOffsets));
  state.addAttribute("instance_coords",
                     builder.getDenseI32ArrayAttr(instanceCoords));
  state.addAttribute("initial_ready_ids",
                     builder.getDenseI32ArrayAttr(initialReadyIds));
  state.addAttribute("producer_edge_offsets",
                     builder.getDenseI32ArrayAttr(producerEdgeOffsets));
  state.addAttribute("edge_consumer_ids",
                     builder.getDenseI32ArrayAttr(edgeConsumerIds));
  builder.create(state);

  func->removeAttr("tle.task_graph.analysis");
  return success();
}

class TritonTleMaterializeTaskSchedulerPass
    : public impl::TritonTleMaterializeTaskSchedulerBase<
          TritonTleMaterializeTaskSchedulerPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool sawError = false;
    module.walk([&](FunctionOpInterface func) {
      if (failed(materializeFunctionScheduler(func)))
        sawError = true;
    });
    if (sawError)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
