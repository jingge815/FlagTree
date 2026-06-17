// flagtree tle
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEANALYZETASKGRAPH
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kMaxMaterializedGraphItems = 1000000;

struct GridDefinition {
  TaskGridCreateOp create;
  std::string name;
  SmallVector<int64_t> shape;
};

struct TaskDefinition {
  TaskDeclareOp op;
  std::string name;
  SmallVector<int64_t> domain;
};

struct InstanceInfo {
  std::string name;
  std::string taskName;
  SmallVector<int64_t> coord;
  SmallVector<std::string> writes;
  std::set<std::string> deps;
};

static SmallVector<int64_t> getShape(Operation *op) {
  SmallVector<int64_t> values;
  auto shape = op->getAttrOfType<DenseI64ArrayAttr>("shape");
  if (!shape)
    return values;
  values.append(shape.asArrayRef().begin(), shape.asArrayRef().end());
  return values;
}

static SmallVector<int64_t> getDomain(TaskDeclareOp task) {
  SmallVector<int64_t> values;
  values.append(task.getDomainShape().begin(), task.getDomainShape().end());
  return values;
}

static SmallVector<int64_t> getWildcardDims(DictionaryAttr entry) {
  SmallVector<int64_t> values;
  auto wildcardDims =
      dyn_cast_or_null<DenseI64ArrayAttr>(entry.get("wildcard_dims"));
  if (!wildcardDims)
    return values;
  values.append(wildcardDims.asArrayRef().begin(),
                wildcardDims.asArrayRef().end());
  llvm::sort(values);
  return values;
}

static std::string formatCoord(ArrayRef<int64_t> coord) {
  std::ostringstream os;
  os << "[";
  for (auto [idx, value] : llvm::enumerate(coord)) {
    if (idx)
      os << ",";
    os << value;
  }
  os << "]";
  return os.str();
}

static std::string formatEntity(StringRef name, ArrayRef<int64_t> coord) {
  return (name + formatCoord(coord)).str();
}

static FailureOr<int64_t> checkedProduct(Operation *op, ArrayRef<int64_t> shape,
                                         StringRef what) {
  int64_t product = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      op->emitOpError() << "expects positive " << what << " shape";
      return failure();
    }
    if (product > std::numeric_limits<int32_t>::max() / dim) {
      op->emitOpError()
          << what << " cardinality exceeds signed 32-bit range";
      return failure();
    }
    product *= dim;
  }
  return product;
}

static void enumerateCoords(ArrayRef<int64_t> shape,
                            function_ref<void(ArrayRef<int64_t>)> callback) {
  SmallVector<int64_t> coord(shape.size(), 0);
  while (true) {
    callback(coord);
    int64_t axis = static_cast<int64_t>(shape.size()) - 1;
    while (axis >= 0) {
      coord[axis] += 1;
      if (coord[axis] < shape[axis])
        break;
      coord[axis] = 0;
      axis -= 1;
    }
    if (axis < 0)
      break;
  }
}

static bool contains(ArrayRef<int64_t> values, int64_t value) {
  return llvm::is_contained(values, value);
}

static FailureOr<SmallVector<SmallVector<int64_t>>>
expandMapTiles(TaskDeclareOp task, DictionaryAttr entry,
               ArrayRef<int64_t> taskCoord, const GridDefinition &grid) {
  AffineMap map = cast<AffineMapAttr>(entry.get("map")).getValue();
  SmallVector<int64_t> wildcardDims = getWildcardDims(entry);
  SmallVector<int64_t> base(grid.shape.size(), 0);

  SmallVector<int64_t> nonWildcardDims;
  for (int64_t dim = 0, e = grid.shape.size(); dim < e; ++dim) {
    if (!contains(wildcardDims, dim))
      nonWildcardDims.push_back(dim);
  }

  if (nonWildcardDims.size() != map.getNumResults()) {
    task.emitOpError()
        << "task map target rank does not match referenced task_grid rank";
    return failure();
  }

  for (auto [resultIdx, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      task.emitOpError()
          << "task graph analysis requires projection-only maps";
      return failure();
    }
    int64_t taskDim = dimExpr.getPosition();
    if (taskDim >= static_cast<int64_t>(taskCoord.size())) {
      task.emitOpError()
          << "task map references a dimension outside task domain";
      return failure();
    }
    int64_t gridDim = nonWildcardDims[resultIdx];
    base[gridDim] = taskCoord[taskDim];
    if (base[gridDim] >= grid.shape[gridDim]) {
      task.emitOpError()
          << "task map produces a grid tile outside task_grid shape";
      return failure();
    }
  }

  SmallVector<int64_t> wildcardShape;
  for (int64_t wildcardDim : wildcardDims) {
    if (wildcardDim >= static_cast<int64_t>(grid.shape.size())) {
      task.emitOpError()
          << "task wildcard dimension is outside referenced grid rank";
      return failure();
    }
    wildcardShape.push_back(grid.shape[wildcardDim]);
  }

  FailureOr<int64_t> expansion =
      checkedProduct(task.getOperation(), wildcardShape, "task read expansion");
  if (failed(expansion))
    return failure();
  if (*expansion > kMaxMaterializedGraphItems) {
    task.emitOpError()
        << "task graph debug materialization exceeds current explicit "
           "analysis limit";
    return failure();
  }

  SmallVector<SmallVector<int64_t>> tiles;
  if (wildcardDims.empty()) {
    tiles.push_back(base);
    return tiles;
  }

  enumerateCoords(wildcardShape, [&](ArrayRef<int64_t> wildcardCoord) {
    SmallVector<int64_t> tile = base;
    for (auto [idx, wildcardDim] : llvm::enumerate(wildcardDims))
      tile[wildcardDim] = wildcardCoord[idx];
    tiles.push_back(tile);
  });
  return tiles;
}

static ArrayAttr makeStringArray(OpBuilder &builder,
                                 ArrayRef<std::string> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (StringRef value : values)
    attrs.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attrs);
}

static DictionaryAttr makeInstanceAttr(OpBuilder &builder,
                                       const InstanceInfo &instance) {
  SmallVector<std::string> deps(instance.deps.begin(), instance.deps.end());
  return builder.getDictionaryAttr({
      builder.getNamedAttr("task", builder.getStringAttr(instance.taskName)),
      builder.getNamedAttr("instance", builder.getStringAttr(instance.name)),
      builder.getNamedAttr("coord",
                           builder.getDenseI64ArrayAttr(instance.coord)),
      builder.getNamedAttr("dep_count",
                           builder.getI64IntegerAttr(instance.deps.size())),
      builder.getNamedAttr("deps", makeStringArray(builder, deps)),
      builder.getNamedAttr("writes", makeStringArray(builder, instance.writes)),
  });
}

static DictionaryAttr makeEdgeAttr(OpBuilder &builder, StringRef tile,
                                   StringRef producer, StringRef consumer) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr("tile", builder.getStringAttr(tile)),
      builder.getNamedAttr("producer", builder.getStringAttr(producer)),
      builder.getNamedAttr("consumer", builder.getStringAttr(consumer)),
  });
}

static LogicalResult detectTaskCycles(
    FunctionOpInterface func,
    const std::map<std::string, std::set<std::string>> &taskEdges) {
  enum class VisitState { Unvisited, Visiting, Done };
  std::map<std::string, VisitState> states;
  for (auto &[task, successors] : taskEdges) {
    states.try_emplace(task, VisitState::Unvisited);
    for (const std::string &successor : successors)
      states.try_emplace(successor, VisitState::Unvisited);
  }

  std::function<LogicalResult(const std::string &)> dfs =
      [&](const std::string &task) -> LogicalResult {
    VisitState state = states[task];
    if (state == VisitState::Visiting)
      return func->emitOpError()
             << "task graph contains a dependency cycle involving " << task;
    if (state == VisitState::Done)
      return success();
    states[task] = VisitState::Visiting;
    auto it = taskEdges.find(task);
    if (it != taskEdges.end()) {
      for (const std::string &successor : it->second) {
        if (failed(dfs(successor)))
          return failure();
      }
    }
    states[task] = VisitState::Done;
    return success();
  };

  for (auto &[task, state] : states) {
    if (state == VisitState::Unvisited && failed(dfs(task)))
      return failure();
  }
  return success();
}

static LogicalResult analyzeFunctionTaskGraph(FunctionOpInterface func) {
  OpBuilder builder(func.getContext());
  std::map<std::string, GridDefinition> grids;
  SmallVector<TaskDefinition> tasks;
  std::set<std::string> taskNames;
  bool sawError = false;

  func->walk([&](TaskGridCreateOp create) {
    std::string gridName = create.getGridName().str();
    SmallVector<int64_t> shape = getShape(create.getOperation());
    if (shape.empty()) {
      create.emitOpError()
          << "task graph analysis requires static task_grid shape";
      sawError = true;
      return;
    }
    auto [it, inserted] =
        grids.try_emplace(gridName, GridDefinition{create, gridName, shape});
    if (!inserted) {
      create.emitOpError()
          << "duplicates task_grid name in task graph metadata";
      sawError = true;
    }
  });

  func->walk([&](TaskDeclareOp task) {
    std::string taskName = task.getTaskName().str();
    if (!taskNames.insert(taskName).second) {
      task.emitOpError() << "duplicates task name in task graph metadata";
      sawError = true;
      return;
    }
    tasks.push_back(TaskDefinition{task, taskName, getDomain(task)});
  });

  if (tasks.empty())
    return success();
  if (sawError)
    return failure();

  int64_t totalInstances = 0;
  for (TaskDefinition &task : tasks) {
    FailureOr<int64_t> count =
        checkedProduct(task.op.getOperation(), task.domain, "task instance");
    if (failed(count))
      return failure();
    if (totalInstances > std::numeric_limits<int32_t>::max() - *count)
      return task.op.emitOpError()
             << "task graph instance count exceeds signed 32-bit range";
    totalInstances += *count;
  }
  if (totalInstances > kMaxMaterializedGraphItems)
    return func->emitOpError()
           << "task graph debug materialization exceeds current explicit "
              "analysis limit";

  std::map<std::string, std::string> producerByTile;
  std::map<std::string, std::string> producerTaskByTile;
  SmallVector<InstanceInfo> instances;

  for (TaskDefinition &task : tasks) {
    enumerateCoords(task.domain, [&](ArrayRef<int64_t> coord) {
      InstanceInfo instance;
      instance.name = formatEntity(task.name, coord);
      instance.taskName = task.name;
      instance.coord.append(coord.begin(), coord.end());
      for (Attribute entryAttr : task.op.getWrites()) {
        auto entry = cast<DictionaryAttr>(entryAttr);
        std::string gridName = cast<StringAttr>(entry.get("grid")).getValue().str();
        auto grid = grids.find(gridName);
        if (grid == grids.end()) {
          task.op.emitOpError()
              << "task write map references unknown task_grid " << gridName;
          sawError = true;
          continue;
        }
        FailureOr<SmallVector<SmallVector<int64_t>>> tiles =
            expandMapTiles(task.op, entry, coord, grid->second);
        if (failed(tiles)) {
          sawError = true;
          return;
        }
        for (ArrayRef<int64_t> tileCoord : *tiles) {
          std::string tileName = formatEntity(gridName, tileCoord);
          if (producerByTile.count(tileName)) {
            task.op.emitOpError()
                << "task graph has multiple producers for " << tileName;
            sawError = true;
            continue;
          }
          producerByTile[tileName] = instance.name;
          producerTaskByTile[tileName] = task.name;
          instance.writes.push_back(tileName);
        }
      }
      instances.push_back(std::move(instance));
    });
  }

  if (sawError)
    return failure();

  std::map<std::string, std::set<std::string>> consumersByTile;
  std::map<std::string, std::set<std::string>> taskEdges;
  for (InstanceInfo &instance : instances) {
    TaskDeclareOp op;
    for (const TaskDefinition &task : tasks) {
      if (task.name == instance.taskName) {
        op = task.op;
        break;
      }
    }
    for (Attribute entryAttr : op.getReads()) {
      auto entry = cast<DictionaryAttr>(entryAttr);
      std::string gridName = cast<StringAttr>(entry.get("grid")).getValue().str();
      auto grid = grids.find(gridName);
      if (grid == grids.end()) {
        op.emitOpError()
            << "task read map references unknown task_grid " << gridName;
        sawError = true;
        continue;
      }
      FailureOr<SmallVector<SmallVector<int64_t>>> tiles =
          expandMapTiles(op, entry, instance.coord, grid->second);
      if (failed(tiles)) {
        sawError = true;
        continue;
      }
      for (ArrayRef<int64_t> tileCoord : *tiles) {
        std::string tileName = formatEntity(gridName, tileCoord);
        auto producer = producerByTile.find(tileName);
        if (producer == producerByTile.end()) {
          op.emitOpError()
              << "task graph read requires unproduced tile " << tileName;
          sawError = true;
          continue;
        }
        instance.deps.insert(tileName);
        consumersByTile[tileName].insert(instance.name);
        taskEdges[producerTaskByTile[tileName]].insert(instance.taskName);
      }
    }
  }

  if (sawError)
    return failure();

  if (failed(detectTaskCycles(func, taskEdges)))
    return failure();

  SmallVector<Attribute> instanceAttrs;
  SmallVector<Attribute> edgeAttrs;
  SmallVector<std::string> initialReady;
  int64_t totalEdges = 0;
  for (const InstanceInfo &instance : instances) {
    if (instance.deps.empty())
      initialReady.push_back(instance.name);
    instanceAttrs.push_back(makeInstanceAttr(builder, instance));
  }
  for (auto &[tile, consumers] : consumersByTile) {
    for (const std::string &consumer : consumers) {
      if (totalEdges == std::numeric_limits<int32_t>::max())
        return func->emitOpError()
               << "task graph edge count exceeds signed 32-bit range";
      totalEdges += 1;
      edgeAttrs.push_back(makeEdgeAttr(builder, tile, producerByTile[tile],
                                       consumer));
    }
  }
  if (totalEdges > kMaxMaterializedGraphItems)
    return func->emitOpError()
           << "task graph debug materialization exceeds current explicit "
              "analysis limit";

  DictionaryAttr analysis = builder.getDictionaryAttr({
      builder.getNamedAttr("analysis_version", builder.getI64IntegerAttr(2)),
      builder.getNamedAttr("instances", builder.getArrayAttr(instanceAttrs)),
      builder.getNamedAttr("edges", builder.getArrayAttr(edgeAttrs)),
      builder.getNamedAttr("initial_ready",
                           makeStringArray(builder, initialReady)),
      builder.getNamedAttr("num_instances",
                           builder.getI64IntegerAttr(totalInstances)),
      builder.getNamedAttr("num_edges", builder.getI64IntegerAttr(totalEdges)),
  });
  func->setAttr("tle.task_graph.analysis", analysis);
  return success();
}

class TritonTleAnalyzeTaskGraphPass
    : public impl::TritonTleAnalyzeTaskGraphBase<
          TritonTleAnalyzeTaskGraphPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool sawError = false;
    module.walk([&](FunctionOpInterface func) {
      if (failed(analyzeFunctionTaskGraph(func)))
        sawError = true;
    });
    if (sawError)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
