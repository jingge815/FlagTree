// flagtree tle
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallSet.h"
#include <map>
#include <set>
#include <string>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEVERIFYTASKGRAPH
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

struct GridDefinition {
  TaskGridCreateOp create;
  SmallVector<int64_t> shape;
};

static SmallVector<int64_t> getShape(Operation *op) {
  SmallVector<int64_t> values;
  auto shape = op->getAttrOfType<DenseI64ArrayAttr>("shape");
  if (!shape)
    return values;
  values.append(shape.asArrayRef().begin(), shape.asArrayRef().end());
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
  return values;
}

static LogicalResult verifyMapAgainstGrid(TaskDeclareOp task,
                                          DictionaryAttr entry,
                                          StringRef mapKind,
                                          const GridDefinition &grid) {
  if (grid.shape.empty())
    return task.emitOpError()
           << "references task_grid " << cast<StringAttr>(entry.get("grid"))
           << " without static shape";

  auto mapAttr = cast<AffineMapAttr>(entry.get("map"));
  AffineMap map = mapAttr.getValue();
  SmallVector<int64_t> wildcardDims = getWildcardDims(entry);
  int64_t gridRank = grid.shape.size();

  llvm::SmallSet<int64_t, 8> seenWildcardDims;
  for (int64_t wildcardDim : wildcardDims) {
    if (wildcardDim >= gridRank)
      return task.emitOpError()
             << "task " << mapKind
             << " wildcard dimension is outside referenced grid rank";
    if (!seenWildcardDims.insert(wildcardDim).second)
      return task.emitOpError()
             << "task " << mapKind << " wildcard dimensions must be unique";
  }

  int64_t targetRank = map.getNumResults() + wildcardDims.size();
  if (targetRank != gridRank)
    return task.emitOpError()
           << "task " << mapKind
           << " map target rank does not match referenced task_grid rank";
  return success();
}

static LogicalResult verifyTaskMapReferences(
    TaskDeclareOp task, ArrayAttr entries, StringRef mapKind,
    const std::map<std::string, GridDefinition> &grids) {
  for (Attribute entryAttr : entries) {
    auto entry = cast<DictionaryAttr>(entryAttr);
    StringRef gridName = cast<StringAttr>(entry.get("grid")).getValue();
    auto it = grids.find(gridName.str());
    if (it == grids.end())
      return task.emitOpError()
             << "task " << mapKind << " map references unknown task_grid "
             << gridName;
    if (failed(verifyMapAgainstGrid(task, entry, mapKind, it->second)))
      return failure();
  }
  return success();
}

static LogicalResult verifyTaskCalleeAbi(TaskDeclareOp task,
                                         FunctionOpInterface graphFunc,
                                         mlir::triton::FuncOp callee) {
  FunctionType graphType = cast<FunctionType>(graphFunc.getFunctionType());
  FunctionType calleeType = callee.getFunctionType();
  int64_t domainRank = task.getDomainShape().size();
  int64_t expectedInputCount = domainRank + graphType.getNumInputs();

  if (calleeType.getNumResults() != 0)
    return task.emitOpError()
           << "task callee @" << callee.getName()
           << " must not return values in the scheduler MVP ABI";
  if (callee.getVisibility() == SymbolTable::Visibility::Public)
    return task.emitOpError()
           << "task callee @" << callee.getName()
           << " must be a private device function in the scheduler MVP ABI";
  if (calleeType.getNumInputs() != expectedInputCount)
    return task.emitOpError()
           << "task callee @" << callee.getName()
           << " input count must be task domain rank plus enclosing graph "
              "function inputs";
  for (int64_t idx = 0; idx < domainRank; ++idx) {
    if (!calleeType.getInput(idx).isInteger(32))
      return task.emitOpError()
             << "task callee @" << callee.getName()
             << " coordinate argument " << idx << " must be i32";
  }
  for (auto [idx, typePair] : llvm::enumerate(llvm::zip(
           graphType.getInputs(),
           calleeType.getInputs().drop_front(domainRank)))) {
    Type graphArgType = std::get<0>(typePair);
    Type calleeArgType = std::get<1>(typePair);
    if (graphArgType != calleeArgType)
      return task.emitOpError()
             << "task callee @" << callee.getName()
             << " payload argument " << idx
             << " must match the enclosing graph function input type";
  }
  return success();
}

static LogicalResult verifyFunctionTaskGraph(FunctionOpInterface func) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  std::set<std::string> functionNames;
  if (module) {
    module.walk([&](FunctionOpInterface candidate) {
      functionNames.insert(candidate.getName().str());
    });
  }

  std::map<std::string, GridDefinition> grids;
  std::set<std::string> taskNames;
  bool sawError = false;

  func->walk([&](TaskGridCreateOp create) {
    std::string gridName = create.getGridName().str();
    auto [it, inserted] = grids.try_emplace(
        gridName, GridDefinition{create, getShape(create.getOperation())});
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
    }
    if (auto callee = task->getAttrOfType<FlatSymbolRefAttr>("callee")) {
      std::string calleeName = callee.getValue().str();
      auto calleeFunc = module ? module.lookupSymbol<mlir::triton::FuncOp>(
                                     calleeName)
                               : mlir::triton::FuncOp();
      bool invalidCallee = false;
      if (!functionNames.count(calleeName) || !calleeFunc) {
        task.emitOpError()
            << "task callee references unknown tt.func @" << calleeName;
        sawError = true;
        invalidCallee = true;
      }
      if (calleeName == func.getName().str()) {
        task.emitOpError()
            << "task callee must not reference the enclosing graph function";
        sawError = true;
        invalidCallee = true;
      }
      if (!invalidCallee && calleeFunc &&
          failed(verifyTaskCalleeAbi(task, func, calleeFunc)))
        sawError = true;
    }
    if (failed(verifyTaskMapReferences(task, task.getReads(), "read", grids)))
      sawError = true;
    if (failed(verifyTaskMapReferences(task, task.getWrites(), "write", grids)))
      sawError = true;
  });

  return failure(sawError);
}

class TritonTleVerifyTaskGraphPass
    : public impl::TritonTleVerifyTaskGraphBase<
          TritonTleVerifyTaskGraphPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool sawError = false;

    module.walk([&](FunctionOpInterface func) {
      if (failed(verifyFunctionTaskGraph(func)))
        sawError = true;
    });

    if (sawError)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
