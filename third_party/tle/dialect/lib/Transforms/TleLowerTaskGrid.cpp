// flagtree tle
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <string>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWERTASKGRID
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

struct TaskGridDefinition {
  TaskGridCreateOp create;
  SmallVector<std::string> fieldNames;
  SmallVector<int64_t> shape;
};

static std::string getTaskGridKey(Operation *op) {
  auto scope = op->getAttrOfType<StringAttr>("scope");
  auto name = op->getAttrOfType<StringAttr>("grid_name");
  if (!scope || !name)
    return "";
  return (scope.getValue() + "::" + name.getValue()).str();
}

static SmallVector<std::string> getStringArray(ArrayAttr attr) {
  SmallVector<std::string> values;
  values.reserve(attr.size());
  for (Attribute item : attr)
    values.push_back(cast<StringAttr>(item).getValue().str());
  return values;
}

static SmallVector<int64_t> getShape(Operation *op) {
  SmallVector<int64_t> values;
  auto shape = op->getAttrOfType<DenseI64ArrayAttr>("shape");
  if (!shape)
    return values;
  values.append(shape.asArrayRef().begin(), shape.asArrayRef().end());
  return values;
}

static bool isTaskGridUseOp(Operation *op) {
  return isa<TaskGridTileIdOp, TaskGridCommitOp>(op);
}

static bool arraysEqual(ArrayAttr lhs, ArrayRef<std::string> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [attr, name] : llvm::zip_equal(lhs, rhs)) {
    if (cast<StringAttr>(attr).getValue() != name)
      return false;
  }
  return true;
}

static LogicalResult verifyTaskGridUse(Operation *op,
                                       const TaskGridDefinition &definition) {
  auto fieldNamesAttr = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNamesAttr)
    return op->emitOpError("requires field_names attribute");
  if (!arraysEqual(fieldNamesAttr, definition.fieldNames))
    return op->emitOpError()
           << "field_names do not match the preceding task_grid.create";

  SmallVector<int64_t> opShape = getShape(op);
  if (!opShape.empty() && opShape != definition.shape)
    return op->emitOpError()
           << "shape does not match the preceding task_grid.create";

  if (auto tile = dyn_cast<TaskGridTileIdOp>(op)) {
    if (llvm::any_of(tile->getResults(),
                     [](OpResult result) { return !result.use_empty(); }))
      return tile.emitOpError()
             << "result use requires scheduler codegen; marker cleanup cannot "
                "lower grid.tile_id values yet";
  }

  return success();
}

static LogicalResult collectTaskGridOps(Operation *container,
                                        SmallVectorImpl<Operation *> &ops) {
  std::map<std::string, TaskGridDefinition> definitions;
  bool sawError = false;

  container->walk([&](Operation *op) {
    if (op == container)
      return;

    if (auto create = dyn_cast<TaskGridCreateOp>(op)) {
      ops.push_back(op);
      std::string key = getTaskGridKey(op);
      auto [it, inserted] = definitions.try_emplace(
          key, TaskGridDefinition{create, getStringArray(create.getFieldNames()),
                                  getShape(create.getOperation())});
      if (!inserted) {
        create.emitOpError("duplicates an existing task_grid.create");
        sawError = true;
      }
      return;
    }

    if (!isTaskGridUseOp(op))
      return;

    ops.push_back(op);
    auto it = definitions.find(getTaskGridKey(op));
    if (it == definitions.end()) {
      op->emitOpError("requires a preceding matching task_grid.create");
      sawError = true;
      return;
    }
    if (failed(verifyTaskGridUse(op, it->second)))
      sawError = true;
  });

  return failure(sawError);
}

class TritonTleLowerTaskGridPass
    : public impl::TritonTleLowerTaskGridBase<TritonTleLowerTaskGridPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> taskGridOps;
    bool sawError = false;

    module.walk([&](FunctionOpInterface func) {
      if (failed(collectTaskGridOps(func.getOperation(), taskGridOps)))
        sawError = true;
    });

    if (sawError) {
      signalPassFailure();
      return;
    }

    for (Operation *op : llvm::reverse(taskGridOps))
      op->erase();
  }
};

} // namespace

} // namespace mlir::triton::tle
