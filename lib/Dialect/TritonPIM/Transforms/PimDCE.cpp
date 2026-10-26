//===----------------------------------------------------------------------===//
//
// 删除结果没有用户的纯算子。
//
// 结果全部没有用户、且没有内存副作用的纯算子，多轮删除。
// 带内存副作用的算子不在纯算子表里，所以留着，不是靠反向标可达保住的。
// 上游 symbol-dce 只删符号，删不了函数体内的死算子。
//
// 整算子函数里，结果没有 SSA 用户不等于死。`pim-lower-to-emitc` 把
// 「最后一批没人用的张量结果」当作内核的输出，产出它的算子不论方言都保留；
// 更早的没人用的 pim.* 结果是被下游节点带外读取的中间值（动态量化的相 1
// 就是这样），也保留。A 路函数的输出走 tt.store 或 tt.return，没人用的
// 纯算子不论方言、不论是否产出张量都删。
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"
#include "PimPure.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMDCE
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton::pim;

namespace {

// 结果经 tt.store 或 tt.return 交出的函数按 A 路约定，没人用的张量是死的。
static bool usesExplicitOutputs(triton::FuncOp func) {
  if (func.getNumResults() > 0)
    return true;
  bool stores = false;
  func.walk([&](triton::StoreOp) { stores = true; });
  return stores;
}

// 整算子函数的内核输出是最后一批没人用的张量结果，口径与 `findOperatorOutputs` 相同。
// 返回产出这批结果的算子；它可能不是 pim.*，照样不能当死代码删。
static Operation *operatorOutputOwner(triton::FuncOp func) {
  if (usesExplicitOutputs(func))
    return nullptr;
  Operation *owner = nullptr;
  func.walk([&](Operation *op) {
    bool unusedTensor = llvm::any_of(op->getResults(), [](Value v) {
      return v.use_empty() && isa<RankedTensorType>(v.getType());
    });
    if (unusedTensor)
      owner = op;
  });
  return owner;
}

// 整算子函数里 pim.* 的张量结果可能是内核输出或带外读取的中间值，不能按无用户删除。
static bool mayBeOperatorOutput(
    Operation *op, const llvm::SmallPtrSetImpl<Operation *> &outputOwners) {
  if (!isOperatorLevelFunc(op->getParentOfType<triton::FuncOp>()))
    return false;
  if (outputOwners.contains(op))
    return true;
  if (!llvm::isa_and_nonnull<TritonPIMDialect>(op->getDialect()))
    return false;
  return llvm::any_of(op->getResults(),
                      [](Value v) { return isa<RankedTensorType>(v.getType()); });
}

struct TritonPIMDCEPass : public mlir::triton::pim::impl::TritonPIMDCEBase<TritonPIMDCEPass> {
  void runOnOperation() override {
    // 内核输出按 findOperatorOutputs 的口径先算出来，删的过程中不会变。
    llvm::SmallPtrSet<Operation *, 8> outputOwners;
    getOperation().walk([&](triton::FuncOp func) {
      if (!isOperatorLevelFunc(func))
        return;
      if (Operation *owner = operatorOutputOwner(func))
        outputOwners.insert(owner);
    });
    // 多轮：删掉一层后它的输入可能也变成无用户。
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      getOperation().walk([&](Operation *op) {
        // 终止符（tt.return）删掉会留下空块，块就非法了。
        if (op->hasTrait<OpTrait::IsTerminator>())
          return;
        if (!isPurePim(op) || mayBeOperatorOutput(op, outputOwners))
          return;
        if (llvm::all_of(op->getResults(),
                         [](Value v) { return v.use_empty(); }))
          dead.push_back(op);
      });
      for (Operation *op : dead) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace
