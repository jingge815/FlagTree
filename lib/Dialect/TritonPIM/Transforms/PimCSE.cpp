//===----------------------------------------------------------------------===//
//
// 合并同一基本块内结构相同的纯 pim.* 算子。
//
// 上游 CSE 只处理标了 Pure 的算子，pim.matmul、pim.eltwise、pim.lut 都没标。
// 这里按性质表判定。跨基本块的不合并，带内存副作用的不合并。
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"
#include "PimPure.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMCSE
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton::pim;

namespace {

// 合并计划里一个值最终换成谁：定义它的算子要被合并时，换成合并对象的同号结果。
using MergePlan = DenseMap<Operation *, Operation *>;

static Value mapped(const MergePlan &plan, Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return value;
  Operation *target = plan.lookup(result.getOwner());
  return target ? target->getResult(result.getResultNumber()) : value;
}

// 结构相同：算子名、属性、结果类型一致，操作数按合并计划换过之后一致。
static bool sameStructure(Operation *a, Operation *b, const MergePlan &plan) {
  if (a->getName() != b->getName())
    return false;
  if (a->getAttrDictionary() != b->getAttrDictionary())
    return false;
  // 结果类型不同不能合并：同一条 arith 在不同分块编码下类型不同，
  // 合到一起会让用户拿到类型对不上的值。
  if (a->getResultTypes() != b->getResultTypes())
    return false;
  if (a->getNumOperands() != b->getNumOperands())
    return false;
  for (auto [x, y] : llvm::zip(a->getOperands(), b->getOperands()))
    if (mapped(plan, x) != mapped(plan, y))
      return false;
  return true;
}

// 整算子函数里没人用的张量结果是内核输出，合并会让降级认错输出，不参与。
// 多结果算子只要有一个结果是这种情况就整体保留。
static bool mayBeOperatorOutput(Operation *op) {
  auto func = op->getParentOfType<triton::FuncOp>();
  if (!isOperatorLevelFunc(func))
    return false;
  return llvm::any_of(op->getResults(), [](Value result) {
    return result.use_empty() && isa<RankedTensorType>(result.getType());
  });
}

static bool isCandidate(Operation &op) {
  // 带区域的算子语义在区域体里，这里不比较区域，交给上游 CSE。
  return isPurePim(&op) && op.getNumResults() != 0 &&
         op.getNumRegions() == 0 && !mayBeOperatorOutput(&op);
}

// 相位算子按数据流并成链，与 `-pim-verify-gml-contract` 同一判据：
// 沿操作数向上走，穿过不带相位的算子，碰到相位算子就并到一条链。返回每个相位算子的链号。
static DenseMap<Operation *, unsigned> phaseChains(Block &block) {
  DenseMap<Operation *, unsigned> index;
  SmallVector<unsigned> parent;
  auto find = [&](unsigned i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  };
  for (Operation &op : block)
    if (op.hasAttr("phases")) {
      index[&op] = parent.size();
      parent.push_back(parent.size());
    }
  for (auto [op, id] : index) {
    SmallVector<Value> work(op->getOperands());
    DenseSet<Operation *> visited;
    while (!work.empty()) {
      Operation *def = work.pop_back_val().getDefiningOp();
      if (!def || !visited.insert(def).second)
        continue;
      auto it = index.find(def);
      if (it != index.end()) {
        parent[find(id)] = find(it->second);
        continue;
      }
      work.append(def->operand_begin(), def->operand_end());
    }
  }
  DenseMap<Operation *, unsigned> chainOf;
  for (auto [op, id] : index)
    chainOf[op] = find(id);
  return chainOf;
}

// 相位链：每个相位算子的链号，以及每条链按块序排好的相位算子。
struct PhaseChains {
  DenseMap<Operation *, unsigned> chainOf;
  DenseMap<unsigned, SmallVector<Operation *>> members;
};

static PhaseChains collectChains(Block &block) {
  PhaseChains chains;
  chains.chainOf = phaseChains(block);
  for (Operation &op : block) {
    auto it = chains.chainOf.find(&op);
    if (it != chains.chainOf.end())
      chains.members[it->second].push_back(&op);
  }
  return chains;
}

// 两条链能否逐相对上：长度相同，同一位置的算子名、属性、结果类型都相同。
// 操作数是否相同交给合并计划判定。
static bool alignedChains(ArrayRef<Operation *> a, ArrayRef<Operation *> b) {
  if (a.size() != b.size())
    return false;
  for (auto [x, y] : llvm::zip(a, b))
    if (x->getName() != y->getName() ||
        x->getAttrDictionary() != y->getAttrDictionary() ||
        x->getResultTypes() != y->getResultTypes())
      return false;
  return true;
}

// 相位链配对：键是后出现的链，值是它整条合并进去的先出现的链。
using ChainPairs = DenseMap<unsigned, unsigned>;

// 相位算子只能合并到配对链上；没有配对的链一相都不合并。
static bool chainsMayPair(const PhaseChains &chains, Operation *op,
                          Operation *prev, const ChainPairs &pairs) {
  auto self = chains.chainOf.find(op);
  if (self == chains.chainOf.end())
    return true;
  auto other = chains.chainOf.find(prev);
  auto into = pairs.find(self->second);
  return other != chains.chainOf.end() && into != pairs.end() &&
         into->second == other->second;
}

// 按块序为每个候选找第一个结构相同的前驱，只记计划不改 IR。
static MergePlan planMerges(Block &block, const PhaseChains &chains,
                            const ChainPairs &pairs) {
  MergePlan plan;
  SmallVector<Operation *> seen;
  for (Operation &op : block) {
    if (!isCandidate(op))
      continue;
    Operation *first = nullptr;
    for (Operation *prev : seen)
      if (sameStructure(prev, &op, plan) &&
          chainsMayPair(chains, &op, prev, pairs)) {
        first = prev;
        break;
      }
    if (first)
      plan[&op] = first;
    else
      seen.push_back(&op);
  }
  return plan;
}

// 链 later 的每一相都合并进链 earlier，且正好覆盖 earlier 的全部相位。
static bool mergesWhole(const PhaseChains &chains, const MergePlan &plan,
                        unsigned later, unsigned earlier) {
  ArrayRef<Operation *> ops = chains.members.find(later)->second;
  DenseSet<Operation *> targets;
  for (Operation *op : ops) {
    Operation *target = plan.lookup(op);
    if (!target || chains.chainOf.lookup(target) != earlier)
      return false;
    targets.insert(target);
  }
  return targets.size() == chains.members.find(earlier)->second.size();
}

// 逐对试合并，只保留整条相同的配对。只合并一段前缀会让两条链共用前缀，
// 契约校验把它们并成一条，相位重号。每条链单独与每条更早的链试，
// 所以别的链只与它共享前缀，不影响它与整条相同的链配对，结果不随排列变化。
static ChainPairs pairWholeChains(Block &block, const PhaseChains &chains) {
  SmallVector<unsigned> order;
  for (Operation &op : block) {
    auto it = chains.chainOf.find(&op);
    if (it != chains.chainOf.end() && !llvm::is_contained(order, it->second))
      order.push_back(it->second);
  }
  ChainPairs pairs;
  // 链的操作数可能来自别的链，那条链合并后才对得上，所以重复到不再新增配对。
  for (bool grew = true; grew;) {
    grew = false;
    for (auto [i, later] : llvm::enumerate(order)) {
      if (pairs.count(later))
        continue;
      for (unsigned earlier : ArrayRef(order).take_front(i)) {
        if (pairs.count(earlier) ||
            !alignedChains(chains.members.find(later)->second,
                           chains.members.find(earlier)->second))
          continue;
        ChainPairs trial = pairs;
        trial[later] = earlier;
        if (mergesWhole(chains, planMerges(block, chains, trial), later,
                        earlier)) {
          pairs = std::move(trial);
          grew = true;
          break;
        }
      }
    }
  }
  return pairs;
}

struct TritonPIMCSEPass
    : public mlir::triton::pim::impl::TritonPIMCSEBase<TritonPIMCSEPass> {
  void runOnOperation() override {
    getOperation().walk([&](Block *block) {
      // 1. 先定相位链怎么配对，再按配对模拟合并，只记计划不改 IR。
      PhaseChains chains = collectChains(*block);
      MergePlan plan = planMerges(*block, chains, pairWholeChains(*block, chains));
      // 2. 按计划替换并删除。合并对象本身从不被合并，替换一步到位。
      for (Operation &op : llvm::make_early_inc_range(*block)) {
        Operation *first = plan.lookup(&op);
        if (!first)
          continue;
        op.replaceAllUsesWith(first);
        op.erase();
      }
    });
  }
};

} // namespace
