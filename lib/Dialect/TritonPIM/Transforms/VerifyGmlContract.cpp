//===----------------------------------------------------------------------===//
//
// Checks the invariants that span more than one attribute of the expanded
// operator IR -- the ones no single op's verifier can see.
//
// Read-only on purpose. A pass that repaired a phase count or filled in a
// missing field would be hiding the disagreement the graph hand-off needs to
// see, and the failure it papers over is the kind that produces a plausible
// number rather than an error.
//
// Runs after `pim-expand-phases`: before it, the phase chains do not exist yet
// and there is nothing to check.
//
// Three of the plan's seven contract rules are NOT here, on purpose -- they are
// single-op facts, so `Dialect.cpp`'s attribute verifiers already reject them
// everywhere, not just after this pass runs:
//
//   weight_binding.role vs. stationarity   -> StationarityAttr / MatmulOp::verify
//   groupDequantAccum vs. binding.groupSize -> DatapathAttr / MatmulOp::verify
//   contraction's two forms are exclusive   -> ContractionAttr::verify
//
// Re-checking them here would add no coverage and two places to keep in sync.
// What does belong here is the opposite kind of rule: one that depends on the
// expansion having run. `verifyPhaseUnitIsSpecific` is that -- see its comment.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMVERIFYGMLCONTRACT
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

// One phase as this pass sees it: which op owns it, its index, and which
// earlier phases that op says it reads.
struct Phase {
  Operation *op;
  int64_t index;
  SmallVector<int64_t> reads;
};

// The phases declared inside `fn`, in walk order.
//
// An op carries an array because the encoding allows more than one, so this
// flattens rather than assuming one per op.
SmallVector<Phase> collectPhases(Operation *fn) {
  SmallVector<Phase> phases;
  fn->walk([&](Operation *op) {
    auto specs = op->getAttrOfType<ArrayAttr>("phases");
    if (!specs)
      return;
    for (Attribute entry : specs) {
      auto spec = dyn_cast<PhaseSpecAttr>(entry);
      if (!spec)
        continue;
      SmallVector<int64_t> reads;
      if (auto list = spec.getReads())
        for (Attribute read : list)
          reads.push_back(cast<IntegerAttr>(read).getInt());
      phases.push_back(Phase{op, spec.getIndex(), std::move(reads)});
    }
  });
  return phases;
}

// `isPhased` and `phases` are two spellings of one fact, so an op carrying one
// without the other is a disagreement a reader cannot resolve. Checked in both
// directions: a marker with no phase would have downstream code treat an
// ordinary op as phase-shaped, and a phase with no marker has the opposite
// failure.
LogicalResult checkPhasedMarker(Operation *fn) {
  WalkResult result = fn->walk([&](Operation *op) {
    bool marked = op->hasAttr("isPhased");
    bool phased = op->hasAttr("phases");
    if (marked == phased)
      return WalkResult::advance();

    if (marked)
      op->emitError("isPhased marks this op as phase-shaped, but it carries no "
                    "phases attribute; the two say the same thing and one of "
                    "them was set alone");
    else
      op->emitError("this op carries a phases attribute but no isPhased "
                    "marker; run -pim-expand-phases to have both written, or "
                    "drop the attribute");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

// 并查集：把由数据流相连的相位 op 并成一条链。
namespace {
class ChainDSU {
public:
  explicit ChainDSU(unsigned n) : parent(n) {
    for (unsigned i = 0; i < n; ++i)
      parent[i] = i;
  }
  unsigned find(unsigned i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  }
  void join(unsigned a, unsigned b) { parent[find(a)] = find(b); }

private:
  SmallVector<unsigned> parent;
};
} // namespace

// 把相位按数据流切成链。
//
// 相位号是**同一条链内**的遍历序号，不是函数级的全局编号：它在 GML 里是同一
// 节点内的 `*_phase_<k>` 字段族后缀，两条互不相干的链各自从 0 数起是合法的。
// 链的判据是数据流——相 1、2 都读相 0 属于同一条（扇出同样是同链），而两个
// 各自独立的算子之间没有任何 def-use 通路。
SmallVector<SmallVector<Phase>> groupIntoChains(ArrayRef<Phase> phases) {
  DenseMap<Operation *, unsigned> indexOf;
  for (unsigned i = 0; i < phases.size(); ++i)
    indexOf[phases[i].op] = i;

  ChainDSU dsu(phases.size());
  // Softmax 的稳定化 `sub` 不带 phase_spec（它折进 exp 相），但仍在相 0
  // 与相 1 之间传值。只看带相位的 def-use 会把一条链拆开，相 0 变成「缺号」。
  // 所以沿操作数向上走，穿过无相位的中间 op，直到碰到另一个相位 op。
  for (unsigned i = 0; i < phases.size(); ++i) {
    SmallVector<Value> work(phases[i].op->getOperands().begin(),
                            phases[i].op->getOperands().end());
    DenseSet<Operation *> seen;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      Operation *def = v.getDefiningOp();
      if (!def || !seen.insert(def).second)
        continue;
      auto it = indexOf.find(def);
      if (it != indexOf.end()) {
        dsu.join(i, it->second);
        continue;
      }
      work.append(def->operand_begin(), def->operand_end());
    }
  }

  DenseMap<unsigned, unsigned> chainOf;
  SmallVector<SmallVector<Phase>> chains;
  for (unsigned i = 0; i < phases.size(); ++i) {
    auto [it, inserted] = chainOf.try_emplace(dsu.find(i), chains.size());
    if (inserted)
      chains.emplace_back();
    chains[it->second].push_back(phases[i]);
  }
  return chains;
}

// 每条链内，相位号必须唯一且跑满 0..n-1。缺号或重号会让按「去重后的 index
// 个数」数相位的消费者数出与展开时不同的套数，表现为字段族少一套或多一套。
//
// 这同时兜住了 `reads`：相位只能读严格更早的一相（属性自己的 verifier 在没有
// 链上下文的情况下已经保证），所以一旦编号连续，每个 reads 都指向存在的相位。
// 没有单独的检查要做。
LogicalResult verifyIndicesInOneChain(ArrayRef<Phase> phases) {
  DenseSet<int64_t> present;
  for (const Phase &phase : phases) {
    if (!present.insert(phase.index).second)
      return phase.op->emitOpError()
             << "phase " << phase.index << " is claimed more than once";
  }

  for (int64_t index = 0, count = phases.size(); index < count; ++index) {
    if (present.contains(index))
      continue;
    for (const Phase &phase : phases) {
      if (phase.index <= index)
        continue;
      return phase.op->emitOpError()
             << "phase indices must run 0.." << (count - 1) << " with no gap; "
             << index << " is missing";
    }
    break;
  }
  return success();
}

LogicalResult verifyIndices(ArrayRef<Phase> phases) {
  for (ArrayRef<Phase> chain : groupIntoChains(phases))
    if (failed(verifyIndicesInOneChain(chain)))
      return failure();
  return success();
}

// Dynamic quantization's chain is the one whose phase 0 is a grouped reduction:
// every other expansion starts from something else. Its phases 1 (the identity
// scale) and 2 (the reciprocal) both read phase 0 -- a fan-out, not a chain.
//
// This is the check worth having. Serializing the two as a chain still
// produces a valid-looking module and a graph that loads; the reciprocal is
// simply wrong by 256x, and nothing else in the pipeline would notice.
//
// 必须按链做：函数里只要有一条 DQ 链，就把「相 1/2 必须读 0」套到同函数里
// 其它链上——Softmax 的 exp 相、RoPE 的乘相本来就不带 reads=[0]，合法 IR
// 会被误杀。
LogicalResult verifyDynamicQuantFanOutInOneChain(ArrayRef<Phase> phases) {
  bool hasGroupedReductionAtZero = false;
  for (const Phase &phase : phases)
    if (phase.index == 0 && isa<GlobalPoolOp>(phase.op))
      hasGroupedReductionAtZero = true;
  if (!hasGroupedReductionAtZero)
    return success();

  for (const Phase &phase : phases) {
    if (phase.index != 1 && phase.index != 2)
      continue;
    if (llvm::is_contained(phase.reads, 0))
      continue;
    return phase.op->emitOpError()
           << "dynamic quantization's phase " << phase.index
           << " must read phase 0; the identity scale and the reciprocal are a "
              "fan-out, and reading them in sequence puts the reciprocal off "
              "by 256x";
  }
  return success();
}

LogicalResult verifyDynamicQuantFanOut(ArrayRef<Phase> phases) {
  for (ArrayRef<Phase> chain : groupIntoChains(phases))
    if (failed(verifyDynamicQuantFanOutInOneChain(chain)))
      return failure();
  return success();
}

// `pim.normalize` runs on the vector unit, and its carrying none of the
// fixed-function datapath fields is the point: filling in a default would emit
// keys the reference graph does not have, which reads as a different operator.
LogicalResult verifyNormalizeIsBare(NormalizeOp op) {
  for (StringRef forbidden : {"datapath", "fpsu", "kantor"}) {
    if (op->hasAttr(forbidden))
      return op.emitOpError()
             << "must not carry " << forbidden
             << ": it runs on the vector unit, and an empty datapath field set "
                "is what says so";
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName().strref().starts_with("pooling"))
      return op.emitOpError()
             << "must not carry " << attr.getName()
             << ": normalization performs no pooling reduction";
  }
  return success();
}

// A freshly expanded phase must name the block it runs on, not the umbrella
// `cstl` that used to stand for six of them (fpsu / kantor / pooling /
// activation / combiner / store). Cost extraction and the graph format both
// key on the specific block: under one name a table lookup, a rescale and a
// grouped reduction are indistinguishable, so the fields for all three get
// filled from whichever rule matched first.
//
// **This is the one contract check a per-attribute verifier cannot make.** The
// rule is not "cstl is invalid" -- old un-expanded IR uses it legitimately and
// must keep round-tripping (see `FunctionalUnit`'s first four ordinals, frozen
// for exactly that reason). The rule is "cstl is invalid *on an op that just
// came out of the expansion*", and an attribute verifier has no way to know
// whether the pass has run. Only a pass scheduled after it does.
LogicalResult verifyPhaseUnitIsSpecific(ArrayRef<Phase> phases) {
  for (const Phase &phase : phases) {
    auto spec = phase.op->getAttrOfType<ArrayAttr>("phases");
    if (!spec)
      continue;
    for (Attribute entry : spec) {
      auto ps = dyn_cast<PhaseSpecAttr>(entry);
      if (!ps || ps.getIndex() != phase.index)
        continue;
      if (ps.getUnit() == FunctionalUnit::CSTL)
        return phase.op->emitOpError()
               << "phase " << phase.index
               << " runs on `cstl`, which names six different blocks at once. "
                  "An expanded phase must say which one (fpsu / kantor / "
                  "pooling / activation / combiner); leaving it as cstl makes a "
                  "table lookup, a rescale and a grouped reduction "
                  "indistinguishable to cost extraction";
    }
  }
  return success();
}

struct TritonPIMVerifyGmlContractPass
    : public mlir::triton::pim::impl::TritonPIMVerifyGmlContractBase<
          TritonPIMVerifyGmlContractPass> {
  using mlir::triton::pim::impl::TritonPIMVerifyGmlContractBase<
      TritonPIMVerifyGmlContractPass>::TritonPIMVerifyGmlContractBase;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](Operation *fn) {
      // Only function-like scopes hold a phase chain; the module itself and the
      // ops inside a function are visited separately.
      if (!isa<FunctionOpInterface>(fn))
        return WalkResult::advance();

      if (failed(verifyNormalizeOps(fn)) || failed(checkPhasedMarker(fn)))
        return WalkResult::interrupt();

      SmallVector<Phase> phases = collectPhases(fn);
      if (phases.empty())
        return WalkResult::advance();

      if (failed(verifyIndices(phases)) ||
          failed(verifyDynamicQuantFanOut(phases)) ||
          failed(verifyPhaseUnitIsSpecific(phases)))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult verifyNormalizeOps(Operation *fn) {
    WalkResult result = fn->walk([&](NormalizeOp op) {
      return failed(verifyNormalizeIsBare(op)) ? WalkResult::interrupt()
                                               : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }
};

} // namespace
