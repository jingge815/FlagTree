//===----------------------------------------------------------------------===//
//
// Folds an activation, and an optional trailing pool, into the operator that
// produces their input:
//
//   matmul/conv/eltwise -> lut           =>  matmul/conv/eltwise {activation}
//   matmul/conv/eltwise -> lut -> pool   =>  ... {activation, fusedPool}
//
// The target's graph format keeps an activation inside its producer's node and
// cannot express a standalone one, so this runs before the graph hand-off.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/IR/Builders.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMFUSEACTIVATION
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

// The operators whose node can hold a folded activation. These are exactly the
// ones that carry `activation` and `fusedPool` attributes.
// 与 flagos-pim-compiler 的 contracts/fusion_contract.py 同步；改一边必须改另一边。
// conv 也在：图编译器那张表没有它，只是因为目标模型没有卷积，不是禁止 conv 折激活。
static bool isFusionTarget(Operation *op) {
  return isa<MatmulOp, ConvOp, EltwiseOp>(op);
}

// Reads the `activation` attribute of a fusion target, whichever op it is.
static ActSpecAttr getActivationOf(Operation *op) {
  return op->getAttrOfType<ActSpecAttr>("activation");
}

// True when `value` feeds exactly one operation, so rewriting that consumer
// cannot strand another reader of the pre-activation result.
static bool hasSingleUse(Value value) { return value.hasOneUse(); }

// The activation configuration a `pim.lut` describes, as it would sit on the
// producing operator. Returns null when the lut carries a table operand: a table
// is data, and an attribute has nowhere to put an SSA value.
static ActSpecAttr activationSpecOf(LutOp lut) {
  return ActSpecAttr::get(lut.getContext(), lut.getKind(), lut.getMode(),
                          lut.getAlphaAttr(), lut.getClipMinAttr(),
                          lut.getClipMaxAttr());
}

// The pooling configuration a `pim.pool` describes, as it would sit on the
// producing operator.
static PoolSpecAttr poolSpecOf(PoolOp pool) {
  return PoolSpecAttr::get(pool.getContext(), pool.getKind(),
                           pool.getWindowAttr());
}

//===----------------------------------------------------------------------===//
// Fusion
//===----------------------------------------------------------------------===//

// Folds `lut` into `producer`, and `pool` too when it is present. The producer
// takes over the last op's result type, since the fused node produces what the
// chain used to produce.
// The fusion as the graph format writes it: a nested block naming the folded
// operator. The block *names* the fusion -- the table, window and activation
// mode stay on the producer, because that is where the datapath they configure
// lives.
// The activation's name as the graph format spells it. Only `silu` occurs in
// the reference artifact (one fusion, on the gate projection), so that one is
// known; the rest are single words whose spelling follows from it.
//
// Multi-word kinds are deliberately absent. `leaky_relu` would be `LeakyRelu`
// or `Leaky_relu` depending on a convention the reference does not exhibit, and
// guessing produces a block name the target's parser may not match -- a field
// it silently fails to read. They are refused below instead.
static std::optional<std::string> graphFormatName(ActivationKind kind) {
  switch (kind) {
  case ActivationKind::Silu:       return "Silu";
  case ActivationKind::Relu:       return "Relu";
  case ActivationKind::Sigmoid:    return "Sigmoid";
  case ActivationKind::Tanh:       return "Tanh";
  case ActivationKind::Gelu:       return "Gelu";
  case ActivationKind::LeakyRelu:  return "LeakyRelu";
  case ActivationKind::Exp:        return "Exp";
  case ActivationKind::Sqrt:       return "Sqrt";
  case ActivationKind::Reciprocal: return "Reciprocal";
  // `rsqrt` 与 `identity` 不折。`rsqrt` 是 RMSNorm 的一部分，折进主算子会让
  // RMSNorm 不再是独立节点；`identity` 是动态量化相位里的查表，不是激活。
  // 这份名单与 pim-compiler 的 contracts/fusion_contract.py 同步，
  // 改一边必须改另一边。
  default:                         return std::nullopt;
  }
}

static ContractionAttr namedContraction(MLIRContext *ctx,
                                        ActivationKind kind) {
  auto name = graphFormatName(kind);
  if (!name)
    return {};
  return ContractionAttr::get(
      ctx, ContractionForm::Named,
      StringAttr::get(ctx, "fused_" + *name + "_act"),
      // Every folded activation is evaluated through the lookup table, so the
      // inner operator is always the table op.
      StringAttr::get(ctx, "Lut"), StringAttr::get(ctx, *name),
      /*flagName=*/StringAttr{});
}

static void fuse(Operation *producer, LutOp lut, ActSpecAttr activation,
                 PoolOp pool, PoolSpecAttr pooling) {
  // The table moves onto the producer before the lut is erased: a table is a
  // value, not an attribute, and the target format wants it on the node that
  // carries the activation.
  if (lut.getTable())
    if (auto mm = dyn_cast<MatmulOp>(producer))
      mm.getLutTableMutable().assign(lut.getTable());

  producer->setAttr("activation", activation);
  // A kind with no known graph-format spelling gets no contraction block rather
  // than a guessed one: a wrong block name is a field the target's parser
  // silently fails to match. The activation attribute above still records what
  // was folded, so nothing is lost -- only the nested block is withheld.
  if (auto contraction =
          namedContraction(producer->getContext(), activation.getKind()))
    producer->setAttr("contraction", contraction);
  if (pooling)
    producer->setAttr("fusedPool", pooling);

  Operation *last = pool ? pool.getOperation() : lut.getOperation();
  producer->getResult(0).setType(last->getResult(0).getType());
  last->getResult(0).replaceAllUsesWith(producer->getResult(0));

  if (pool)
    pool.erase();
  lut.erase();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TritonPIMFuseActivationPass
    : public mlir::triton::pim::impl::TritonPIMFuseActivationBase<
          TritonPIMFuseActivationPass> {
  using mlir::triton::pim::impl::TritonPIMFuseActivationBase<
      TritonPIMFuseActivationPass>::TritonPIMFuseActivationBase;

  void runOnOperation() override {
    // Collect first: fusing erases ops, so walking live IR would be unsound.
    SmallVector<LutOp> candidates;
    getOperation().walk([&](LutOp lut) { candidates.push_back(lut); });

    for (LutOp lut : candidates) {
      Operation *producer = lut.getSrc().getDefiningOp();
      if (!producer || !isFusionTarget(producer))
        continue;

      // 不在可折名单里的激活不折。`rsqrt` 是 RMSNorm 的一部分，`identity`
      // 是动态量化相位的查表，折进去会让主算子吞掉一个本该独立的节点。
      // 这份名单与 pim-compiler 的 contracts/fusion_contract.py 同步。
      if (!graphFormatName(lut.getKind()))
        continue;

      // `silu` folds only into the gate projection, and the gate projection is
      // a matmul. The same rule is spelled `GATE_TARGETS` in the graph
      // compiler's `contracts/fusion_contract.py`. Folding it into conv or
      // eltwise would put a `fused_Silu_act` contraction on a node the graph
      // format never carries one on, so the lut stays separate instead.
      if (lut.getKind() == ActivationKind::Silu && !isa<MatmulOp>(producer))
        continue;

      // A node holds one activation, so a producer that already has one has no
      // room for a second.
      if (getActivationOf(producer))
        continue;

      // Folding rewires the producer's result to the activation's; another
      // reader of the pre-activation value would lose its definition.
      if (!hasSingleUse(lut.getSrc()))
        continue;

      // A table folds into a matmul, which is where the graph format keeps the
      // gate projection's table. The other producers have nowhere to put it;
      // refusing beats leaving the activation silently unfused, which would
      // drop a whole activation/contraction field family from the graph.
      if (lut.getTable() && !isa<MatmulOp>(producer)) {
        lut.emitError(
            "a lut with a table operand folds only into a matmul; this "
            "producer has nowhere to put the table");
        return signalPassFailure();
      }

      ActSpecAttr activation = activationSpecOf(lut);
      if (!activation)
        continue;

      // A pool directly after the activation folds into the same node. It has to
      // be the activation's only consumer, for the same reason.
      PoolOp pool;
      PoolSpecAttr pooling;
      if (hasSingleUse(lut.getResult())) {
        if (auto candidate = dyn_cast<PoolOp>(*lut.getResult().user_begin())) {
          if (!producer->getAttrOfType<PoolSpecAttr>("fusedPool")) {
            pool = candidate;
            pooling = poolSpecOf(candidate);
          }
        }
      }

      fuse(producer, lut, activation, pool, pooling);
    }
  }
};

} // namespace
