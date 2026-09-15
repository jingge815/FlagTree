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
  if (lut.getTable())
    return {};

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
static void fuse(Operation *producer, LutOp lut, ActSpecAttr activation,
                 PoolOp pool, PoolSpecAttr pooling) {
  producer->setAttr("activation", activation);
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

      // A node holds one activation, so a producer that already has one has no
      // room for a second.
      if (getActivationOf(producer))
        continue;

      // Folding rewires the producer's result to the activation's; another
      // reader of the pre-activation value would lose its definition.
      if (!hasSingleUse(lut.getSrc()))
        continue;

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
