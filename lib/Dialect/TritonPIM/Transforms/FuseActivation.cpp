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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMFUSEACTIVATION
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

// 能持有折入激活的算子，按有没有 `activation` 属性位判定，不按名字。
// 当前声明了这个属性位的是 matmul、conv、eltwise 三个，与原先按名字
// 判定的名单相同；以后新增带该属性位的算子不用再改这里。
// 与 flagos-pim-compiler 的 contracts/fusion_contract.py 同步，改一边必须改另一边。
static bool isFusionTarget(Operation *op) {
  // 带 activation 属性位的算子才能承接折入的激活。属性要么已经写上了，
  // 要么在 ODS 里声明了这个名字。
  if (op->getAttr("activation"))
    return true;
  if (auto info = op->getRegisteredInfo())
    return llvm::is_contained(info->getAttributeNames(), "activation");
  return false;
}

// Reads the `activation` attribute of a fusion target, whichever op it is.
static ActSpecAttr getActivationOf(Operation *op) {
  return op->getAttrOfType<ActSpecAttr>("activation");
}

// True when `value` feeds exactly one operation, so rewriting that consumer
// cannot strand another reader of the pre-activation result.
static bool hasSingleUse(Value value) { return value.hasOneUse(); }

// The activation configuration a `pim.lut` describes, as it would sit on the
// producing operator. A table operand is not part of the attribute: `fuse` moves
// it onto a matmul, and the caller rejects a table on any other producer.
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


// A splat f32 constant with exactly this value. The TTIR chain's 0 and 1 are
// dense splats, not scalar constants.
static bool isSplatConstant(Value v, double value) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  auto dense = dyn_cast<SplatElementsAttr>(cst.getValue());
  if (!dense || !dense.getElementType().isF32())
    return false;
  return dense.getSplatValue<APFloat>().convertToDouble() == value;
}

// Triton 对 `sigmoid(r)` 展开出的四连：0-r -> exp -> 1+e -> 1/(1+e)。
// 从 subf 开始认，每个中间结果单用户；返回四个算子，认不出返回空。
static SmallVector<Operation *> matchSigmoidFrom(arith::SubFOp subf,
                                                 Value root) {
  if (!subf.getResult().hasOneUse() || subf.getRhs() != root ||
      !isSplatConstant(subf.getLhs(), 0.0))
    return {};
  auto exp = dyn_cast<math::ExpOp>(*subf.getResult().user_begin());
  if (!exp || !exp.getResult().hasOneUse())
    return {};
  auto addf = dyn_cast<arith::AddFOp>(*exp.getResult().user_begin());
  if (!addf || !addf.getResult().hasOneUse())
    return {};
  Value one = addf.getLhs() == exp.getResult() ? addf.getRhs() : addf.getLhs();
  if (!isSplatConstant(one, 1.0))
    return {};
  auto divf = dyn_cast<arith::DivFOp>(*addf.getResult().user_begin());
  if (!divf || divf.getRhs() != addf.getResult() ||
      !isSplatConstant(divf.getLhs(), 1.0))
    return {};
  return {subf.getOperation(), exp.getOperation(), addf.getOperation(),
          divf.getOperation()};
}

// dot 结果上的尾部激活。按链的结构判定种类，不按名字：
//   relu    = max(r, 0)
//   sigmoid = 上面的四连
//   silu    = r * sigmoid(r)，链根另有一个 mulf 用户
// 返回种类与链上算子（最后一个是链尾），认不出返回空链。
// 种类与 pim-compiler 的 contracts/fusion_contract.py 里
// LINEAR_TAIL_ACTIVATIONS 同步，改一边必须改另一边。
static std::pair<ActivationKind, SmallVector<Operation *>>
matchTtirActivation(Value root) {
  SmallVector<Operation *> users(root.getUsers());
  // relu：唯一用户是与 0 取大。
  if (users.size() == 1)
    if (auto max = dyn_cast<arith::MaxNumFOp>(users[0])) {
      Value other = max.getLhs() == root ? max.getRhs() : max.getLhs();
      if (isSplatConstant(other, 0.0))
        return {ActivationKind::Relu, {max.getOperation()}};
      return {};
    }
  // sigmoid：唯一用户是四连的开头。
  if (users.size() == 1)
    if (auto subf = dyn_cast<arith::SubFOp>(users[0])) {
      auto chain = matchSigmoidFrom(subf, root);
      if (!chain.empty())
        return {ActivationKind::Sigmoid, chain};
      return {};
    }
  // silu：两个用户，一个是四连开头，一个是乘回链根的 mulf。
  if (users.size() != 2)
    return {};
  auto subf = dyn_cast<arith::SubFOp>(users[0]);
  auto mulf = dyn_cast<arith::MulFOp>(users[1]);
  if (!subf || !mulf) {
    subf = dyn_cast<arith::SubFOp>(users[1]);
    mulf = dyn_cast<arith::MulFOp>(users[0]);
  }
  if (!subf || !mulf)
    return {};
  auto chain = matchSigmoidFrom(subf, root);
  if (chain.empty())
    return {};
  Value sig = chain.back()->getResult(0);
  if (!sig.hasOneUse() || *sig.user_begin() != mulf.getOperation())
    return {};
  Value other = mulf.getLhs() == root ? mulf.getRhs() : mulf.getLhs();
  if (other != sig)
    return {};
  chain.push_back(mulf.getOperation());
  return {ActivationKind::Silu, chain};
}

// 尾部激活挂在哪个值上：dot 结果直接出去时就是它，被 K 循环携带出去时
// 是循环结果。
static Value activationChainRoot(triton::DotOp dot) {
  Value acc = dot.getResult();
  // 链根就是 dot 结果本身；dot 结果被 K 循环携带出去时（唯一用户是
  // yield），链根换成循环结果——链在循环外面，读的是携带值。
  if (acc.hasOneUse()) {
    auto yield = dyn_cast<scf::YieldOp>(*acc.getUses().begin()->getOwner());
    if (yield) {
      // 只认单携带值循环：多携带值认不出下标就不猜。
      if (yield.getNumOperands() != 1)
        return {};
      auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!forOp || forOp.getNumResults() != 1)
        return {};
      return forOp.getResult(0);
    }
  }
  // dot 的累加器是所在循环的携带值时，链在 K 循环里面、每轮都激活；降到 C
  // 时整段 K 归约完只激活一次，折进去会改数值。
  if (auto arg = dyn_cast<BlockArgument>(dot.getC()))
    if (isa<scf::ForOp>(arg.getOwner()->getParentOp()))
      return {};
  return acc;
}

// 链上每个算子都要和链根在同一个块里。链在 scf.if 等嵌套区域里时只按条件
// 执行，折进 dot 会让激活无条件生效。
static bool chainInRootBlock(Value root, ArrayRef<Operation *> chain) {
  Block *block = root.getParentBlock();
  return llvm::all_of(chain,
                      [&](Operation *op) { return op->getBlock() == block; });
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

    // A 路形态：tt.dot 后接尾部激活的算术链。dot 没有 activation 属性位，
    // 链由 arith/math 算子组成，所以走这条独立的识别路径。
    SmallVector<triton::DotOp> dots;
    getOperation().walk([&](triton::DotOp dot) { dots.push_back(dot); });
    for (triton::DotOp dot : dots) {
      if (getActivationOf(dot))
        continue;
      Value root = activationChainRoot(dot);
      if (!root)
        continue;
      auto [kind, chain] = matchTtirActivation(root);
      if (chain.empty() || !chainInRootBlock(root, chain))
        continue;
      dot->setAttr("activation",
                   ActSpecAttr::get(dot.getContext(), kind, LutMode::Regular,
                                    nullptr, nullptr, nullptr));
      // 链尾结果改接链根：下游 truncf/store 读的就是循环携带值或 dot 结果。
      chain.back()->getResult(0).replaceAllUsesWith(root);
      for (Operation *op : llvm::reverse(chain))
        op->erase();
    }
  }
};

} // namespace
