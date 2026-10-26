//===----------------------------------------------------------------------===//
//
// 折叠操作数全是常量的算子。
//
// 上游 canonicalize 折不动这些算子：PIM 算子没有 hasFolder，也没标 Pure。
// 覆盖 pim.eltwise、pim.lut 和整数 arith 运算。pim.matmul 不折，
// 它的一个操作数是整块权重，折叠等于把矩阵乘搬进编译期。
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/APFloat.h"

#include "triton/Dialect/TritonPIM/IR/Dialect.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMCONSTANTFOLD
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton::pim;

namespace {

// 常量值：dense 张量或标量。取不到就不是常量。
static Attribute constAttrOf(Value value) {
  auto def = value.getDefiningOp<arith::ConstantOp>();
  if (!def)
    return {};
  return def.getValue();
}

// 结果元素数超过这个上限就不折，避免把大常量计算搬进编译期。
static constexpr int64_t kMaxFoldElements = 4096;

// 一个浮点元素的运算。除零没有对齐的结果，返回空。
static std::optional<APFloat> evalFloat(EltwiseKind kind, APFloat l, APFloat r) {
  switch (kind) {
  case EltwiseKind::Add: l.add(r, APFloat::rmNearestTiesToEven); return l;
  case EltwiseKind::Sub: l.subtract(r, APFloat::rmNearestTiesToEven); return l;
  case EltwiseKind::Mul: l.multiply(r, APFloat::rmNearestTiesToEven); return l;
  case EltwiseKind::Max: return l.compare(r) == APFloat::cmpLessThan ? r : l;
  case EltwiseKind::Min: return r.compare(l) == APFloat::cmpLessThan ? r : l;
  case EltwiseKind::Div:
    if (r.isZero())
      return std::nullopt;
    l.divide(r, APFloat::rmNearestTiesToEven);
    return l;
  default:
    return std::nullopt;
  }
}

// 逐元素求值。覆盖 f32 与 f16 的 dense 张量，其余跳过。
// f16 是 B 路整算子级 IR 的张量类型，不覆盖它那条路的常量链折不了。
static Attribute evalEltwise(EltwiseKind kind, Attribute lhs, Attribute rhs,
                              Type resultTy) {
  auto lDense = dyn_cast<DenseElementsAttr>(lhs);
  auto rDense = dyn_cast<DenseElementsAttr>(rhs);
  if (!lDense || !rDense || lDense.getType() != rDense.getType())
    return {};
  // 结果类型与操作数不同时，按操作数类型折出会让下游类型对不上。
  if (resultTy != lDense.getType())
    return {};
  auto ty = dyn_cast<RankedTensorType>(lDense.getType());
  if (!ty || ty.getNumElements() > kMaxFoldElements)
    return {};

  auto elemTy = dyn_cast<FloatType>(ty.getElementType());
  if (!elemTy || (!elemTy.isF32() && !elemTy.isF16()))
    return {};
  SmallVector<APFloat> out;
  for (auto [l, r] : llvm::zip(lDense.getValues<APFloat>(),
                               rDense.getValues<APFloat>())) {
    auto v = evalFloat(kind, l, r);
    if (!v)
      return {};
    out.push_back(*v);
  }
  return DenseFPElementsAttr::get(ty, out);
}

// 激活只覆盖能在编译期算准的几种，其余不折。f16 与 f32 都覆盖。
static Attribute evalLut(ActivationKind kind, Attribute src, Type resultTy) {
  auto dense = dyn_cast<DenseElementsAttr>(src);
  if (!dense)
    return {};
  // 结果类型与输入不同意味着有类型转换，求值看不到，不折。
  if (resultTy != dense.getType())
    return {};
  auto ty = dyn_cast<RankedTensorType>(dense.getType());
  auto elemTy = ty ? dyn_cast<FloatType>(ty.getElementType()) : nullptr;
  if (!ty || !elemTy || (!elemTy.isF32() && !elemTy.isF16()) ||
      ty.getNumElements() > kMaxFoldElements)
    return {};
  SmallVector<APFloat> out;
  for (APFloat v : dense.getValues<APFloat>()) {
    switch (kind) {
    case ActivationKind::Relu:
      if (v.isNegative())
        v = APFloat::getZero(v.getSemantics());
      break;
    case ActivationKind::Exp:
    case ActivationKind::Sqrt:
    case ActivationKind::Reciprocal:
    case ActivationKind::Silu: {
      if (kind == ActivationKind::Sqrt && v.isNegative())
        return {};
      if (kind == ActivationKind::Reciprocal && v.isZero())
        return {};
      // 用 float 算再转回原语义：f16 没有直接的这些运算。
      float f = v.convertToFloat();
      float r = kind == ActivationKind::Exp        ? std::exp(f)
                : kind == ActivationKind::Sqrt      ? std::sqrt(f)
                : kind == ActivationKind::Reciprocal ? 1.0f / f
                                                     : f / (1.0f + std::exp(-f));
      APFloat back(r);
      bool losesInfo = false;
      back.convert(v.getSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
      v = back;
      break;
    }
    default:
      return {};
    }
    out.push_back(v);
  }
  return DenseFPElementsAttr::get(ty, out);
}

// 相位记录整份都只记账：index 是序号，bytes 是缓冲大小，unit 是执行块，
// forceConsecutive 说的是不能重排，reads 记的是消费哪些更早的相位。
// 四项都不改变运算。
static bool phasesAreBookkeeping(std::optional<ArrayAttr> phases) {
  if (!phases)
    return true;
  for (Attribute attr : *phases)
    if (!dyn_cast<PhaseSpecAttr>(attr))
      return false;
  return true;
}

// fpsu 在两种浮点模式且按默认粒度时是硬件的记账配置，不改数值。
// floating_point_32 是 32 位累加路径的记账，求值用操作数自己的类型。
// 定点模式和改过的粒度会把结果重新定标，求值看不到那份定标，不放行。
static bool fpsuIsBookkeeping(std::optional<FpsuSpecAttr> fpsu) {
  if (!fpsu)
    return true;
  auto mode = fpsu->getMode();
  bool floating = mode == FpsuMode::FloatingPoint ||
                  mode == FpsuMode::FloatingPoint32;
  return floating && fpsu->getSpc() && fpsu->getSpcAxis() == 1 &&
         !fpsu->getSpg() && fpsu->getSpgAxis() == -1 &&
         fpsu->getSpgGroupSize() == -1;
}

// 查表窗口与表选择描述的是表的排布。kind 已经决定了算哪个函数，
// 编译期按 kind 直接算，不经过表。specialOperators = 4 选的是倒数表，
// 与 kind = reciprocal 是同一个选择，两者一致时不挡住求值。
// 偶数或奇数寻址、以及表选择与 kind 对不上的，求值对不上，不放行。
static bool lutTableIsDefault(LutOp lut) {
  // table 是查表的数据本身，mode 选偶数或奇数寻址。两者都会让按 kind
  // 直接算的结果对不上，不放行。
  if (lut.getTable() || lut.getMode() != LutMode::Regular)
    return false;
  auto mode = lut.getActivationMode();
  if (mode && *mode != 0)
    return false;
  auto special = lut.getSpecialOperators();
  if (!special || *special == 0)
    return true;
  return *special == 4 && lut.getKind() == ActivationKind::Reciprocal;
}

// dense 整数张量的逐元素加减乘。真实 linear kernel 转 PIM 后，地址偏移
// 的常量全是这种形态，只折标量的话 A 路上一次都触发不了。
// 两侧必须同类型：形状不同意味着有广播，广播的对齐规则这里不猜。
static Attribute evalDenseInt(Operation *op, DenseElementsAttr lhs,
                              DenseElementsAttr rhs) {
  if (lhs.getType() != rhs.getType())
    return {};
  auto ty = dyn_cast<RankedTensorType>(lhs.getType());
  if (!ty || !ty.getElementType().isInteger(32) ||
      ty.getNumElements() > kMaxFoldElements)
    return {};
  SmallVector<APInt> out;
  for (auto [l, r] : llvm::zip(lhs.getValues<APInt>(), rhs.getValues<APInt>())) {
    if (isa<arith::AddIOp>(op))
      out.push_back(l + r);
    else if (isa<arith::SubIOp>(op))
      out.push_back(l - r);
    else if (isa<arith::MulIOp>(op))
      out.push_back(l * r);
    else
      return {};
  }
  return DenseIntElementsAttr::get(ty, out);
}

// 整数运算，覆盖标量与 dense 张量。
// 只折加减乘：除法有除零，移位有越界，两者都没有与参考语义对齐的结果。
static Attribute evalArith(Operation *op) {
  auto lval = constAttrOf(op->getOperand(0));
  auto rval = constAttrOf(op->getOperand(1));
  if (!lval || !rval)
    return {};
  auto lDense = dyn_cast<DenseElementsAttr>(lval);
  auto rDense = dyn_cast<DenseElementsAttr>(rval);
  if (lDense && rDense)
    return evalDenseInt(op, lDense, rDense);
  auto li = dyn_cast<IntegerAttr>(lval);
  auto ri = dyn_cast<IntegerAttr>(rval);
  if (!li || !ri || li.getType() != ri.getType())
    return {};
  int64_t l = li.getInt(), r = ri.getInt(), v = 0;
  if (isa<arith::AddIOp>(op))
    v = l + r;
  else if (isa<arith::SubIOp>(op))
    v = l - r;
  else if (isa<arith::MulIOp>(op))
    v = l * r;
  else
    return {};
  return IntegerAttr::get(li.getType(), v);
}

// datapath 描述操作数在相加前的定标。缩放规格、kantor 块和分组反量化
// 会改变数值，按 kind 直接算会对不上，不放行。模式本身只选数值格式，
// 求值用操作数自己的类型，不在这里排除。
static bool datapathIsPlain(DatapathAttr datapath) {
  return !datapath.getScaleSpec() && !datapath.getKantorBlocks() &&
         !datapath.getGroupDequantAccum() && datapath.getGroupSize() == 0;
}

struct TritonPIMConstantFoldPass
    : public mlir::triton::pim::impl::TritonPIMConstantFoldBase<TritonPIMConstantFoldPass> {
  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) { candidates.push_back(op); });

    for (Operation *op : candidates) {
      if (op->use_empty())
        continue;
      Attribute result;
      if (auto eltwise = dyn_cast<EltwiseOp>(op)) {
        if (eltwise.getOperands().size() != 2)
          continue;
        // 这些属性会改变结果：折入的激活、池化、半旋转、逐槽定标、
        // 收缩编码。求值只看 kind，带了就不折。
        // 三处例外只记账、不改数值：straightforward 说明结果原样进入下一相；
        // phases 只记录相位序号和缓冲大小；默认浮点 fpsu 是硬件的记账配置。
        // 其余取值会把结果折进别的链或重新定标，求值看不到，仍然不折。
        auto combine = eltwise.getCombineMode();
        if (eltwise.getActivation() || eltwise.getFusedPool() ||
            eltwise.getRotateHalf() ||
            (combine && *combine != CombineMode::Straightforward) ||
            eltwise.getContraction() || eltwise.getPerSlotDatapath() ||
            eltwise.getRhsDatapath() ||
            !fpsuIsBookkeeping(eltwise.getFpsu()) ||
            eltwise.getKantor() ||
            !phasesAreBookkeeping(eltwise.getPhases()) ||
            !datapathIsPlain(eltwise.getDatapath()))
          continue;
        auto l = constAttrOf(eltwise.getOperands()[0]);
        auto r = constAttrOf(eltwise.getOperands()[1]);
        if (!l || !r)
          continue;
        result = evalEltwise(eltwise.getKind(), l, r, eltwise.getType());
      } else if (auto lut = dyn_cast<LutOp>(op)) {
        // 缩放、斜率、截断在 kind 之外改变结果。查表窗口和表选择是表的
        // 排布，编译期按 kind 直接算，不经过表，默认值不挡住求值。
        // 相位序号和默认浮点 fpsu 只记账，与 eltwise 同一条口径。
        if (lut.getFpsuScale() || lut.getAlpha() || lut.getClipMin() ||
            lut.getClipMax() || !lutTableIsDefault(lut) ||
            !fpsuIsBookkeeping(lut.getFpsu()) || lut.getKantor() ||
            !phasesAreBookkeeping(lut.getPhases()))
          continue;
        auto src = constAttrOf(lut.getSrc());
        if (!src)
          continue;
        result = evalLut(lut.getKind(), src, lut.getType());
      } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
        result = evalArith(op);
      }
      if (!result)
        continue;
      OpBuilder builder(op);
      auto typed = cast<TypedAttr>(result);
      auto constant = builder.create<arith::ConstantOp>(op->getLoc(), typed.getType(), typed);
      op->replaceAllUsesWith(constant);
      op->erase();
    }
  }
};

} // namespace
