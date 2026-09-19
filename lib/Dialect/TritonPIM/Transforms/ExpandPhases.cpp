//===----------------------------------------------------------------------===//
//
// Expands three opaque operator-level ops into the phase SSA the target
// executes, one engine traversal per phase:
//
//   pim.quantize {dynamic}  ->  4 phases (absmax, identity-scale, reciprocal, fp2int)
//   pim.softmax             ->  5 phases (max, exp, sum, reciprocal, mul)
//   pim.rope                ->  3 phases (mul-cos, mul-sin, add)
//
// Templates are op-local and hardcoded against the llama2 W4A8 reference
// artifact. Other operators are left alone. Each produced op carries
// `pim.phase-bytes` (logical buffer size); L2 offsets / Layer IDs are not
// assigned here.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMEXPANDPHASES
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

// Logical-buffer size, matching contracts.gml_names / prepare_out "phase-bytes".
static constexpr StringRef kPhaseBytesAttr = "pim.phase-bytes";
// RoPE's three phases share an intermediate buffer and must not be reordered.
static constexpr StringRef kForceConsecutiveAttr = "pim.force-consecutive";
// 0-based hardware phase index. Only ops that are a real engine traversal carry
// it: layout reshapes and the softmax stabilization (an FPSU affine folded into
// the exp phase) are helpers, not phases, so a consumer can count phases by
// counting distinct values of this attribute rather than counting ops.
static constexpr StringRef kPhaseAttr = "pim.phase";

static int64_t elementCount(RankedTensorType ty) {
  int64_t n = 1;
  for (int64_t d : ty.getShape())
    n *= d;
  return n;
}

static int64_t elementBytes(Type elem) {
  if (elem.isF16() || elem.isBF16())
    return 2;
  if (elem.isF32())
    return 4;
  if (elem.isInteger(8) || elem.isInteger(1))
    return 1;
  if (elem.isInteger(16))
    return 2;
  if (elem.isInteger(32))
    return 4;
  return 1;
}

static void setPhaseBytes(Operation *op, RankedTensorType ty) {
  op->setAttr(kPhaseBytesAttr,
              IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                               elementCount(ty) * elementBytes(ty.getElementType())));
}

static DatapathAttr floatDatapath(MLIRContext *ctx) {
  return DatapathAttr::get(ctx, NumericMode::FloatingPoint,
                           NumericMode::FloatingPoint,
                           /*scaleSpec=*/QuantSpecAttr{},
                           /*kantorBlocks=*/ArrayAttr{});
}

static DatapathAttr datapathWithKantor(MLIRContext *ctx, KantorMode mode) {
  auto block = KantorBlockAttr::get(ctx, "A", mode, /*spec=*/QuantSpecAttr{});
  return DatapathAttr::get(ctx, NumericMode::FloatingPoint,
                           NumericMode::FloatingPoint,
                           /*scaleSpec=*/QuantSpecAttr{},
                           ArrayAttr::get(ctx, {block}));
}

static void setPhase(Operation *op, int64_t phase) {
  op->setAttr(kPhaseAttr, IntegerAttr::get(
                              IntegerType::get(op->getContext(), 64), phase));
}

static LutOp emitLut(OpBuilder &b, Location loc, Value src,
                     ActivationKind kind, FunctionalUnit unit,
                     RankedTensorType resultTy, int64_t phase) {
  auto op = b.create<LutOp>(loc, resultTy, src, /*table=*/Value{}, kind,
                            LutMode::Regular, /*alpha=*/nullptr,
                            /*clipMin=*/nullptr, /*clipMax=*/nullptr,
                            FunctionalUnitAttr::get(b.getContext(), unit));
  setPhaseBytes(op, resultTy);
  setPhase(op, phase);
  return op;
}

static ReduceAxisOp emitReduce(OpBuilder &b, Location loc, Value src,
                               EltwiseKind kind, int64_t axis,
                               FunctionalUnit unit, RankedTensorType resultTy,
                               int64_t phase) {
  auto op = b.create<ReduceAxisOp>(
      loc, resultTy, src, EltwiseKindAttr::get(b.getContext(), kind),
      b.getI64IntegerAttr(axis),
      FunctionalUnitAttr::get(b.getContext(), unit));
  setPhaseBytes(op, resultTy);
  setPhase(op, phase);
  return op;
}

// Collapse the last axis of `srcTy` by `groupSize`, keeping rank.
// [1, 4096] / 128 -> [1, 32].
static RankedTensorType groupedType(RankedTensorType srcTy, int64_t groupSize) {
  SmallVector<int64_t> shape(srcTy.getShape().begin(), srcTy.getShape().end());
  int64_t last = shape.back();
  shape.back() = last / groupSize;
  return RankedTensorType::get(shape, srcTy.getElementType());
}

// [..., N] -> [..., groups, groupSize] so reduce_axis can collapse groupSize.
static RankedTensorType grouped3Type(RankedTensorType srcTy, int64_t groupSize) {
  SmallVector<int64_t> shape(srcTy.getShape().begin(), srcTy.getShape().end());
  int64_t last = shape.back();
  shape.back() = last / groupSize;
  shape.push_back(groupSize);
  return RankedTensorType::get(shape, srcTy.getElementType());
}

static RankedTensorType reducedLast(RankedTensorType ty) {
  SmallVector<int64_t> shape(ty.getShape().begin(), ty.getShape().end());
  shape.back() = 1;
  return RankedTensorType::get(shape, ty.getElementType());
}

static RankedTensorType sameShape(RankedTensorType ty, Type elem) {
  return RankedTensorType::get(ty.getShape(), elem);
}

// I64Attr getters return uint64_t; negative axes arrive as huge positives.
static int64_t signedAxis(uint64_t raw, int64_t rank) {
  int64_t axis = static_cast<int64_t>(raw);
  if (axis < 0)
    axis += rank;
  return axis;
}

//===----------------------------------------------------------------------===//
// Dynamic quantize: 4 phases, p1 and p2 fan out from p0.
//===----------------------------------------------------------------------===//

static LogicalResult expandQuantize(QuantizeOp op) {
  if (!op.getDynamic())
    return success();

  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto spec = op.getSpec();
  if (spec.getGranularity() != QuantGranularity::PerGroup)
    return op.emitOpError(
        "dynamic quantize expansion requires a per_group spec");
  int64_t groupSize = spec.getGroupSize();
  if (groupSize <= 0 || srcTy.getShape().back() % groupSize != 0)
    return op.emitOpError("axis extent is not divisible by groupSize");

  OpBuilder b(op);
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  Value src = op.getSrc();

  auto g3Ty = grouped3Type(srcTy, groupSize);
  auto reshaped = b.create<::mlir::triton::pim::ReshapeOp>(loc, g3Ty, src);

  auto redTy = reducedLast(g3Ty);
  auto p0 = emitReduce(b, loc, reshaped.getResult(), EltwiseKind::AbsMax,
                       /*axis=*/g3Ty.getRank() - 1, FunctionalUnit::VPU, redTy,
                       /*phase=*/0);

  auto groupsTy = groupedType(srcTy, groupSize);
  auto p0Flat = b.create<::mlir::triton::pim::ReshapeOp>(loc, groupsTy,
                                                          p0.getResult());
  setPhaseBytes(p0Flat, groupsTy);

  // p1: identity LUT (÷256 lives in the FPSU scale, not here).
  auto p1 = emitLut(b, loc, p0Flat.getResult(), ActivationKind::Relu,
                    FunctionalUnit::CSTL, groupsTy, /*phase=*/1);
  p1->setAttr("activation_mode", b.getI64IntegerAttr(1));

  // p2: reciprocal of p0 (fan-out from p0, not from p1).
  auto p2 = emitLut(b, loc, p0Flat.getResult(), ActivationKind::Reciprocal,
                    FunctionalUnit::CSTL, groupsTy, /*phase=*/2);

  auto i8Ty = sameShape(srcTy, b.getIntegerType(8));
  auto qSpec = QuantSpecAttr::get(ctx, QuantGranularity::PerGroup,
                                  spec.getAxis(), groupSize,
                                  DataExtension::Signed);
  auto p3 = b.create<QuantizeOp>(
      loc, i8Ty, src, p2.getResult(), /*zeroPoint=*/Value{}, qSpec,
      /*dynamic=*/UnitAttr{},
      FunctionalUnitAttr::get(ctx, FunctionalUnit::CSTL));
  p3->setAttr(
      "datapath",
      datapathWithKantor(ctx, KantorMode::Fp2IntConverter));
  setPhaseBytes(p3, i8Ty);
  setPhase(p3, 3);

  op.getResult().replaceAllUsesWith(p3.getResult());
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Softmax: 5 phases. p1 (exp) and p4 (mul) fan out from earlier phases.
//===----------------------------------------------------------------------===//

static LogicalResult expandSoftmax(SoftmaxOp op) {
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  int64_t axis = signedAxis(op.getAxis(), srcTy.getRank());
  if (axis < 0 || axis >= srcTy.getRank())
    return op.emitOpError("axis out of range");

  OpBuilder b(op);
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  Value src = op.getSrc();
  auto f16 = b.getF16Type();
  auto unitCstl = FunctionalUnitAttr::get(ctx, FunctionalUnit::CSTL);

  SmallVector<int64_t> reducedShape(srcTy.getShape().begin(),
                                    srcTy.getShape().end());
  reducedShape[axis] = 1;
  auto maxTy = RankedTensorType::get(reducedShape, f16);
  auto p0 = emitReduce(b, loc, src, EltwiseKind::Max, axis, FunctionalUnit::VPU,
                       maxTy, /*phase=*/0);

  // Numerical stabilization: x - max. This is the FPSU affine folded into the
  // exp phase on hardware, not a phase of its own -- hence no `pim.phase`.
  auto shifted = b.create<EltwiseOp>(
      loc, srcTy, src, p0.getResult(),
      EltwiseKindAttr::get(ctx, EltwiseKind::Sub), floatDatapath(ctx),
      DatapathAttr{}, ActSpecAttr{}, PoolSpecAttr{}, unitCstl);

  auto expTy = sameShape(srcTy, f16);
  auto p1 = emitLut(b, loc, shifted.getResult(), ActivationKind::Exp,
                    FunctionalUnit::CSTL, expTy, /*phase=*/1);

  auto p2 = emitReduce(b, loc, p1.getResult(), EltwiseKind::Add, axis,
                       FunctionalUnit::VPU, maxTy, /*phase=*/2);

  auto p3 = emitLut(b, loc, p2.getResult(), ActivationKind::Reciprocal,
                    FunctionalUnit::CSTL, maxTy, /*phase=*/3);

  auto p4 = b.create<EltwiseOp>(
      loc, expTy, p1.getResult(), p3.getResult(),
      EltwiseKindAttr::get(ctx, EltwiseKind::Mul), floatDatapath(ctx),
      DatapathAttr{}, ActSpecAttr{}, PoolSpecAttr{}, unitCstl);
  setPhaseBytes(p4, expTy);
  setPhase(p4, 4);

  op.getResult().replaceAllUsesWith(p4.getResult());
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// RoPE: 3 consecutive eltwise phases. rotate_half is a layout annotation on
// the sin path (the hardware folds it into the Kantor block); materializing
// split/neg/concat is the orchestrator's job.
//===----------------------------------------------------------------------===//

static LogicalResult expandRope(RopeOp op) {
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  OpBuilder b(op);
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();

  auto mulPath = datapathWithKantor(ctx, KantorMode::ElementwiseMulFp16);
  auto addPath = floatDatapath(ctx);
  auto unit = FunctionalUnitAttr::get(ctx, FunctionalUnit::CSTL);

  auto mulCos = b.create<EltwiseOp>(
      loc, srcTy, op.getSrc(), op.getCos(),
      EltwiseKindAttr::get(ctx, EltwiseKind::Mul), mulPath, DatapathAttr{},
      ActSpecAttr{}, PoolSpecAttr{}, unit);
  mulCos->setAttr(kForceConsecutiveAttr, b.getUnitAttr());
  setPhaseBytes(mulCos, srcTy);
  setPhase(mulCos, 0);

  auto mulSin = b.create<EltwiseOp>(
      loc, srcTy, op.getSrc(), op.getSin(),
      EltwiseKindAttr::get(ctx, EltwiseKind::Mul), mulPath, DatapathAttr{},
      ActSpecAttr{}, PoolSpecAttr{}, unit);
  mulSin->setAttr(kForceConsecutiveAttr, b.getUnitAttr());
  mulSin->setAttr("pim.rotate-half", b.getUnitAttr());
  setPhaseBytes(mulSin, srcTy);
  setPhase(mulSin, 1);

  auto add = b.create<EltwiseOp>(
      loc, srcTy, mulCos.getResult(), mulSin.getResult(),
      EltwiseKindAttr::get(ctx, EltwiseKind::Add), addPath, DatapathAttr{},
      ActSpecAttr{}, PoolSpecAttr{}, unit);
  add->setAttr(kForceConsecutiveAttr, b.getUnitAttr());
  setPhaseBytes(add, srcTy);
  setPhase(add, 2);

  op.getResult().replaceAllUsesWith(add.getResult());
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TritonPIMExpandPhasesPass
    : public mlir::triton::pim::impl::TritonPIMExpandPhasesBase<
          TritonPIMExpandPhasesPass> {
  using mlir::triton::pim::impl::TritonPIMExpandPhasesBase<
      TritonPIMExpandPhasesPass>::TritonPIMExpandPhasesBase;

  void runOnOperation() override {
    // Collect first: expansion erases ops.
    SmallVector<QuantizeOp> dqs;
    SmallVector<SoftmaxOp> sms;
    SmallVector<RopeOp> ropes;
    getOperation().walk([&](Operation *op) {
      if (auto q = dyn_cast<QuantizeOp>(op))
        dqs.push_back(q);
      else if (auto s = dyn_cast<SoftmaxOp>(op))
        sms.push_back(s);
      else if (auto r = dyn_cast<RopeOp>(op))
        ropes.push_back(r);
    });

    for (auto q : dqs)
      if (failed(expandQuantize(q)))
        return signalPassFailure();
    for (auto s : sms)
      if (failed(expandSoftmax(s)))
        return signalPassFailure();
    for (auto r : ropes)
      if (failed(expandRope(r)))
        return signalPassFailure();
  }
};

} // namespace
