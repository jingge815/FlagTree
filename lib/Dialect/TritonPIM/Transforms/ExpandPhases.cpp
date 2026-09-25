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
// `#pim.phase_spec` (its `bytes` field is the logical buffer size); L2 offsets
// and Layer IDs are not assigned here.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMEXPANDPHASES
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

// Kantor mode as the target's layer card spells it, NOT the dialect enum
// ordinal: the RoPE multiplies want 5 and the K-path add wants 3, while the
// Q-path add wants 0. A consumer cannot recover those from `#pim.datapath`
// because several distinct card values share one dialect KantorMode. The
// per-chain tail value now rides on `pim.rope`'s `tailCardValue` attribute
// (it used to be a bare `pim.kantor-mode` string looked up by name here).

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
  if (elem.isF64())
    return 8;
  if (elem.isInteger(8) || elem.isInteger(1))
    return 1;
  if (elem.isInteger(16))
    return 2;
  if (elem.isInteger(32))
    return 4;
  if (elem.isInteger(64))
    return 8;
  llvm::report_fatal_error(
      "elementBytes: unhandled element type; inventing a width would "
      "silently mis-size a phase buffer");
}

// Mark `op` as owning one engine traversal. `reads` names the earlier phases it
// consumes, which is how the fan-out survives: dynamic quantization's phases 1
// and 2 both read phase 0, and a consumer that assumed a chain would put the
// reciprocal off by 256x.
static void setPhaseSpec(Operation *op, int64_t index, RankedTensorType ty,
                         FunctionalUnit unit, bool consecutive = false,
                         ArrayRef<int64_t> reads = {}) {
  MLIRContext *ctx = op->getContext();
  SmallVector<Attribute> readAttrs;
  for (int64_t read : reads)
    readAttrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), read));

  auto spec = PhaseSpecAttr::get(
      ctx, index, elementCount(ty) * elementBytes(ty.getElementType()), unit,
      consecutive,
      readAttrs.empty() ? ArrayAttr{} : ArrayAttr::get(ctx, readAttrs));
  op->setAttr("phases", ArrayAttr::get(ctx, {spec}));
  // Whether this op is phase-shaped, said outright rather than inferred. The
  // target's revision field is not a usable proxy: 37 nodes carry it while 69
  // are phase-shaped, because the softmax nodes carry none. This marker and the
  // `phases` attribute are two spellings of one fact, and
  // `pim-verify-gml-contract` holds them to each other in both directions.
  op->setAttr("isPhased", UnitAttr::get(ctx));
}

static DatapathAttr floatDatapath(MLIRContext *ctx) {
  return DatapathAttr::get(ctx, NumericMode::FloatingPoint,
                           NumericMode::FloatingPoint,
                           /*scaleSpec=*/QuantSpecAttr{},
                           /*kantorBlocks=*/ArrayAttr{},
                           /*groupDequantAccum=*/false, /*groupSize=*/0);
}

static DatapathAttr datapathWithKantor(MLIRContext *ctx, KantorMode mode) {
  auto block = KantorBlockAttr::get(ctx, "A", mode, /*spec=*/QuantSpecAttr{});
  return DatapathAttr::get(ctx, NumericMode::FloatingPoint,
                           NumericMode::FloatingPoint,
                           /*scaleSpec=*/QuantSpecAttr{},
                           ArrayAttr::get(ctx, {block}),
                           /*groupDequantAccum=*/false, /*groupSize=*/0);
}

// Still a bare attribute: the target's field it would describe has no hits in
// the reference artifact, so it is carried but not yet a first-class encoding.
// The addressing card a phase walks with. It rides alongside the purpose
// rather than claiming one of its own -- a strided walk and a scalar walk are
// the same kind of layout decision. `cardValue` is 0 for "no walk of its own".
static void setTransposePurpose(Operation *op, TransposePurpose purpose,
                                int64_t cardValue) {
  op->setAttr("transposePurpose", TransposePurposeAttr::get(
                                      op->getContext(), purpose, cardValue));
}

static void setI64(Operation *op, StringRef name, int64_t value) {
  op->setAttr(name, IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                     value));
}

// The activation unit's addressing window: which segment of the table an input
// lands in. The three are one triple -- a window without its exponent bounds
// addresses the wrong segment, silently. Set through the op rather than the
// typed accessor because a fused activation carries the window on its
// *producer* (a matmul), which does not declare these itself.
static void setLutWindow(Operation *op, int64_t minExp, int64_t maxExp,
                         int64_t mantisa) {
  setI64(op, "flpMinExp", minExp);
  setI64(op, "flpMaxExp", maxExp);
  setI64(op, "flpMantisa", mantisa);
}

// Which of the two tables the unit reads. 0 is the regular table, 4 the
// reciprocal one; it is a flag of its own rather than something derived from
// `kind`, so a reader that only looks at the kind cannot tell them apart.
static void setLutTable(Operation *op, int64_t activationMode,
                        int64_t specialOperators) {
  setI64(op, "activationMode", activationMode);
  setI64(op, "specialOperators", specialOperators);
}

// The fixed-point unit's setting for this phase. The two card values in the
// reference are 1 for the 16-bit phase pipelines and 2 for the 32-bit
// accumulator path, so the mode is what gets stored and the card value is
// recovered from it rather than the other way round.
//
// `spec` is the quantization decision this phase belongs to, when it has one:
// the per-channel / per-group fields are that decision projected onto this
// unit, so they are derived from it rather than spelled out. The phases of a
// dynamic quantization are the ones that have a spec; the softmax, RoPE and
// normalization phases carry the unit's defaults, which say the same thing for
// a spec-shaped view of them (per-channel on, grouping off).
static void setFpsuSpec(Operation *op, int64_t card,
                        QuantSpecAttr spec = QuantSpecAttr{}) {
  FpsuMode mode =
      card == 2 ? FpsuMode::FloatingPoint32 : FpsuMode::FloatingPoint;
  SpcSpg axes = spec ? deriveHardwareAxes(spec, HardwareBlock::Fpsu)
                     : SpcSpg{/*spc=*/true, /*spcAxis=*/1, /*spg=*/false,
                              /*spgAxis=*/-1, /*spgGroupSize=*/-1};
  op->setAttr("fpsu",
              FpsuSpecAttr::get(op->getContext(), mode, axes.spc,
                                axes.spcAxis, axes.spg, axes.spgAxis,
                                axes.spgGroupSize));
}

// The elementwise-multiply unit's setting. `card` is the target's own field
// value and is kept even where it disagrees with the mode, because several card
// values share one dialect mode and a consumer cannot recover which was meant.
static void setKantorSpec(Operation *op, KantorMode mode, int64_t card) {
  op->setAttr("kantor",
              KantorSpecAttr::get(op->getContext(), mode, card, ArrayAttr{}));
}

static LutOp emitLut(OpBuilder &b, Location loc, Value src,
                     ActivationKind kind, FunctionalUnit unit,
                     RankedTensorType resultTy, int64_t phase,
                     bool consecutive = false, ArrayRef<int64_t> reads = {}) {
  auto op = b.create<LutOp>(
      loc, resultTy, /*phases=*/ArrayAttr{}, /*isPhased=*/false, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, src, /*table=*/Value{}, kind, LutMode::Regular,
      /*alpha=*/nullptr, /*clipMin=*/nullptr, /*clipMax=*/nullptr,
      /*flpMinExp=*/nullptr, /*flpMaxExp=*/nullptr, /*flpMantisa=*/nullptr,
      /*activationMode=*/nullptr, /*specialOperators=*/nullptr,
      /*fpsuScale=*/FloatAttr{}, /*transposePurpose=*/TransposePurposeAttr{},
      FunctionalUnitAttr::get(b.getContext(), unit));
  setPhaseSpec(op, phase, resultTy, unit, consecutive, reads);
  return op;
}

static ReduceAxisOp emitReduce(OpBuilder &b, Location loc, Value src,
                               EltwiseKind kind, int64_t axis,
                               FunctionalUnit unit, RankedTensorType resultTy,
                               int64_t phase, bool consecutive = false,
                               ArrayRef<int64_t> reads = {}) {
  auto op = b.create<ReduceAxisOp>(
      loc, resultTy, /*phases=*/ArrayAttr{}, /*isPhased=*/UnitAttr{}, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, src, EltwiseKindAttr::get(b.getContext(), kind),
      b.getI64IntegerAttr(axis),
      FunctionalUnitAttr::get(b.getContext(), unit));
  setPhaseSpec(op, phase, resultTy, unit, consecutive, reads);
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

// `gml_bridge/phase_data.py` 的 `DQ_PHASE1_SCALE`。两处必须同值，
// `tests/test_phase_data.py` 从产物 IR 里读回来对着比。
static constexpr double kDqPhase1Scale = 1.0 / 256.0;

static LogicalResult expandDynamicQuant(Operation *op, Value src,
                                        QuantSpecAttr spec) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  if (spec.getGranularity() != QuantGranularity::PerGroup)
    return op->emitOpError(
        "dynamic quantize expansion requires a per_group spec");
  int64_t groupSize = spec.getGroupSize();
  if (groupSize <= 0 || srcTy.getShape().back() % groupSize != 0)
    return op->emitOpError("axis extent is not divisible by groupSize");

  OpBuilder b(op);
  Location loc = op->getLoc();
  MLIRContext *ctx = op->getContext();

  // p0: one absmax per quantization group, on the pooling unit. A grouped
  // reduction, not a tile-level reduce with shape surgery around it: the two
  // reshapes existed only to make `pim.reduce_axis` stand in for a reduction
  // along groups, which is not what that op does.
  auto groupsTy = groupedType(srcTy, groupSize);
  // The per-channel and per-group flags are this unit's view of `spec`, the
  // quantization decision the whole phase chain implements -- derived, not
  // spelled out, so that a spec of another shape lands on the right axes.
  SpcSpg poolingAxes = deriveHardwareAxes(spec, HardwareBlock::Pooling);
  auto p0 = b.create<GlobalPoolOp>(
      loc, groupsTy, /*phases=*/ArrayAttr{}, /*isPhased=*/false, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, src, GlobalPoolKind::AbsMax,
      static_cast<uint64_t>(groupSize), poolingAxes.spc, poolingAxes.spcAxis,
      poolingAxes.spg, poolingAxes.spgAxis, PoolDType::FloatingPoint,
      FunctionalUnitAttr::get(ctx, FunctionalUnit::Pooling));
  setPhaseSpec(p0, 0, groupsTy, FunctionalUnit::Pooling);

  // p1: identity LUT (÷256 lives in the FPSU scale, not here).
  auto p1 = emitLut(b, loc, p0.getResult(), ActivationKind::Identity,
                    FunctionalUnit::Activation, groupsTy, /*phase=*/1,
                    /*consecutive=*/false, /*reads=*/{0});
  setLutWindow(p1, /*min=*/10, /*max=*/17, /*mantisa=*/3);
  setLutTable(p1, /*activationMode=*/1, /*specialOperators=*/0);
  setFpsuSpec(p1, 1, spec);
  // The x1/256 the hardware applies after the identity table. It lives in the
  // FPSU's scaling buffer, not in the table, so it cannot be derived from the
  // window or the activation kind -- it has to be carried here or the lowering
  // silently computes a value 256x the declared one.
  //
  // Pinned to `gml_bridge/phase_data.py:DQ_PHASE1_SCALE` by
  // `tests/test_phase_data.py` reading it back out of the emitted IR: the
  // constant exists in two languages because the C side has no Python, and the
  // test is what keeps the two copies from drifting apart.
  p1->setAttr("fpsuScale",
              b.getF64FloatAttr(kDqPhase1Scale));

  // p2: reciprocal of p0 (fan-out from p0, not from p1).
  auto p2 = emitLut(b, loc, p0.getResult(), ActivationKind::Reciprocal,
                    FunctionalUnit::Activation, groupsTy, /*phase=*/2,
                    /*consecutive=*/false, /*reads=*/{0});
  setLutWindow(p2, /*min=*/15, /*max=*/15, /*mantisa=*/0);
  setLutTable(p2, /*activationMode=*/0, /*specialOperators=*/4);
  setFpsuSpec(p2, 1, spec);
  // The reciprocal phase walks the per-group scalars, so how it addresses them
  // depends on how many groups there are: a single group (the whole row is one
  // group, as for attention scores) is a scalar walk, several groups are a
  // strided one. Derived from groupSize, not hardcoded -- both cases occur in
  // one network and the card values differ (2 vs 1).
  setTransposePurpose(p2, TransposePurpose::LayoutReorder,
                      elementCount(groupsTy) <= 1 ? 2 : 1);

  auto i8Ty = sameShape(srcTy, b.getIntegerType(8));
  // The quantized activation this phase produces is an intermediate: it is
  // consumed by the next operator, not held as a weight and not a placeholder.
  // That is also the attribute's default, so it is spelled out here only
  // because the remaining two parameters have to be named anyway.
  // p3: 定点化。发独立 `pim.kantor`（方案 §5.3 要求卡值走 cardValue），
  // 不再用 `pim.quantize` + `kantor` 属性——那样这个 op 只有 ODS 没有发射点。
  // 相 SSA 与相位数都不变：还是同一处的第 3 相。
  auto kantorSpec = KantorSpecAttr::get(
      ctx, KantorMode::Fp2IntConverter, /*cardValue=*/3, /*blocks=*/ArrayAttr{});
  // `shift` 是这一相真正的定点移位量，方案 §5.3 定为 -8（左移 8 位 = ×256）。
  // 它必须**发出来**：只写在方案里而 IR 上是空的话，降级侧只能写死一个 256，
  // 改 shift 不会改变任何产物——那这个字段就是装饰。
  auto shift = IntegerAttr::get(IntegerType::get(ctx, 8), -8);
  auto p3 = b.create<KantorOp>(
      loc, i8Ty, /*phases=*/ArrayAttr{}, /*isPhased=*/UnitAttr{}, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, src, /*rhs=*/Value{}, /*scale=*/p2.getResult(),
      /*bias=*/Value{}, /*shift=*/shift,
      kantorSpec, FunctionalUnitAttr::get(ctx, FunctionalUnit::Kantor));
  // 不写 `datapath`：KantorOp 的 ODS 没有这个字段（A/B 两块写在 `spec` 里），
  // 挂上去只是个没人读、verifier 也查不到的裸属性名——降级侧读的是 `getSpec()`，
  // GML 侧的卡值来自 `spec.cardValue`。
  setPhaseSpec(p3, 3, i8Ty, FunctionalUnit::Kantor);
  // 卡值已经在构造参数 `kantorSpec` 里，不再另写一个同名属性：
  // KantorOp 的 ODS 字段就叫 spec，再写一份 kantor 会让同一个值出现两次。
  setFpsuSpec(p3, 1, spec);

  if (auto dq = dyn_cast<DynamicQuantOp>(op)) {
    dq.getResult().replaceAllUsesWith(p3.getResult());
    // `scale` 是这一链输出的量化 scale，方案 §5.6 的相位表把它定在**相 1**
    // （`lut identity`，值 = p0/256）。相 2 是倒数 `1/p0`，接过它是错的：
    // 下游按 `q * scale` 反量化会得到 `q/p0`，量级差 p0²/256。
    dq.getScale().replaceAllUsesWith(p1.getResult());
  } else {
    op->getResult(0).replaceAllUsesWith(p3.getResult());
  }
  op->erase();
  return success();
}

static LogicalResult expandQuantize(QuantizeOp op) {
  if (!op.getDynamic())
    return success();
  return expandDynamicQuant(op.getOperation(), op.getSrc(), op.getSpec());
}

static LogicalResult expandDynamicQuantOp(DynamicQuantOp op) {
  QuantSpecAttr spec = op.getSpec().value_or(QuantSpecAttr::get(
      op.getContext(), QuantGranularity::PerGroup,
      op.getAxis(), op.getGroupSize(), DataExtension::Signed,
      QuantRole::Intermediate, TypeAttr{}, ArrayAttr{},
      /*spc=*/true, /*spcAxis=*/1, /*spg=*/true, /*spgAxis=*/3,
      /*spgGroupSize=*/op.getGroupSize()));
  return expandDynamicQuant(op.getOperation(), op.getSrc(), spec);
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
  // 稳定化跑在定点定标块（它就是一次仿射），归一化乘跑在合成块。
  // 这两个以前都只能叫 `cstl`——六个块一个名字，成本抽取分不出遍历类型。
  auto unitFpsu = FunctionalUnitAttr::get(ctx, FunctionalUnit::FPSU);
  auto unitCombiner = FunctionalUnitAttr::get(ctx, FunctionalUnit::Combiner);

  SmallVector<int64_t> reducedShape(srcTy.getShape().begin(),
                                    srcTy.getShape().end());
  reducedShape[axis] = 1;
  auto maxTy = RankedTensorType::get(reducedShape, f16);
  auto p0 = emitReduce(b, loc, src, EltwiseKind::Max, axis, FunctionalUnit::VPU,
                       maxTy, /*phase=*/0);

  // Numerical stabilization: x - max. This is the FPSU affine folded into the
  // exp phase on hardware, not a phase of its own -- hence no `pim.phase`.
  auto shifted = b.create<EltwiseOp>(
      loc, srcTy, /*phases=*/ArrayAttr{}, /*isPhased=*/UnitAttr{}, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, ValueRange{src, p0.getResult()},
      EltwiseKindAttr::get(ctx, EltwiseKind::Sub), floatDatapath(ctx),
      DatapathAttr{}, /*perSlotDatapath=*/ArrayAttr{}, ActSpecAttr{},
      PoolSpecAttr{}, ContractionAttr{}, CombineModeAttr{}, /*rotateHalf=*/UnitAttr{}, /*transposePurpose=*/TransposePurposeAttr{}, /*sourceSubBlocks=*/ArrayAttr{}, /*sourceBroadcastSpec=*/BroadcastSpecAttr{}, unitFpsu);

  auto expTy = sameShape(srcTy, f16);
  auto p1 = emitLut(b, loc, shifted.getResult(), ActivationKind::Exp,
                    FunctionalUnit::Activation, expTy, /*phase=*/1);
  setLutWindow(p1, /*min=*/9, /*max=*/16, /*mantisa=*/3);
  setLutTable(p1, /*activationMode=*/0, /*specialOperators=*/0);
  setFpsuSpec(p1, 1);
  setTransposePurpose(p1, TransposePurpose::LayoutReorder, 1);

  // The sum is real fp32 on hardware, unlike the max: phase 0 stores an fp16
  // value (which is why its encoding is an fp16 bit pattern in the high half of
  // a four-byte slot), while the accumulated total is read onward in fp32.
  // Declaring it f16 here would narrow it on the way to the reciprocal, and the
  // reciprocal of a narrowed sum is off by the sum's own rounding -- about a
  // tenth of a percent, which is far larger than the arithmetic's own noise.
  auto sumTy = RankedTensorType::get(reducedShape, b.getF32Type());
  auto p2 = emitReduce(b, loc, p1.getResult(), EltwiseKind::Add, axis,
                       FunctionalUnit::VPU, sumTy, /*phase=*/2);
  setFpsuSpec(p2, 1);

  auto p3 = emitLut(b, loc, p2.getResult(), ActivationKind::Reciprocal,
                    FunctionalUnit::Activation, maxTy, /*phase=*/3);
  setLutWindow(p3, /*min=*/15, /*max=*/15, /*mantisa=*/0);
  setLutTable(p3, /*activationMode=*/0, /*specialOperators=*/4);
  setFpsuSpec(p3, 2);
  setTransposePurpose(p3, TransposePurpose::LayoutReorder, 2);

  auto p4 = b.create<EltwiseOp>(
      loc, expTy, /*phases=*/ArrayAttr{}, /*isPhased=*/UnitAttr{}, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, ValueRange{p1.getResult(), p3.getResult()},
      EltwiseKindAttr::get(ctx, EltwiseKind::Mul), floatDatapath(ctx),
      DatapathAttr{}, /*perSlotDatapath=*/ArrayAttr{}, ActSpecAttr{},
      PoolSpecAttr{}, ContractionAttr{}, CombineModeAttr{}, /*rotateHalf=*/UnitAttr{},
      /*transposePurpose=*/TransposePurposeAttr{}, /*sourceSubBlocks=*/ArrayAttr{}, /*sourceBroadcastSpec=*/BroadcastSpecAttr{}, unitCombiner);
  setPhaseSpec(p4, 4, expTy, FunctionalUnit::Combiner, /*consecutive=*/false,
               /*reads=*/{1, 3});

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
  auto unit = FunctionalUnitAttr::get(ctx, FunctionalUnit::Kantor);

  auto mulCos = b.create<EltwiseOp>(
      loc, srcTy, /*phases=*/ArrayAttr{}, /*isPhased=*/UnitAttr{}, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, ValueRange{op.getSrc(), op.getCos()},
      EltwiseKindAttr::get(ctx, EltwiseKind::Mul), mulPath, DatapathAttr{},
      /*perSlotDatapath=*/ArrayAttr{}, ActSpecAttr{}, PoolSpecAttr{},
      ContractionAttr{}, CombineModeAttr{}, /*rotateHalf=*/UnitAttr{},
      /*transposePurpose=*/TransposePurposeAttr{}, /*sourceSubBlocks=*/ArrayAttr{}, /*sourceBroadcastSpec=*/BroadcastSpecAttr{}, unit);
  setPhaseSpec(mulCos, 0, srcTy, FunctionalUnit::Kantor, /*consecutive=*/true);
  setKantorSpec(mulCos, KantorMode::ElementwiseMulFp16, /*card=*/5);
  setFpsuSpec(mulCos, 1);

  auto mulSin = b.create<EltwiseOp>(
      loc, srcTy, /*phases=*/ArrayAttr{}, /*isPhased=*/UnitAttr{}, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, ValueRange{op.getSrc(), op.getSin()},
      EltwiseKindAttr::get(ctx, EltwiseKind::Mul), mulPath, DatapathAttr{},
      /*perSlotDatapath=*/ArrayAttr{}, ActSpecAttr{}, PoolSpecAttr{},
      ContractionAttr{}, CombineModeAttr{}, /*rotateHalf=*/UnitAttr{},
      /*transposePurpose=*/TransposePurposeAttr{}, /*sourceSubBlocks=*/ArrayAttr{}, /*sourceBroadcastSpec=*/BroadcastSpecAttr{}, unit);
  mulSin.setRotateHalf(true);
  setPhaseSpec(mulSin, 1, srcTy, FunctionalUnit::Kantor, /*consecutive=*/true);
  setKantorSpec(mulSin, KantorMode::ElementwiseMulFp16, /*card=*/5);
  setFpsuSpec(mulSin, 1);

  auto add = b.create<EltwiseOp>(
      loc, srcTy, /*phases=*/ArrayAttr{}, /*isPhased=*/UnitAttr{}, /*fpsu=*/FpsuSpecAttr{}, /*kantor=*/KantorSpecAttr{}, ValueRange{mulCos.getResult(), mulSin.getResult()},
      EltwiseKindAttr::get(ctx, EltwiseKind::Add), addPath, DatapathAttr{},
      /*perSlotDatapath=*/ArrayAttr{}, ActSpecAttr{}, PoolSpecAttr{},
      ContractionAttr{}, CombineModeAttr{}, /*rotateHalf=*/UnitAttr{},
      /*transposePurpose=*/TransposePurposeAttr{}, /*sourceSubBlocks=*/ArrayAttr{}, /*sourceBroadcastSpec=*/BroadcastSpecAttr{}, unit);
  setPhaseSpec(add, 2, srcTy, FunctionalUnit::Kantor, /*consecutive=*/true);
  // The add phase differs between the two RoPE chains: the K path requantizes
  // on its way into the cache (card value 3), the Q path leaves the result in
  // fp16 for the following dynamic-quantize op (card value 0). Which chain
  // this is comes from the op's `tailCardValue`, so the pass never has to
  // guess -- and the verifier rejects anything other than those two values,
  // which a name-based lookup could not do.
  int64_t addCard = op.getTailCardValue();
  setKantorSpec(add, addCard == 3 ? KantorMode::Fp2IntConverter
                                  : KantorMode::Off,
                addCard);
  setFpsuSpec(add, 1);

  // The K path's tail *is* a quantization, and the graph format says so with a
  // flat flag rather than a nested block: there is no folded operator to name,
  // only the fact that this node ends in a requantization. That is the other
  // contraction form, and it does not come from `pim-fuse-activation` -- that
  // pass folds a `pim.lut`, and there is none here.
  if (addCard == 3)
    add->setAttr("contraction",
                 ContractionAttr::get(ctx, ContractionForm::Flat,
                                      /*blockName=*/StringAttr{},
                                      /*innerOp=*/StringAttr{},
                                      /*actKind=*/StringAttr{},
                                      StringAttr::get(ctx, "dq_contraction")));

  // 6 个子块名和广播规格是整个 rope 变换的属性，不是某一路 mul 或最终 add
  // 的属性，所以整块搬到 add（原 op 的直接替代者）上，而不是拆分到三个
  // eltwise 上。不搬的话这两个属性会随 op.erase() 一起消失，展开后的 IR
  // 里就再读不到 sin/cos 广播到多头那一路的独立量化参数了。
  if (auto subBlocks = op.getSubBlocksAttr())
    add->setAttr("sourceSubBlocks", subBlocks);
  if (auto broadcastSpec = op.getBroadcastSpecAttr())
    add->setAttr("sourceBroadcastSpec", broadcastSpec);

  op.getResult().replaceAllUsesWith(add.getResult());
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Matmul: single phase, so nothing to expand -- but the layer card still needs
// the FPSU / FLP / Kantor configuration, and that is operator-level knowledge
// this pass owns. Stamping it here keeps one source of truth: a consumer reads
// the same attributes off matmul as off the expanded phases, instead of falling
// back to its own hardcoded table for the matmul family only.
//===----------------------------------------------------------------------===//

static void stampMatmul(MatmulOp op) {
  // Accumulation for the matmul family runs on the 32-bit float FPSU
  // (card value 2), unlike the phase pipelines which use value 1.
  setFpsuSpec(op, 2);

  auto act = op.getActivationAttr();
  if (!act)
    return;
  // A fused activation is evaluated through the LUT, so the interpolation
  // window has to be declared. Measured on the reference gate projection.
  setLutWindow(op, /*min=*/10, /*max=*/17, /*mantisa=*/3);
  setLutTable(op, /*activationMode=*/0, /*specialOperators=*/0);
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
    SmallVector<DynamicQuantOp> dynqs;
    SmallVector<SoftmaxOp> sms;
    SmallVector<RopeOp> ropes;
    SmallVector<MatmulOp> matmuls;
    getOperation().walk([&](Operation *op) {
      if (auto q = dyn_cast<QuantizeOp>(op))
        dqs.push_back(q);
      else if (auto dq = dyn_cast<DynamicQuantOp>(op))
        dynqs.push_back(dq);
      else if (auto s = dyn_cast<SoftmaxOp>(op))
        sms.push_back(s);
      else if (auto r = dyn_cast<RopeOp>(op))
        ropes.push_back(r);
      else if (auto m = dyn_cast<MatmulOp>(op))
        matmuls.push_back(m);
    });

    // Matmul is single-phase: annotate in place, never expand. Do it before the
    // expansions so a matmul fed by an expanded op is untouched by this walk.
    for (auto m : matmuls)
      stampMatmul(m);

    for (auto q : dqs)
      if (failed(expandQuantize(q)))
        return signalPassFailure();
    for (auto dq : dynqs)
      if (failed(expandDynamicQuantOp(dq)))
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
