#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

using namespace mlir;
using namespace mlir::triton::pim;

#define GET_OP_CLASSES
#include "triton/Dialect/TritonPIM/IR/Ops.cpp.inc"

namespace mlir::triton::pim {

//===----------------------------------------------------------------------===//
// Shared verification helpers
//===----------------------------------------------------------------------===//

// A staged tile and the buffer holding it must agree on element type and shape:
// the DMA moves the tile whole, so there is no reshaping in flight.
static LogicalResult verifyTileAgainstBuffer(Operation *op, ShapedType tileTy,
                                             MemDescType bufTy,
                                             StringRef tileName) {
  if (tileTy.getElementType() != bufTy.getElementType())
    return op->emitOpError()
           << tileName << " element type " << tileTy.getElementType()
           << " must match buffer element type " << bufTy.getElementType();

  if (tileTy.getShape() != bufTy.getShape())
    return op->emitOpError()
           << tileName << " shape [" << tileTy.getShape()
           << "] must match buffer shape [" << bufTy.getShape() << "]";

  return success();
}

// The pointee of a tensor-of-pointers, i.e. what a DMA actually transfers.
static Type getPointeeElementType(RankedTensorType ptrTensorTy) {
  auto ptrTy = dyn_cast<triton::PointerType>(ptrTensorTy.getElementType());
  return ptrTy ? ptrTy.getPointeeType() : Type();
}

// Checks the address-pattern attributes a DMA op may carry. Absent attributes
// are always fine -- that is how the IR says "not proven" -- but a present one
// must be self-consistent.
static LogicalResult verifyDmaLayoutAttrs(Operation *op, int64_t rank,
                                          std::optional<int64_t> contiguousDim,
                                          std::optional<int64_t> elemStride) {
  if (contiguousDim && (*contiguousDim < 0 || *contiguousDim >= rank))
    return op->emitOpError("contiguous_dim ")
           << *contiguousDim << " is out of range for rank " << rank;

  if (elemStride && *elemStride <= 0)
    return op->emitOpError("elem_stride must be positive; got ") << *elemStride;

  // A stride is a property of a particular dimension, so it is meaningless on
  // its own.
  if (elemStride && !contiguousDim)
    return op->emitOpError("elem_stride requires contiguous_dim to be set");

  return success();
}

// A quantization parameter tensor must supply exactly as many values as its
// granularity implies. `per_tensor` is one value; `per_channel` is one per index
// along the axis; `per_group` is one per group along it.
static LogicalResult verifyQuantOperand(Operation *op, QuantSpecAttr spec,
                                        RankedTensorType dataTy,
                                        RankedTensorType paramTy,
                                        StringRef paramName) {
  int64_t expected = 1;
  if (spec.getGranularity() != QuantGranularity::PerTensor) {
    int64_t axis = spec.getAxis();
    if (axis < 0 || axis >= dataTy.getRank())
      return op->emitOpError("quantization axis ")
             << axis << " is out of range for rank " << dataTy.getRank();
    int64_t extent = dataTy.getDimSize(axis);
    if (spec.getGranularity() == QuantGranularity::PerGroup) {
      int64_t group = spec.getGroupSize();
      if (extent % group != 0)
        return op->emitOpError("axis extent ")
               << extent << " is not divisible by group size " << group;
      expected = extent / group;
    } else {
      expected = extent;
    }
  }

  int64_t actual = 1;
  for (int64_t dim : paramTy.getShape())
    actual *= dim;
  if (actual != expected)
    return op->emitOpError()
           << paramName << " must supply " << expected
           << " values for this quantization spec; got " << actual;

  return success();
}

// The unit attribute names which functional unit runs an operator, so it has to
// name one that can: a DMA engine does not compute.
static LogicalResult verifyComputeUnit(Operation *op,
                                       std::optional<FunctionalUnit> unit) {
  if (unit && *unit == FunctionalUnit::DMA)
    return op->emitOpError("cannot run on the DMA unit");
  return success();
}

} // namespace mlir::triton::pim

//===----------------------------------------------------------------------===//
// TaskletIdOp / DpuIdOp
//===----------------------------------------------------------------------===//

void TaskletIdOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRanges) {
  int numTasklets = lookupNumTasklets(getOperation());
  setResultRanges(getResult(),
                  ConstantIntRanges::fromUnsigned(
                      APInt(32, 0), APInt(32, std::max(numTasklets - 1, 0))));
}

void DpuIdOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                SetIntRangeFn setResultRanges) {
  int numDpus = lookupNumDpus(getOperation());
  setResultRanges(getResult(),
                  ConstantIntRanges::fromUnsigned(
                      APInt(32, 0), APInt(32, std::max(numDpus - 1, 0))));
}

//===----------------------------------------------------------------------===//
// WRAMAllocOp
//===----------------------------------------------------------------------===//

void WRAMAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0), WRAM::get());
}

int32_t WRAMAllocOp::getAlignmentOrDefault() {
  if (auto align = getAlignment())
    return *align;
  // UPMEM-class DMA engines want 8-byte-aligned transfers; use that as the
  // default until the target hardware pins down its own requirement.
  return 8;
}

LogicalResult WRAMAllocOp::verify() {
  MemDescType ty = getType();
  if (!ty.isWRAM())
    return emitOpError("must allocate in #pim.wram; got ") << ty.getMemorySpace();

  if (auto bytes = ty.getSizeInBytes()) {
    if (auto budget = maybeLookupWramBytes(getOperation())) {
      if (*bytes > *budget)
        return emitOpError("allocation of ")
               << *bytes << " bytes exceeds the WRAM budget of " << *budget
               << " bytes";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DmaLoadOp
//===----------------------------------------------------------------------===//

void DmaLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable(),
                       MRAM::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getResultMutable(),
                       WRAM::get());
}

LogicalResult DmaLoadOp::verify() {
  auto ptrTy = cast<RankedTensorType>(getPtr().getType());
  MemDescType bufTy = getMemDescType();

  Type pointee = getPointeeElementType(ptrTy);
  if (!pointee)
    return emitOpError("ptr must be a tensor of pointers");

  if (pointee != bufTy.getElementType())
    return emitOpError("pointee type ")
           << pointee << " must match buffer element type "
           << bufTy.getElementType();

  if (ptrTy.getShape() != bufTy.getShape())
    return emitOpError("ptr shape [")
           << ptrTy.getShape() << "] must match buffer shape ["
           << bufTy.getShape() << "]";

  if (auto other = getOther()) {
    if (auto otherTy = dyn_cast<ShapedType>(other.getType()))
      if (failed(verifyTileAgainstBuffer(getOperation(), otherTy, bufTy,
                                         "other")))
        return failure();
  }

  return verifyDmaLayoutAttrs(getOperation(), ptrTy.getRank(),
                              getContiguousDim(), getElemStride());
}

//===----------------------------------------------------------------------===//
// DmaStoreOp
//===----------------------------------------------------------------------===//

void DmaStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       WRAM::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable(),
                       MRAM::get());
}

LogicalResult DmaStoreOp::verify() {
  auto ptrTy = cast<RankedTensorType>(getPtr().getType());
  MemDescType bufTy = getMemDescType();

  Type pointee = getPointeeElementType(ptrTy);
  if (!pointee)
    return emitOpError("ptr must be a tensor of pointers");

  if (pointee != bufTy.getElementType())
    return emitOpError("pointee type ")
           << pointee << " must match buffer element type "
           << bufTy.getElementType();

  if (ptrTy.getShape() != bufTy.getShape())
    return emitOpError("ptr shape [")
           << ptrTy.getShape() << "] must match buffer shape ["
           << bufTy.getShape() << "]";

  return verifyDmaLayoutAttrs(getOperation(), ptrTy.getRank(),
                              getContiguousDim(), getElemStride());
}

//===----------------------------------------------------------------------===//
// WRAMLoadOp / WRAMStoreOp
//===----------------------------------------------------------------------===//

void WRAMLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       WRAM::get());
}

LogicalResult WRAMLoadOp::verify() {
  return verifyTileAgainstBuffer(getOperation(),
                                 cast<ShapedType>(getResult().getType()),
                                 getSrc().getType(), "result");
}

void WRAMStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       WRAM::get());
}

LogicalResult WRAMStoreOp::verify() {
  return verifyTileAgainstBuffer(getOperation(),
                                 cast<ShapedType>(getSrc().getType()),
                                 getDst().getType(), "src");
}

//===----------------------------------------------------------------------===//
// ConvertLayoutOp
//===----------------------------------------------------------------------===//

LogicalResult ConvertLayoutOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!srcTy || !dstTy)
    return success(); // shape/element type already checked by op traits

  // A conversion that changes nothing is pointless; catching it here keeps the
  // canonicalizer from having to reason about no-op conversions.
  if (srcTy.getEncoding() == dstTy.getEncoding())
    return emitOpError("source and destination layouts are identical");

  return success();
}

// `I64Attr` getters return `uint64_t`, so a negative axis written as `-1 : i64`
// arrives as a huge positive number. Reinterpret the bits as signed before doing
// anything with it.
static int64_t asSigned(uint64_t raw) { return static_cast<int64_t>(raw); }

// Resolves an axis that may count back from the end, as elsewhere in Triton.
// Returns nullopt when it lands outside the rank.
static std::optional<int64_t> normalizeAxis(uint64_t raw, int64_t rank) {
  int64_t axis = asSigned(raw);
  int64_t norm = axis < 0 ? axis + rank : axis;
  if (norm < 0 || norm >= rank)
    return std::nullopt;
  return norm;
}

//===----------------------------------------------------------------------===//
// QuantizeOp / DequantizeOp
//===----------------------------------------------------------------------===//

// Shared by both directions: the result must keep the source's shape, and each
// scale/zero-point operand must match the declared granularity.
static LogicalResult verifyQuantConversion(Operation *op, QuantSpecAttr spec,
                                           Value src, Value result, Value scale,
                                           Value zeroPoint) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto resTy = cast<RankedTensorType>(result.getType());
  if (srcTy.getShape() != resTy.getShape())
    return op->emitOpError("result shape [")
           << resTy.getShape() << "] must match source shape ["
           << srcTy.getShape() << "]";

  if (failed(verifyQuantOperand(op, spec, srcTy,
                                cast<RankedTensorType>(scale.getType()),
                                "scale")))
    return failure();

  if (zeroPoint)
    if (failed(verifyQuantOperand(op, spec, srcTy,
                                  cast<RankedTensorType>(zeroPoint.getType()),
                                  "zeroPoint")))
      return failure();

  return success();
}

LogicalResult QuantizeOp::verify() {
  return verifyQuantConversion(getOperation(), getSpec(), getSrc(), getResult(),
                               getScale(), getZeroPoint());
}

LogicalResult DequantizeOp::verify() {
  return verifyQuantConversion(getOperation(), getSpec(), getSrc(), getResult(),
                               getScale(), getZeroPoint());
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto aTy = cast<RankedTensorType>(getA().getType());
  auto bTy = cast<RankedTensorType>(getB().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  if (aTy.getRank() != 2 || bTy.getRank() != 2 || resTy.getRank() != 2)
    return emitOpError("operands and result must be 2-D");

  // With `transposeB` the second operand arrives as [N, K] rather than [K, N].
  int64_t m = aTy.getDimSize(0), k = aTy.getDimSize(1);
  int64_t bK = getTransposeB() ? bTy.getDimSize(1) : bTy.getDimSize(0);
  int64_t n = getTransposeB() ? bTy.getDimSize(0) : bTy.getDimSize(1);

  if (k != bK)
    return emitOpError("contraction dimensions disagree: a has ")
           << k << ", b has " << bK;

  if (resTy.getDimSize(0) != m || resTy.getDimSize(1) != n)
    return emitOpError("result shape [")
           << resTy.getShape() << "] does not match the expected [" << m << ", "
           << n << "]";

  if (auto bias = getBias()) {
    auto biasTy = cast<RankedTensorType>(bias.getType());
    int64_t elems = 1;
    for (int64_t dim : biasTy.getShape())
      elems *= dim;
    // A bias is per output column, or a single value broadcast over all of them.
    if (elems != n && elems != 1)
      return emitOpError("bias must supply ")
             << n << " or 1 values; got " << elems;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

LogicalResult ConvOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto inTy = cast<RankedTensorType>(getInput().getType());
  auto wTy = cast<RankedTensorType>(getWeight().getType());

  // NCHW input against OIHW weight.
  if (inTy.getRank() != 4 || wTy.getRank() != 4)
    return emitOpError("input and weight must be 4-D (NCHW and OIHW)");

  int64_t group = getWindow().getGroup();
  if (inTy.getDimSize(1) % group != 0)
    return emitOpError("input channels ")
           << inTy.getDimSize(1) << " not divisible by group " << group;

  if (inTy.getDimSize(1) / group != wTy.getDimSize(1))
    return emitOpError("weight input channels ")
           << wTy.getDimSize(1) << " must equal input channels / group ("
           << inTy.getDimSize(1) / group << ")";

  ArrayRef<int64_t> kernel = getWindow().getKernel().asArrayRef();
  if (kernel.size() != 2 || kernel[0] != wTy.getDimSize(2) ||
      kernel[1] != wTy.getDimSize(3))
    return emitOpError("window kernel does not match the weight's spatial "
                       "extent [")
           << wTy.getDimSize(2) << ", " << wTy.getDimSize(3) << "]";

  if (auto bias = getBias()) {
    auto biasTy = cast<RankedTensorType>(bias.getType());
    int64_t elems = 1;
    for (int64_t dim : biasTy.getShape())
      elems *= dim;
    // One bias per output channel, i.e. per filter.
    if (elems != wTy.getDimSize(0))
      return emitOpError("bias must supply ")
             << wTy.getDimSize(0) << " values; got " << elems;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LutOp
//===----------------------------------------------------------------------===//

LogicalResult LutOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  // Only the bounded activations take a clip range or a slope; carrying either
  // on an activation that ignores it would silently mislead a reader.
  ActivationKind kind = getKind();
  bool bounded = kind == ActivationKind::ReluX;
  if (!bounded && (getClipMin() || getClipMax()))
    return emitOpError("clip bounds are only meaningful for relu_x");

  bool sloped = kind == ActivationKind::LeakyRelu;
  if (!sloped && getAlpha())
    return emitOpError("alpha is only meaningful for leaky_relu");

  if (bounded && getClipMin() && getClipMax() && *getClipMin() > *getClipMax())
    return emitOpError("clipMin must not exceed clipMax");

  return success();
}

//===----------------------------------------------------------------------===//
// EltwiseOp
//===----------------------------------------------------------------------===//

LogicalResult EltwiseOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto lhsTy = cast<RankedTensorType>(getLhs().getType());
  auto rhsTy = cast<RankedTensorType>(getRhs().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  if (lhsTy.getShape() != resTy.getShape())
    return emitOpError("result shape [")
           << resTy.getShape() << "] must match lhs shape [" << lhsTy.getShape()
           << "]";

  // `rhs` either matches `lhs` exactly or is broadcast against its trailing
  // dimensions, which is how a per-channel factor or a bias is supplied.
  if (rhsTy.getShape() != lhsTy.getShape()) {
    if (rhsTy.getRank() > lhsTy.getRank())
      return emitOpError("rhs rank ")
             << rhsTy.getRank() << " exceeds lhs rank " << lhsTy.getRank();
    unsigned offset = lhsTy.getRank() - rhsTy.getRank();
    for (int64_t i = 0; i < rhsTy.getRank(); ++i) {
      int64_t r = rhsTy.getDimSize(i), l = lhsTy.getDimSize(i + offset);
      if (r != l && r != 1)
        return emitOpError("rhs shape [")
               << rhsTy.getShape() << "] is not broadcastable against lhs shape ["
               << lhsTy.getShape() << "]";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PoolOp
//===----------------------------------------------------------------------===//

LogicalResult PoolOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  // A global reduction covers the whole spatial extent, so a window would be
  // ignored; a windowed one has no geometry without it.
  bool global = getKind() == PoolKind::GlobalAverage;
  if (global && getWindow())
    return emitOpError("global_average takes no window");
  if (!global && !getWindow())
    return emitOpError("windowed pooling requires a window");

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceAxisOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceAxisOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  int64_t axis = getAxis();
  if (axis < 0 || axis >= srcTy.getRank())
    return emitOpError("axis ")
           << axis << " is out of range for rank " << srcTy.getRank();

  // The rank is kept and the reduced axis collapses to 1, so the result shape is
  // fully determined by the source.
  if (resTy.getRank() != srcTy.getRank())
    return emitOpError("result rank must equal source rank");
  for (int64_t i = 0; i < srcTy.getRank(); ++i) {
    int64_t expected = (i == axis) ? 1 : srcTy.getDimSize(i);
    if (resTy.getDimSize(i) != expected)
      return emitOpError("result shape [")
             << resTy.getShape() << "] does not match the reduction of ["
             << srcTy.getShape() << "] along axis " << axis;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RopeOp
//===----------------------------------------------------------------------===//

LogicalResult RopeOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  if (getNumHeads() <= 0)
    return emitOpError("numHeads must be positive");

  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());
  if (srcTy.getShape() != resTy.getShape())
    return emitOpError("result shape [")
           << resTy.getShape() << "] must match source shape ["
           << srcTy.getShape() << "]";

  auto cosTy = cast<RankedTensorType>(getCos().getType());
  auto sinTy = cast<RankedTensorType>(getSin().getType());
  if (cosTy.getShape() != sinTy.getShape())
    return emitOpError("cos and sin tables must have the same shape");

  return success();
}

//===----------------------------------------------------------------------===//
// NormalizeOp
//===----------------------------------------------------------------------===//

LogicalResult NormalizeOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());
  if (srcTy.getShape() != resTy.getShape())
    return emitOpError("result shape [")
           << resTy.getShape() << "] must match source shape ["
           << srcTy.getShape() << "]";

  int64_t rank = srcTy.getRank();
  auto maybeNorm = normalizeAxis(getAxis(), rank);
  if (!maybeNorm)
    return emitOpError("axis ")
           << asSigned(getAxis()) << " is out of range for rank " << rank;
  int64_t norm = *maybeNorm;

  if (getEpsilon().convertToDouble() <= 0.0)
    return emitOpError("epsilon must be positive");

  // Both affine parameters run along the normalized axis.
  int64_t extent = srcTy.getDimSize(norm);
  for (auto [val, name] :
       {std::pair{getWeight(), "weight"}, std::pair{getBias(), "bias"}}) {
    if (!val)
      continue;
    auto ty = cast<RankedTensorType>(val.getType());
    int64_t elems = 1;
    for (int64_t dim : ty.getShape())
      elems *= dim;
    if (elems != extent)
      return emitOpError()
             << name << " must supply " << extent << " values; got " << elems;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());
  if (srcTy.getShape() != resTy.getShape())
    return emitOpError("result shape [")
           << resTy.getShape() << "] must match source shape ["
           << srcTy.getShape() << "]";

  int64_t rank = srcTy.getRank();
  if (!normalizeAxis(getAxis(), rank))
    return emitOpError("axis ")
           << asSigned(getAxis()) << " is out of range for rank " << rank;

  return success();
}

//===----------------------------------------------------------------------===//
// BufferAllocOp / BufferCopyOp / DecompressWeightOp
//===----------------------------------------------------------------------===//

void BufferAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  MemDescType ty = getType();
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getOpResult(0),
                       ty.isL1() ? (SideEffects::Resource *)L1::get()
                                 : (SideEffects::Resource *)L2::get());
}

LogicalResult BufferAllocOp::verify() {
  MemDescType ty = getType();
  if (!ty.isL1() && !ty.isL2())
    return emitOpError("must allocate in #pim.l1 or #pim.l2; got ")
           << ty.getMemorySpace();

  // L1 belongs to a specific unit; L2 belongs to all of them, so naming one
  // there would be a claim the hardware cannot honour.
  if (ty.isL2() && getUnit())
    return emitOpError("an L2 allocation is shared and takes no unit");

  if (auto bytes = ty.getSizeInBytes()) {
    auto budget = ty.isL1() ? maybeLookupL1Bytes(getOperation())
                            : maybeLookupL2Bytes(getOperation());
    if (budget && *bytes > *budget)
      return emitOpError("allocation of ")
             << *bytes << " bytes exceeds the "
             << (ty.isL1() ? "L1" : "L2") << " budget of " << *budget
             << " bytes";
  }

  return success();
}

void BufferCopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto resourceFor = [](MemDescType ty) -> SideEffects::Resource * {
    if (ty.isL1())
      return L1::get();
    if (ty.isL2())
      return L2::get();
    return WRAM::get();
  };
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       resourceFor(getSrc().getType()));
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       resourceFor(getDst().getType()));
}

LogicalResult BufferCopyOp::verify() {
  MemDescType srcTy = getSrc().getType();
  MemDescType dstTy = getDst().getType();

  if (failed(verifyTileAgainstBuffer(getOperation(), srcTy, dstTy, "src")))
    return failure();

  // A copy within one level moves nothing the allocator could not have avoided.
  if (srcTy.getMemorySpace() == dstTy.getMemorySpace())
    return emitOpError("source and destination are in the same memory space; "
                       "an on-chip copy must cross a level");

  return success();
}

void DecompressWeightOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  MemDescType srcTy = getSrc().getType();
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       srcTy.isMRAM() ? (SideEffects::Resource *)MRAM::get()
                                      : (SideEffects::Resource *)L2::get());
  MemDescType dstTy = getDst().getType();
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       dstTy.isL1() ? (SideEffects::Resource *)L1::get()
                                    : (SideEffects::Resource *)L2::get());
}

LogicalResult DecompressWeightOp::verify() {
  if (getSrc().getType().getElementType() !=
      getDst().getType().getElementType())
    return emitOpError("compressed and expanded buffers must share an element "
                       "type; decompression changes the extent, not the format");

  // Expansion has to expand: equal or smaller output means the compressed form
  // bought nothing, and the shapes are static so this is checkable here.
  auto srcBytes = getSrc().getType().getSizeInBytes();
  auto dstBytes = getDst().getType().getSizeInBytes();
  if (srcBytes && dstBytes && *srcBytes >= *dstBytes)
    return emitOpError("expanded buffer of ")
           << *dstBytes << " bytes must be larger than the compressed "
           << *srcBytes << " bytes";

  if (auto ratio = getRatio())
    if (ratio->convertToDouble() <= 1.0)
      return emitOpError("ratio must exceed 1");

  return success();
}

//===----------------------------------------------------------------------===//
// ParamOp
//===----------------------------------------------------------------------===//

LogicalResult ParamOp::verify() {
  // The name is the only link between this value and the buffer file holding its
  // bytes, so an empty one silently loses the parameter.
  if (getName().empty())
    return emitOpError("name must not be empty");

  auto ty = cast<RankedTensorType>(getResult().getType());
  if (!ty.hasStaticShape())
    return emitOpError("a parameter must have a static shape");

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());
  ArrayRef<int64_t> axes = getAxes();

  int64_t rank = srcTy.getRank();
  if (static_cast<int64_t>(axes.size()) != rank)
    return emitOpError("axes must have one entry per dimension (")
           << rank << ")";

  // Every dimension has to appear exactly once, or the permutation would drop or
  // duplicate one.
  SmallVector<bool> seen(rank, false);
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= rank)
      return emitOpError("axis ") << axis << " is out of range";
    if (seen[axis])
      return emitOpError("axis ") << axis << " appears more than once";
    seen[axis] = true;
  }

  if (resTy.getRank() != rank)
    return emitOpError("result rank must equal source rank");
  for (int64_t i = 0; i < rank; ++i)
    if (resTy.getDimSize(i) != srcTy.getDimSize(axes[i]))
      return emitOpError("result shape [")
             << resTy.getShape() << "] does not match the permutation of ["
             << srcTy.getShape() << "]";

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// Product of a shape's dimensions, i.e. how many elements it holds.
static int64_t elementCountOf(ShapedType ty) {
  int64_t count = 1;
  for (int64_t dim : ty.getShape())
    count *= dim;
  return count;
}

LogicalResult ReshapeOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  if (srcTy.getElementType() != resTy.getElementType())
    return emitOpError("reshape must not change the element type");

  // No data moves, so the element count has to be preserved exactly.
  if (elementCountOf(srcTy) != elementCountOf(resTy))
    return emitOpError("result holds ")
           << elementCountOf(resTy) << " elements but the source holds "
           << elementCountOf(srcTy);

  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  int64_t axis = getAxis();

  if (axis < 0 || axis >= srcTy.getRank())
    return emitOpError("axis ")
           << axis << " is out of range for rank " << srcTy.getRank();

  if (getResults().empty())
    return emitOpError("split must produce at least one result");

  // The pieces tile the source along `axis` and match it everywhere else.
  int64_t total = 0;
  for (Value result : getResults()) {
    auto resTy = cast<RankedTensorType>(result.getType());
    if (resTy.getRank() != srcTy.getRank())
      return emitOpError("every result must have the source's rank");
    if (resTy.getElementType() != srcTy.getElementType())
      return emitOpError("every result must have the source's element type");
    for (int64_t i = 0; i < srcTy.getRank(); ++i) {
      if (i == axis)
        continue;
      if (resTy.getDimSize(i) != srcTy.getDimSize(i))
        return emitOpError("results may differ from the source only along axis ")
               << axis;
    }
    total += resTy.getDimSize(axis);
  }

  if (total != srcTy.getDimSize(axis))
    return emitOpError("results sum to ")
           << total << " along axis " << axis << " but the source has "
           << srcTy.getDimSize(axis);

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  if (getSrcs().empty())
    return emitOpError("concat needs at least one input");

  auto resTy = cast<RankedTensorType>(getResult().getType());
  int64_t axis = getAxis();

  if (axis < 0 || axis >= resTy.getRank())
    return emitOpError("axis ")
           << axis << " is out of range for rank " << resTy.getRank();

  int64_t total = 0;
  for (Value src : getSrcs()) {
    auto srcTy = cast<RankedTensorType>(src.getType());
    if (srcTy.getRank() != resTy.getRank())
      return emitOpError("every input must have the result's rank");
    if (srcTy.getElementType() != resTy.getElementType())
      return emitOpError("every input must have the result's element type");
    for (int64_t i = 0; i < resTy.getRank(); ++i) {
      if (i == axis)
        continue;
      if (srcTy.getDimSize(i) != resTy.getDimSize(i))
        return emitOpError("inputs may differ from the result only along axis ")
               << axis;
    }
    total += srcTy.getDimSize(axis);
  }

  if (total != resTy.getDimSize(axis))
    return emitOpError("inputs sum to ")
           << total << " along axis " << axis << " but the result has "
           << resTy.getDimSize(axis);

  return success();
}

//===----------------------------------------------------------------------===//
// MaskOp
//===----------------------------------------------------------------------===//

LogicalResult MaskOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto scoresTy = cast<RankedTensorType>(getScores().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());
  if (scoresTy.getShape() != resTy.getShape())
    return emitOpError("result shape [")
           << resTy.getShape() << "] must match scores shape ["
           << scoresTy.getShape() << "]";

  // The mask either matches the scores exactly or broadcasts against their
  // trailing dimensions, which is how one mask covers every head.
  auto maskTy = cast<RankedTensorType>(getMask().getType());
  if (maskTy.getRank() > scoresTy.getRank())
    return emitOpError("mask rank ")
           << maskTy.getRank() << " exceeds scores rank "
           << scoresTy.getRank();
  unsigned offset = scoresTy.getRank() - maskTy.getRank();
  for (int64_t i = 0; i < maskTy.getRank(); ++i) {
    int64_t m = maskTy.getDimSize(i), s = scoresTy.getDimSize(i + offset);
    if (m != s && m != 1)
      return emitOpError("mask shape [")
             << maskTy.getShape() << "] is not broadcastable against scores ["
             << scoresTy.getShape() << "]";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// KvCacheOp
//===----------------------------------------------------------------------===//

void KvCacheOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  MemDescType cacheTy = getCache().getType();
  auto *resource = cacheTy.isMRAM() ? (SideEffects::Resource *)MRAM::get()
                                    : (SideEffects::Resource *)L2::get();
  // A read pulls from the cache; a write pushes into it. The cache outlives the
  // step either way, so the effect is on the buffer rather than on a result.
  if (getIsRead())
    effects.emplace_back(MemoryEffects::Read::get(), &getCacheMutable(),
                         resource);
  else
    effects.emplace_back(MemoryEffects::Write::get(), &getCacheMutable(),
                         resource);
}

LogicalResult KvCacheOp::verify() {
  if (getLayer() < 0)
    return emitOpError("layer must be non-negative");

  MemDescType cacheTy = getCache().getType();
  auto valueTy = cast<RankedTensorType>(getValue().getType());
  if (cacheTy.getElementType() != valueTy.getElementType())
    return emitOpError("cache element type ")
           << cacheTy.getElementType() << " must match value element type "
           << valueTy.getElementType();

  // The cache holds many steps of what `value` carries for one, so it has to be
  // the larger of the two.
  auto cacheBytes = cacheTy.getSizeInBytes();
  if (cacheBytes) {
    int64_t valueElems = 1;
    for (int64_t dim : valueTy.getShape())
      valueElems *= dim;
    int64_t cacheElems = 1;
    for (int64_t dim : cacheTy.getShape())
      cacheElems *= dim;
    if (cacheElems < valueElems)
      return emitOpError("cache holds ")
             << cacheElems << " elements, fewer than the " << valueElems
             << " being moved";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SplitHeadsOp
//===----------------------------------------------------------------------===//

LogicalResult SplitHeadsOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  if (getNumHeads() <= 0)
    return emitOpError("numHeads must be positive");

  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  int64_t rank = srcTy.getRank();
  auto maybeAxis = normalizeAxis(getAxis(), rank);
  if (!maybeAxis)
    return emitOpError("axis ")
           << asSigned(getAxis()) << " is out of range for rank " << rank;
  int64_t axis = *maybeAxis;

  if (static_cast<int64_t>(getResults().size()) != getNumHeads())
    return emitOpError("numHeads is ")
           << getNumHeads() << " but the op has " << getResults().size()
           << " results";

  int64_t extent = srcTy.getDimSize(axis);
  if (extent % getNumHeads() != 0)
    return emitOpError("axis extent ")
           << extent << " is not divisible by numHeads " << getNumHeads();
  int64_t perHead = extent / getNumHeads();

  // Every head gets the same slice, which is what distinguishes this from
  // `pim.split`.
  for (Value result : getResults()) {
    auto resTy = cast<RankedTensorType>(result.getType());
    if (resTy.getRank() != rank)
      return emitOpError("every result must have the source's rank");
    if (resTy.getElementType() != srcTy.getElementType())
      return emitOpError("every result must have the source's element type");
    for (int64_t i = 0; i < rank; ++i) {
      int64_t expected = (i == axis) ? perHead : srcTy.getDimSize(i);
      if (resTy.getDimSize(i) != expected)
        return emitOpError("result shape [")
               << resTy.getShape() << "] does not match the per-head slice of ["
               << srcTy.getShape() << "]";
    }
  }

  return success();
}
