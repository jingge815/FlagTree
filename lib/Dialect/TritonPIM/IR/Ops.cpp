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
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();
  return verifyQuantConversion(getOperation(), getSpec(), getSrc(), getResult(),
                               getScale(), getZeroPoint());
}

LogicalResult DynamicQuantOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
  auto scaleTy = dyn_cast<RankedTensorType>(getScale().getType());
  if (!srcTy || !resTy || !scaleTy)
    return emitOpError("src, result and scale must be ranked tensors");
  if (srcTy.getShape() != resTy.getShape())
    return emitOpError("result shape must match src");

  // The chain is float in, int8 out. Both are checked because a wrong element
  // type still verifies structurally -- the failure would surface much later,
  // as a lowering that reads an f16 buffer as fixed point.
  if (!srcTy.getElementType().isF16())
    return emitOpError("src must be f16; got ") << srcTy.getElementType();
  if (!resTy.getElementType().isInteger(8))
    return emitOpError("result must be i8; got ") << resTy.getElementType();

  int64_t groupSize = getGroupSize();
  if (groupSize <= 0)
    return emitOpError("groupSize must be positive");
  int64_t axis = getAxis();
  int64_t rank = srcTy.getRank();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return emitOpError("axis out of range");

  // The operand attributes and the spec describe **the same** quantization.
  // Checking only one of them lets the two disagree and nobody notices: the
  // verifier would pass on the operand's group size while the expansion reads
  // the spec's, producing a chain whose real group size is not the one the
  // declared scale shape was built for.
  auto spec = getSpec();
  if (!spec)
    return emitOpError("dynamic quantization needs a spec");
  if (spec->getGranularity() != QuantGranularity::PerGroup)
    return emitOpError("dynamic quantization is per_group; spec says ")
           << stringifyQuantGranularity(spec->getGranularity());
  if (spec->getGroupSize() != groupSize)
    return emitOpError("spec groupSize is ")
           << spec->getGroupSize() << " but the op says " << groupSize
           << "; the two describe one quantization and must agree";
  int64_t specAxis = spec->getAxis();
  if (specAxis < 0)
    specAxis += rank;
  if (specAxis != axis)
    return emitOpError("spec axis is ")
           << spec->getAxis() << " but the op says " << getAxis()
           << "; the two describe one quantization and must agree";

  int64_t extent = srcTy.getDimSize(axis);
  if (extent % groupSize != 0)
    return emitOpError("axis extent is not divisible by groupSize");
  int64_t groups = extent / groupSize;
  int64_t scaleElems = 1;
  for (int64_t d : scaleTy.getShape())
    scaleElems *= d;
  if (scaleElems != groups)
    return emitOpError("scale must hold one value per group");

  // Shape contract, checked only when the graph compiler states it. The last
  // axis of `originalShape` is the one being grouped, and the per-group view
  // replaces it with `[groups, groupSize]` -- a different split would mean the
  // scale blob and the data disagree about where the groups are.
  if (auto original = getOriginalShape()) {
    if ((*original).size() != static_cast<size_t>(rank))
      return emitOpError("originalShape must have the source's rank; got ")
             << (*original).size() << " entries for rank " << rank;
    int64_t last = cast<IntegerAttr>((*original)[rank - 1]).getInt();
    if (last % groupSize != 0)
      return emitOpError("originalShape's last axis ")
             << last << " is not divisible by groupSize " << groupSize;
    if (auto byGroup = getOutShapeByGroup()) {
      if ((*byGroup).size() != static_cast<size_t>(rank) + 1)
        return emitOpError("outShapeByGroup must have rank + 1 entries; got ")
               << (*byGroup).size() << " for rank " << rank;
      SmallVector<int64_t> want;
      for (int64_t i = 0; i < rank - 1; ++i)
        want.push_back(cast<IntegerAttr>((*original)[i]).getInt());
      want.push_back(last / groupSize);
      want.push_back(groupSize);
      for (size_t i = 0; i < want.size(); ++i)
        if (cast<IntegerAttr>((*byGroup)[i]).getInt() != want[i])
          return emitOpError("outShapeByGroup must be originalShape with the "
                             "grouped axis split into [groups, groupSize]");
    }
  }
  return success();
}

LogicalResult DequantizeOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();
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

  // The binding and `transposeB` state the same fact about the weight's layout.
  // Both are kept for now -- `transposeB` predates the binding -- so they have
  // to agree rather than leave a reader to pick one.
  if (auto binding = getWeightBinding()) {
    bool bindingSaysTransposed =
        binding->getFormat() == WeightFormat::WeightsTranspose;
    if (bindingSaysTransposed != static_cast<bool>(getTransposeB()))
      return emitOpError("weightBinding format weights_transpose must agree "
                         "with transposeB");
  }

  // A weight-role quant spec states the value range the weight was stored in;
  // the binding states the element width. Both describe the same storage, so a
  // range outside what the width can hold means one of them is wrong -- and the
  // rescale would then use a bound that never applies, silently.
  if (auto binding = getWeightBinding()) {
    if (auto spec = getDatapath().getScaleSpec()) {
      if (spec.getRole() == QuantRole::Weight && spec.getRange()) {
        int64_t elemBits = binding->getElemBits();
        auto lo = cast<FloatAttr>(spec.getRange()[0]).getValueAsDouble();
        auto hi = cast<FloatAttr>(spec.getRange()[1]).getValueAsDouble();
        double wantLo = -static_cast<double>(int64_t{1} << (elemBits - 1));
        double wantHi = static_cast<double>((int64_t{1} << (elemBits - 1)) - 1);
        if (lo != wantLo || hi != wantHi)
          return emitOpError("role weight states range [")
                 << lo << ", " << hi << "], but elemBits = " << elemBits
                 << " holds [" << wantLo << ", " << wantHi
                 << "]; the two describe the same storage";
      }
    }
  }

  // The accumulator dequantizes a group at a time, so it has to use the same
  // grouping the weight was quantized at. Two different group sizes would mean
  // the accumulator folds in a scale that belongs to a different run of
  // elements -- wrong numbers, and no error anywhere to say so.
  if (getDatapath().getGroupDequantAccum()) {
    auto binding = getWeightBinding();
    if (!binding)
      return emitOpError("groupDequantAccum dequantizes the weight per group, "
                         "so the weight's layout has to be stated: it needs a "
                         "weightBinding");
    if (getDatapath().getGroupSize() != binding->getGroupSize())
      return emitOpError("groupDequantAccum groupSize ")
             << getDatapath().getGroupSize()
             << " must equal the weight's groupSize "
             << binding->getGroupSize();

    // The factors have to be present, and there have to be one per (group,
    // column). Without them the boundary multiplies by a single scalar, and a
    // scalar at each boundary equals the same scalar once at the end of the row
    // -- so "dequantize per group" would produce the same numbers as "accumulate
    // then dequantize", and the rule this datapath exists to state could not
    // fail a test.
    if (!getWeightScales())
      return emitOpError("groupDequantAccum dequantizes at every group "
                         "boundary, so it needs the per-group factors as an "
                         "operand; with one scalar the grouped and ungrouped "
                         "orders give the same number");
    auto aTy = cast<RankedTensorType>(getA().getType());
    auto scalesTy = cast<RankedTensorType>(getWeightScales().getType());
    int64_t k = aTy.getShape().back();
    int64_t groupSize = getDatapath().getGroupSize();
    if (groupSize <= 0)
      return emitOpError("groupDequantAccum needs a positive groupSize");
    if (k % groupSize != 0)
      return emitOpError("the K dimension ")
             << k << " is not a whole number of " << groupSize
             << "-wide groups, so the last group would be short";
    int64_t groups = k / groupSize;
    int64_t n = cast<RankedTensorType>(getResult().getType()).getShape().back();
    SmallVector<int64_t> want{groups, n};
    if (scalesTy.getShape() != ArrayRef<int64_t>(want))
      return emitOpError("weightScales must be [K/groupSize, N] = [")
             << groups << ", " << n << "], got [" << scalesTy.getShape() << "]";
  } else if (getWeightScales()) {
    return emitOpError("weightScales is only meaningful with "
                       "groupDequantAccum: without it nothing dequantizes at a "
                       "group boundary, so the factors would go unread");
  }

  // Residency and the binding describe the same operand from two sides, so a
  // disagreement means one of them is wrong -- and a reader has no way to tell
  // which. Pinning them together here is what makes either one trustworthy.
  if (auto residency = getStationarity()) {
    auto binding = getWeightBinding();
    switch (*residency) {
    case Stationarity::Weight:
      if (!binding || binding->getRole() != WeightRole::ModelWeight)
        return emitOpError("stationarity weight means a model weight stays "
                           "resident, so weightBinding must say role = "
                           "model_weight");
      break;
    case Stationarity::KV:
      if (!binding || binding->getRole() != WeightRole::ActivationAsWeight)
        return emitOpError("stationarity kv means a cached K/V slice rides the "
                           "weight path, so weightBinding must say role = "
                           "activation_as_weight");
      break;
    case Stationarity::Activation:
      // Prefill multiplies activation by activation: nothing is resident, so a
      // weight binding would describe a path this multiply does not use.
      if (binding)
        return emitOpError("stationarity activation means neither operand is "
                           "resident, so it takes no weightBinding");
      break;
    }

    // `bIsActivation` covers two of the three cases and cannot express the
    // third. While both exist they have to agree.
    bool flagSaysActivation = static_cast<bool>(getBIsActivation());
    bool residencySaysActivation = *residency != Stationarity::Weight;
    if (flagSaysActivation != residencySaysActivation)
      return emitOpError("bIsActivation and stationarity disagree about "
                         "whether the second operand is an activation");
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

  // The window is one triple: it says which segment of the table an input lands
  // in, so a partial triple would address the wrong segment -- wrong numerics
  // with nothing to report it.
  bool anyFlp = getFlpMinExp() || getFlpMaxExp() || getFlpMantisa();
  bool allFlp = getFlpMinExp() && getFlpMaxExp() && getFlpMantisa();
  if (anyFlp != allFlp)
    return emitOpError("flpMinExp, flpMaxExp and flpMantisa describe one window "
                       "and must be given together");

  // The reciprocal table is selected by this flag rather than by `kind`, so the
  // two have to agree; 0 is the regular table and 4 the reciprocal one.

  // 激活单元的映射模式只有三种取值；写别的数会一路传到 GML 而不报错。
  if (auto mode = getActivationMode())
    if (*mode < 0 || *mode > 2)
      return emitOpError("activationMode is 0, 1 or 2; got ") << *mode;
  if (auto special = getSpecialOperators()) {
    if (*special != 0 && *special != 4)
      return emitOpError("specialOperators is 0 (regular) or 4 (reciprocal); "
                         "got ")
             << *special;
  }

  // A table is 144 fp16 segments -- the 288 bytes the activation unit can
  // address. A table of any other size is not one it can read.
  if (auto table = getTable()) {
    auto ty = dyn_cast<RankedTensorType>(table.getType());
    if (!ty || !ty.hasStaticShape())
      return emitOpError("table must have a static shape");
    if (ty.getNumElements() != 144 || !ty.getElementType().isF16())
      return emitOpError("table must be 144 f16 entries (288 bytes); got ")
             << ty.getNumElements() << " x " << ty.getElementType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FpsuScaleOp
//===----------------------------------------------------------------------===//

LogicalResult FpsuScaleOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto biasTy = cast<RankedTensorType>(getBias().getType());
  auto scaleTy = cast<RankedTensorType>(getScale().getType());

  // Three different widths on purpose: a 32-bit bias, a 16-bit scale and an
  // 8-bit shift. They are not interchangeable, and swapping two of them is a
  // silent numerical error rather than a type error.
  if (!biasTy.getElementType().isF32())
    return emitOpError("bias must be f32; got ") << biasTy.getElementType();
  if (scaleTy.getElementType() != srcTy.getElementType())
    return emitOpError("scale must have the source's element type (")
           << srcTy.getElementType() << "); got " << scaleTy.getElementType();

  if (scaleTy.getShape() != biasTy.getShape())
    return emitOpError("bias and scale must have the same shape; got ")
           << biasTy.getShape() << " and " << scaleTy.getShape();

  // A per-channel or per-group set is smaller than the tensor and broadcasts
  // against it; one of a higher rank than the source cannot.
  if (scaleTy.getRank() > srcTy.getRank())
    return emitOpError("the bias/scale rank (")
           << scaleTy.getRank() << ") exceeds the source rank ("
           << srcTy.getRank() << ")";

  // 逐通道的轴号要落在源张量的秩内：轴号越界时定标会按一个不存在的维广播。
  // 秩为 1 时只有轴 0，而这个单元的默认轴号是 1（通道轴），对一维张量不适用，
  // 所以只在秩大于 1 时校验。
  FpsuSpecAttr spec = getSpec();
  if (spec.getSpc() && srcTy.getRank() > 1) {
    int64_t axis = spec.getSpcAxis();
    if (axis < 0 || axis >= srcTy.getRank())
      return emitOpError("spcAxis ")
             << axis << " is out of range for a rank-" << srcTy.getRank()
             << " source";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// KantorOp
//===----------------------------------------------------------------------===//

LogicalResult KantorOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  KantorMode mode = getSpec().getMode();

  // The multiply modes read a second operand; the conversions do not. A
  // float-to-fixed conversion has nothing to multiply by, and a multiply
  // without a multiplier would produce garbage rather than fail.
  bool multiplies = mode == KantorMode::ElementwiseMulFp16 ||
                    mode == KantorMode::ElementwiseMulFixedPoint ||
                    mode == KantorMode::FloatEltwiseAndScale;
  if (multiplies && !getRhs())
    return emitOpError("mode ")
           << stringifyKantorMode(mode) << " requires a rhs operand";
  if (!multiplies && getRhs())
    return emitOpError("mode ")
           << stringifyKantorMode(mode) << " takes no rhs operand";

  // The fixed-point conversion is defined as a scaling by 2^-shift, so a
  // conversion without the shift has no defined result -- and a lowering that
  // spelled the shift out itself would keep compiling while ignoring whatever
  // the IR says. Requiring it here is what makes the field load-bearing.
  if (mode == KantorMode::Fp2IntConverter && !getShift())
    return emitOpError(
        "the float-to-fixed conversion needs the shift attribute; without it "
        "the scaling is undefined and a lowering can only hardcode one");

  if (auto scale = getScale()) {
    auto ty = cast<RankedTensorType>(scale.getType());
    if (ty.getElementType() !=
        cast<RankedTensorType>(getLhs().getType()).getElementType())
      return emitOpError("scale must have lhs's element type; got ")
             << ty.getElementType();
  }
  if (auto bias = getBias()) {
    auto ty = cast<RankedTensorType>(bias.getType());
    if (!ty.getElementType().isF32())
      return emitOpError("bias must be f32; got ") << ty.getElementType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EltwiseOp
//===----------------------------------------------------------------------===//

LogicalResult EltwiseOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  // An elementwise op combines operands, so one is not an operation -- and the
  // accessors the rest of this dialect uses (`getLhs`, `getRhs`) would read out
  // of bounds.
  if (getOperands().size() < 2)
    return emitOpError("takes at least two operands; got ")
           << getOperands().size();

  auto lhsTy = cast<RankedTensorType>(getLhs().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  if (lhsTy.getShape() != resTy.getShape())
    return emitOpError("result shape [")
           << resTy.getShape() << "] must match lhs shape [" << lhsTy.getShape()
           << "]";

  // Every operand after the first either matches slot 0 exactly or broadcasts
  // against its trailing dimensions, which is how a per-channel factor, a bias,
  // or RoPE's shared cos/sin table is supplied.
  for (auto [slot, operand] : llvm::enumerate(getOperands().drop_front())) {
    auto ty = cast<RankedTensorType>(operand.getType());
    if (ty.getShape() == lhsTy.getShape())
      continue;
    if (ty.getRank() > lhsTy.getRank())
      return emitOpError("operand ")
             << (slot + 1) << " rank " << ty.getRank() << " exceeds lhs rank "
             << lhsTy.getRank();
    unsigned offset = lhsTy.getRank() - ty.getRank();
    for (int64_t i = 0; i < ty.getRank(); ++i) {
      int64_t r = ty.getDimSize(i), l = lhsTy.getDimSize(i + offset);
      if (r != l && r != 1)
        return emitOpError("operand ")
               << (slot + 1) << " shape [" << ty.getShape()
               << "] is not broadcastable against lhs shape ["
               << lhsTy.getShape() << "]";
    }
  }

  // A per-slot datapath list describes the slots, so it has to have one entry
  // each -- a short list would silently leave the tail slots on slot 0's
  // scaling, which is a different operation from the one written down.
  if (auto perSlot = getPerSlotDatapath()) {
    if (perSlot->size() != getOperands().size())
      return emitOpError("perSlotDatapath has ")
             << perSlot->size() << " entries for " << getOperands().size()
             << " operands; it needs exactly one per slot";
    for (Attribute entry : *perSlot)
      if (!isa<DatapathAttr>(entry))
        return emitOpError("perSlotDatapath must contain only #pim.datapath "
                           "entries");
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
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto tableTy = cast<RankedTensorType>(getTable().getType());
  auto indicesTy = cast<RankedTensorType>(getIndices().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  // Indices address rows. A float index would have to be rounded somewhere, and
  // wherever that happened would be a silent decision about which row is read.
  if (!indicesTy.getElementType().isIntOrIndex())
    return emitOpError("indices must be an integer type; got ")
           << indicesTy.getElementType();

  if (tableTy.getRank() < 1)
    return emitOpError("the table needs at least one dimension to index");

  // The result is the indices' shape followed by the table's row shape: one
  // looked-up row per index.
  SmallVector<int64_t> expected(indicesTy.getShape().begin(),
                               indicesTy.getShape().end());
  expected.append(tableTy.getShape().begin() + 1, tableTy.getShape().end());
  if (resTy.getShape() != ArrayRef<int64_t>(expected))
    return emitOpError("result shape [")
           << resTy.getShape() << "] must be the indices shape followed by the "
              "table's row shape, i.e. ["
           << expected << "]";

  if (resTy.getElementType() != tableTy.getElementType())
    return emitOpError("a lookup copies rows, so the element type is the "
                       "table's (")
           << tableTy.getElementType() << "); got " << resTy.getElementType();

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalPoolOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalPoolOp::verify() {
  if (failed(verifyComputeUnit(getOperation(), getUnit())))
    return failure();

  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  int64_t groupSize = getGroupSize();
  if (groupSize <= 0)
    return emitOpError("groupSize must be positive; got ") << groupSize;

  int64_t extent = srcTy.getShape().back();
  if (extent % groupSize != 0)
    return emitOpError("the reduced axis (")
           << extent << ") is not divisible by groupSize " << groupSize;

  // Each group collapses to one value and the rank is kept, so the shape is the
  // source's with its last axis divided by the group size.
  SmallVector<int64_t> expected(srcTy.getShape().begin(), srcTy.getShape().end());
  expected.back() = extent / groupSize;
  if (resTy.getShape() != ArrayRef<int64_t>(expected))
    return emitOpError("result shape [")
           << resTy.getShape() << "] must be [" << expected << "]";

  if (resTy.getElementType() != srcTy.getElementType())
    return emitOpError("a reduction keeps the element type; got ")
           << srcTy.getElementType() << " in and " << resTy.getElementType()
           << " out";

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

  // Only two chains exist in the reference graph, and they use exactly these
  // two card values: 0 (Q chain, result stays fp16 for the following dynamic
  // quantize) and 3 (K chain, requantized on its way into the KV cache).
  // Accepting any other integer would let a typo through as a card value the
  // hardware has no addressing mode for -- the expansion would still run and
  // still verify.
  int64_t tailCard = getTailCardValue();
  if (tailCard != 0 && tailCard != 3)
    return emitOpError("tailCardValue must be 0 (Q chain, fp16 tail) or "
                       "3 (K chain, requantized into the cache), got ")
           << tailCard;

  // 6 个编号定点子块，顺序固定。顺序就是语义：读回侧按位置展开，乱序会让
  // 整块字段错位而不报错。广播是 broadcastSpec 的事，不是子块名——
  // `Sin_Broadcast` 这种名字会让读回侧少掉一个真正的子块。
  // 这份名单与 pim-compiler 的 contracts/gml_hw_constants.py::ROPE_UNITS 逐字
  // 相同，改一边必须改另一边。
  if (auto blocks = getSubBlocks()) {
    static const char *order[] = {
        "Llama2Activation_Add_Cos", "Llama2Activation_Add_Sin",
        "Llama2Activation_Sin",     "Llama2Activation_Sin",
        "Llama2Activation_Cos",     "Llama2Activation_Cos"};
    constexpr size_t count = sizeof(order) / sizeof(order[0]);
    if (blocks->size() != count)
      return emitOpError("subBlocks 必须正好 ")
             << count << " 个，收到 " << blocks->size();
    for (auto [index, attr] : llvm::enumerate(*blocks)) {
      auto name = dyn_cast<StringAttr>(attr);
      if (!name || name.getValue() != order[index])
        return emitOpError("subBlocks 顺序不对，第 ")
               << index << " 个应为 " << order[index];
    }
  }

  // 广播轴必须落在源张量的秩内：轴号写错时广播会静默作用到别的维上。
  if (auto spec = getBroadcastSpec()) {
    int64_t rank = srcTy.getRank();
    for (int64_t axis : spec->getAxes().asArrayRef())
      if (axis < 0 || axis >= rank)
        return emitOpError("广播轴超出秩");
  }

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

  // RMS normalization divides by the root mean square and subtracts no mean, so
  // there is no shift for a bias to apply. Carrying one would describe an
  // affine step the operator does not perform.
  if (getRmsNorm() && getBias())
    return emitOpError("rmsNorm subtracts no mean, so it takes no bias");

  // The epsilon buffer holds one value, read as fp32. A wider shape would mean
  // a per-element epsilon, which the unit does not do.
  if (auto eps = getEpsilon()) {
    auto ty = cast<RankedTensorType>(eps.getType());
    if (ty.getNumElements() != 1)
      return emitOpError("epsilon is a single value; got ")
             << ty.getNumElements() << " elements";
    if (!ty.getElementType().isF32())
      return emitOpError("epsilon is read as f32; got ")
             << ty.getElementType();
  }

  // The scale set is per-tensor int8 with its own fp32 scale -- not the int4
  // per-group layout the projections use. Those two paths are entirely
  // different on the way to the graph format, and the element type is what
  // separates them.
  if (auto weight = getWeight()) {
    auto ty = cast<RankedTensorType>(weight.getType());
    if (!ty.getElementType().isInteger(8) && !ty.getElementType().isF16() &&
        !ty.getElementType().isF32())
      return emitOpError("the scale tensor is int8 (per-tensor) or float; got ")
             << ty.getElementType();
  }

  // Normalization runs on the vector unit. Its carrying none of the
  // fixed-function datapath fields is the point: filling one in would emit keys
  // the reference graph does not have for this operator, which reads as a
  // different node. `pim-verify-gml-contract` checks the same thing across the
  // whole module; this catches it at the op.
  for (StringRef forbidden : {"datapath", "fpsu", "kantor"})
    if ((*this)->hasAttr(forbidden))
      return emitOpError("must not carry ")
             << forbidden
             << ": it runs on the vector unit, and an empty datapath field set "
                "is what says so";

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

// The data extension digit each element type lands on. Only these two appear
// in the target's graphs, and the digit is a property of the type rather than
// of any particular node -- which is why the op carries it optionally instead
// of requiring it on every edge.
static std::optional<int64_t> derivedDataExtension(Type elemTy) {
  if (elemTy.isF16() || elemTy.isBF16())
    return 3;
  if (elemTy.isInteger(8))
    return 1;
  return std::nullopt;
}

LogicalResult ConvertOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());

  for (auto [label, attr, ty] :
       {std::tuple{"srcExtension", getSrcExtension(), srcTy},
        std::tuple{"dstExtension", getDstExtension(), resTy}}) {
    if (!attr)
      continue;
    auto derived = derivedDataExtension(ty.getElementType());
    if (!derived)
      return emitOpError()
             << label << " is given, but the element type "
             << ty.getElementType()
             << " has no extension digit to check it against";
    if (static_cast<int64_t>(*attr) != *derived)
      return emitOpError()
             << label << " is " << *attr << " but the element type "
             << ty.getElementType() << " implies " << *derived
             << "; a digit that disagrees with the type describes a layout no "
                "reader can act on";
  }

  if (srcTy.getShape() != resTy.getShape())
    return emitOpError("conversion keeps the shape; got ")
           << srcTy.getShape() << " -> " << resTy.getShape();
  // 两侧元素类型相同就不是一次转换。扩展位是元素类型的函数（f16→3、
  // i8→1），类型不变时扩展位也不可能变，所以"只改扩展位"不存在。
  if (srcTy.getElementType() == resTy.getElementType())
    return emitOpError("both sides are ")
           << srcTy.getElementType()
           << ", which is not a conversion; the extension digit follows the "
              "element type, so it cannot change on its own";

  return success();
}

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

  // `onthefly` says the data needs no pass of its own, which is a claim about
  // the layout rather than about the permutation: the two are separately
  // checkable, and disagreeing means one of them was set by mistake.
  if (getOnthefly() && getPurpose() &&
      getPurpose()->getPurpose() != TransposePurpose::Absorbed)
    return emitOpError("onthefly marks the permutation as free of a layer, but "
                       "the purpose is ")
           << stringifyTransposePurpose(getPurpose()->getPurpose())
           << ", which does produce one";

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

  // The mask is **additive fp16**, not boolean: masking adds a large negative
  // number so the following softmax drives that position to zero. An i1 mask
  // would have to be converted somewhere, and wherever that happened would
  // decide the masked value -- which is the numerics, not a detail.
  auto maskElem = cast<RankedTensorType>(getMask().getType()).getElementType();
  if (maskElem.isInteger(1))
    return emitOpError("the mask is additive (fp16), not boolean; an i1 mask "
                       "leaves the masked value undecided");

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

  // The two geometries are not just different extents, and once a batch axis is
  // 1 the shapes alone no longer say which one is meant -- hence the attribute,
  // and hence checking it against the shape rather than trusting either alone.
  //
  //   vector      (decode):  one query position against the whole cache, so the
  //                          mask is a row -- everything but the last axis is 1.
  //   causal_tril (prefill): a square where each position sees only what
  //                          precedes it, so the last two axes agree with the
  //                          scores'.
  if (maskTy.getRank() >= 2) {
    if (getLayout() == MaskLayout::Vector) {
      for (int64_t i = 0; i + 1 < maskTy.getRank(); ++i)
        if (maskTy.getDimSize(i) != 1)
          return emitOpError("layout vector masks a single query position, so "
                             "every axis but the last is 1; got mask shape [")
                 << maskTy.getShape() << "]";
    } else {
      int64_t rows = maskTy.getDimSize(maskTy.getRank() - 2);
      int64_t cols = maskTy.getDimSize(maskTy.getRank() - 1);
      if (rows != cols)
        return emitOpError("layout causal_tril masks a square, so the last two "
                           "axes agree; got mask shape [")
               << maskTy.getShape() << "]";
    }
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

  // 定标只在定点通路上成立：浮点模式下那对 scale/zp 没有含义。放在
  // 模式分支之前——scatter 分支自己就 return，放后面对它不生效。
  if (auto spec = getFpsuSpec())
    if (spec->getMode() != FpsuMode::FixedPoint)
      return emitOpError("fpsuSpec 要求 mode 为 fixed_point，收到 ")
             << stringifyFpsuMode(spec->getMode());

  // `isRead` is the older spelling of `mode = read`, kept as assembly sugar for
  // one release. Two spellings of one fact have to agree, or a reader cannot
  // tell which one was meant.
  if (getIsRead() && getMode() != KvMode::Read)
    return emitOpError("the `read` keyword and mode = ")
           << stringifyKvMode(getMode())
           << " disagree; the keyword is the older spelling of mode = read";

  // A scatter takes its destination from the index, so the index is what makes
  // it a scatter. A range write walks contiguous addresses and has nothing to
  // index with -- an index here would be read by nobody.
  // 散写是 mode 的默认值，也是最常用的写入形态。它的索引约束在这里查完
  // 就结束，**不能提前返回**——元素类型与容量两条是所有模式共用的，散写
  // 提前返回会让默认形态悄悄绕过它们。
  if (getMode() == KvMode::Scatter) {
    if (!getIndices())
      return emitOpError("a scatter takes its destination from an index, so it "
                         "needs an indices operand");
    if (!getIndices().getType().getElementType().isInteger(16))
      return emitOpError("scatter indices must be i16; a wider index would "
                         "name a cache row the hardware cannot reach, and a "
                         "narrower one cannot");
  } else if (getIndices()) {
    return emitOpError("mode = ") << stringifyKvMode(getMode())
                                  << " walks contiguous addresses, so it has "
                                     "nothing to index with";
  }


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
