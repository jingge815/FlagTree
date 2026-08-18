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
