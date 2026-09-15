#include "triton/Dialect/TritonPIM/IR/Types.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "triton/Dialect/TritonPIM/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::pim;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonPIM/IR/Types.cpp.inc"

static constexpr llvm::StringRef kImmutableMemory = "immutable";

// Format: !pim.memdesc<64x32xf16, #pim.wram> with an optional trailing
// `, immutable`. Buffers are mutable by default, which is the common case for
// staging: a buffer is DMA'd into, read, and reused next iteration.
Type MemDescType::parse(AsmParser &parser) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (failed(parser.parseLess()))
    return Type();

  SmallVector<int64_t> dimensions;
  if (failed(parser.parseDimensionList(dimensions, /*allowDynamic=*/false)))
    return Type();

  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();

  Attribute memorySpace;
  if (failed(parser.parseComma()) || failed(parser.parseAttribute(memorySpace)))
    return Type();

  bool mutableMemory = true;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword(kImmutableMemory)))
      return Type();
    mutableMemory = false;
  }

  if (parser.parseGreater())
    return Type();

  return MemDescType::getChecked(loc, parser.getContext(), dimensions,
                                 elementType, memorySpace, mutableMemory);
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  printer << ", " << getMemorySpace();
  if (!getMutableMemory())
    printer << ", " << kImmutableMemory;
  printer << ">";
}

LogicalResult MemDescType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  Attribute memorySpace, bool mutableMemory) {
  if (shape.empty())
    return emitError() << "rank 0 memdesc is not allowed";

  if (llvm::any_of(shape, [](int64_t dim) { return dim <= 0; }))
    return emitError() << "shape must have non-zero positive dimensions; got "
                       << shape;

  // Unlike TTGIR's memdesc, PIM does not require power-of-2 dimensions: a WRAM
  // buffer is a plain contiguous span with no swizzling to constrain it.

  if (!isa<WRAMSpaceAttr, MRAMSpaceAttr, L1SpaceAttr, L2SpaceAttr>(memorySpace))
    return emitError() << "memorySpace must be #pim.wram, #pim.mram, #pim.l1 "
                          "or #pim.l2; got "
                       << memorySpace;

  return success();
}

bool MemDescType::isWRAM() const { return isa<WRAMSpaceAttr>(getMemorySpace()); }

bool MemDescType::isMRAM() const { return isa<MRAMSpaceAttr>(getMemorySpace()); }

bool MemDescType::isL1() const { return isa<L1SpaceAttr>(getMemorySpace()); }

bool MemDescType::isL2() const { return isa<L2SpaceAttr>(getMemorySpace()); }

std::optional<int64_t> MemDescType::getSizeInBytes() const {
  Type elemTy = getElementType();
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  // Sub-byte element types would need a packing convention we have not fixed.
  if (bits % 8 != 0)
    return std::nullopt;

  int64_t elems = 1;
  for (int64_t dim : getShape()) {
    if (dim <= 0)
      return std::nullopt;
    elems *= dim;
  }
  return elems * (bits / 8);
}

void ::mlir::triton::pim::TritonPIMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonPIM/IR/Types.cpp.inc"
      >();
}
