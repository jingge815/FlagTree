#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

using namespace mlir;
using namespace mlir::triton::pim;

#include "triton/Dialect/TritonPIM/IR/Dialect.cpp.inc"

#include "triton/Dialect/TritonPIM/IR/AttrEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonPIM/IR/AttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TaskletTiledEncodingAttr
//===----------------------------------------------------------------------===//

static unsigned productOf(ArrayRef<unsigned> vals) {
  return std::accumulate(vals.begin(), vals.end(), 1u,
                         std::multiplies<unsigned>());
}

unsigned TaskletTiledEncodingAttr::getTotalTaskletsPerDpu() const {
  return productOf(getTaskletsPerDpu());
}

unsigned TaskletTiledEncodingAttr::getTotalDpusPerDevice() const {
  return productOf(getDpusPerDevice());
}

unsigned TaskletTiledEncodingAttr::getTotalSizePerTasklet() const {
  return productOf(getSizePerTasklet());
}

// Reads one `name = [a, b, ...]` entry out of the attribute dictionary.
static LogicalResult parseUnsignedArray(AsmParser &parser,
                                        const NamedAttribute &attr,
                                        SmallVector<unsigned> &res,
                                        StringRef desc) {
  auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute elem : arrayAttr) {
    auto intAttr = dyn_cast<IntegerAttr>(elem);
    if (!intAttr) {
      parser.emitError(parser.getNameLoc(), "expected an integer in ") << desc;
      return failure();
    }
    // Integer literals parse as signless, so read them as signed and range
    // check rather than calling getUInt(), which asserts on signless types.
    int64_t value = intAttr.getInt();
    if (value < 0) {
      parser.emitError(parser.getNameLoc(), "expected a non-negative value in ")
          << desc;
      return failure();
    }
    res.push_back(static_cast<unsigned>(value));
  }
  return success();
}

Attribute TaskletTiledEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> sizePerTasklet, taskletsPerDpu, dpusPerDevice, order;
  for (const NamedAttribute &attr : dict) {
    StringRef name = attr.getName().strref();
    if (name == "sizePerTasklet") {
      if (parseUnsignedArray(parser, attr, sizePerTasklet, name).failed())
        return {};
    } else if (name == "taskletsPerDpu") {
      if (parseUnsignedArray(parser, attr, taskletsPerDpu, name).failed())
        return {};
    } else if (name == "dpusPerDevice") {
      if (parseUnsignedArray(parser, attr, dpusPerDevice, name).failed())
        return {};
    } else if (name == "order") {
      if (parseUnsignedArray(parser, attr, order, name).failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(),
                       "unexpected key in #pim.tasklet_tiled: ")
          << name;
      return {};
    }
  }

  // dpusPerDevice may be omitted; a kernel is single-DPU unless told otherwise.
  if (dpusPerDevice.empty())
    dpusPerDevice.assign(sizePerTasklet.size(), 1);

  return parser.getChecked<TaskletTiledEncodingAttr>(
      parser.getContext(), sizePerTasklet, taskletsPerDpu, dpusPerDevice, order);
}

void TaskletTiledEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerTasklet = [" << ArrayRef(getSizePerTasklet()) << "]"
          << ", taskletsPerDpu = [" << ArrayRef(getTaskletsPerDpu()) << "]";

  // Elide the all-ones default, keeping single-DPU kernels readable.
  if (!llvm::all_of(getDpusPerDevice(), [](unsigned v) { return v == 1; }))
    printer << ", dpusPerDevice = [" << ArrayRef(getDpusPerDevice()) << "]";

  printer << ", order = [" << ArrayRef(getOrder()) << "]"
          << "}>";
}

LogicalResult TaskletTiledEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<unsigned> sizePerTasklet, ArrayRef<unsigned> taskletsPerDpu,
    ArrayRef<unsigned> dpusPerDevice, ArrayRef<unsigned> order) {
  size_t rank = sizePerTasklet.size();
  if (rank == 0)
    return emitError() << "rank 0 layout is not allowed";

  if (taskletsPerDpu.size() != rank || dpusPerDevice.size() != rank ||
      order.size() != rank)
    return emitError() << "sizePerTasklet, taskletsPerDpu, dpusPerDevice and "
                          "order must all have the same rank; got "
                       << rank << ", " << taskletsPerDpu.size() << ", "
                       << dpusPerDevice.size() << ", " << order.size();

  for (auto [name, vals] :
       {std::pair{"sizePerTasklet", sizePerTasklet},
        std::pair{"taskletsPerDpu", taskletsPerDpu},
        std::pair{"dpusPerDevice", dpusPerDevice}}) {
    if (llvm::any_of(vals, [](unsigned v) { return v == 0; }))
      return emitError() << name << " must be positive in every dimension";
  }

  // `order` must be a permutation of [0, rank).
  SmallVector<bool> seen(rank, false);
  for (unsigned dim : order) {
    if (dim >= rank)
      return emitError() << "order contains out-of-range dimension " << dim
                         << " for rank " << rank;
    if (seen[dim])
      return emitError() << "order contains duplicate dimension " << dim;
    seen[dim] = true;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Layout inference
//===----------------------------------------------------------------------===//
//
// Shape-changing Triton ops (`expand_dims`, `trans`, `reduce`, `reshape`,
// `join`, `split`) do not know about any particular layout; they ask the
// defining dialect of their operand encoding to derive the result encoding. PIM
// layouts are plain per-dimension tuples, so each of these is an
// insert/drop/permute on four parallel arrays.

namespace {

// Drops entry `axis` from a per-dimension array.
static SmallVector<unsigned> dropAt(ArrayRef<unsigned> vec, unsigned axis) {
  SmallVector<unsigned> res(vec.begin(), vec.end());
  res.erase(res.begin() + axis);
  return res;
}

// Inserts `val` at `axis` in a per-dimension array.
static SmallVector<unsigned> insertAt(ArrayRef<unsigned> vec, unsigned axis,
                                      unsigned val) {
  SmallVector<unsigned> res(vec.begin(), vec.end());
  res.insert(res.begin() + axis, val);
  return res;
}

// Renumbers an `order` permutation after dimension `axis` was removed.
static SmallVector<unsigned> dropFromOrder(ArrayRef<unsigned> order,
                                           unsigned axis) {
  SmallVector<unsigned> res;
  res.reserve(order.size() - 1);
  for (unsigned dim : order) {
    if (dim == axis)
      continue;
    res.push_back(dim > axis ? dim - 1 : dim);
  }
  return res;
}

// Renumbers an `order` permutation after a dimension was inserted at `axis`,
// making the new dimension the slowest-changing one (it has extent 1, so its
// position in the order is arbitrary).
static SmallVector<unsigned> insertIntoOrder(ArrayRef<unsigned> order,
                                             unsigned axis) {
  SmallVector<unsigned> res;
  res.reserve(order.size() + 1);
  for (unsigned dim : order)
    res.push_back(dim >= axis ? dim + 1 : dim);
  res.push_back(axis);
  return res;
}

// Applies a permutation to a per-dimension array.
static SmallVector<unsigned> permute(ArrayRef<unsigned> vec,
                                     ArrayRef<int32_t> perm) {
  SmallVector<unsigned> res;
  res.reserve(perm.size());
  for (int32_t idx : perm)
    res.push_back(vec[idx]);
  return res;
}

struct TritonPIMInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult inferTransOpEncoding(Attribute operandEncoding,
                                     ArrayRef<int64_t> shape,
                                     ArrayRef<int32_t> order,
                                     Attribute &resultEncoding,
                                     std::optional<Location> loc) const override {
    auto enc = dyn_cast<TaskletTiledEncodingAttr>(operandEncoding);
    if (!enc)
      return emitOptionalError(loc, "expected a #pim.tasklet_tiled encoding");

    // Permuting the tensor permutes each per-dimension tuple. `order` itself
    // lists dimension indices, so it needs remapping rather than permuting:
    // an entry naming old dimension d now names the position d sits at.
    SmallVector<unsigned> invOrder(order.size());
    for (unsigned i = 0; i < order.size(); ++i)
      invOrder[order[i]] = i;
    SmallVector<unsigned> retOrder;
    retOrder.reserve(enc.getOrder().size());
    for (unsigned dim : enc.getOrder())
      retOrder.push_back(invOrder[dim]);

    resultEncoding = TaskletTiledEncodingAttr::get(
        enc.getContext(), permute(enc.getSizePerTasklet(), order),
        permute(enc.getTaskletsPerDpu(), order),
        permute(enc.getDpusPerDevice(), order), retOrder);
    return success();
  }

  LogicalResult inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                                      Attribute &resultEncoding,
                                      std::optional<Location> loc) const override {
    auto enc = dyn_cast<TaskletTiledEncodingAttr>(operandEncoding);
    if (!enc)
      return emitOptionalError(loc, "expected a #pim.tasklet_tiled encoding");

    // The reduced dimension disappears. Tasklets that covered it fold into the
    // slowest-changing surviving dimension, so the tasklet count stays exact --
    // a reduction on PIM is a barrier plus passes over WRAM, and every tasklet
    // still participates.
    SmallVector<unsigned> size = dropAt(enc.getSizePerTasklet(), axis);
    SmallVector<unsigned> tasklets = dropAt(enc.getTaskletsPerDpu(), axis);
    SmallVector<unsigned> dpus = dropAt(enc.getDpusPerDevice(), axis);
    SmallVector<unsigned> order = dropFromOrder(enc.getOrder(), axis);

    if (size.empty())
      return emitOptionalError(loc, "cannot reduce a rank-1 tensor to rank 0");

    unsigned lost = enc.getTaskletsPerDpu()[axis];
    if (lost > 1)
      tasklets[order.back()] *= lost;
    unsigned lostDpus = enc.getDpusPerDevice()[axis];
    if (lostDpus > 1)
      dpus[order.back()] *= lostDpus;

    resultEncoding = TaskletTiledEncodingAttr::get(enc.getContext(), size,
                                                   tasklets, dpus, order);
    return success();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const override {
    auto enc = dyn_cast<TaskletTiledEncodingAttr>(operandEncoding);
    if (!enc)
      return emitOptionalError(loc, "expected a #pim.tasklet_tiled encoding");

    // The new dimension has extent 1, so it gets one element and one tasklet.
    resultEncoding = TaskletTiledEncodingAttr::get(
        enc.getContext(), insertAt(enc.getSizePerTasklet(), axis, 1),
        insertAt(enc.getTaskletsPerDpu(), axis, 1),
        insertAt(enc.getDpusPerDevice(), axis, 1),
        insertIntoOrder(enc.getOrder(), axis));
    return success();
  }

  LogicalResult inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                                   Attribute retEncoding,
                                   std::optional<Location> loc) const override {
    // PIM has no tensor core, so `tt.dot` imposes no layout on its operands:
    // any tasklet layout is acceptable. Contrast TTGIR, where operands must
    // carry a #ttg.dot_op encoding to feed the MMA unit.
    if (!isa<TaskletTiledEncodingAttr>(operandEncoding))
      return emitOptionalError(loc, "expected a #pim.tasklet_tiled encoding");
    return success();
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    auto enc = dyn_cast<TaskletTiledEncodingAttr>(srcEnc);
    if (!enc)
      return emitOptionalError(loc, "expected a #pim.tasklet_tiled encoding");

    // Reshaping in general permutes which tasklet holds which element, and
    // there is no layout that makes it free. Rather than pretend otherwise,
    // hand back the default layout for the destination shape: correct, and the
    // relayout it implies becomes visible as a `pim.convert_layout`.
    dstEnc = getDefaultTaskletTiledEncoding(enc.getContext(), dstShape,
                                            enc.getTotalTaskletsPerDpu(),
                                            enc.getTotalDpusPerDevice());
    return success();
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    if (expected == got)
      return success();
    return emitOptionalError(loc, "layouts differ: expected ", expected,
                             " but got ", got);
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    auto enc = dyn_cast<TaskletTiledEncodingAttr>(srcEnc);
    if (!enc)
      return emitOptionalError(loc, "expected a #pim.tasklet_tiled encoding");

    // `join` appends a trailing dimension of extent 2, holding both halves in
    // the same tasklet so that no data movement is implied.
    unsigned rank = enc.getSizePerTasklet().size();
    resultEncodingForJoin(enc, rank, dstEnc);
    return success();
  }

  LogicalResult inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                                     ArrayRef<int64_t> shape,
                                     std::optional<Location> loc) const override {
    auto enc = dyn_cast<TaskletTiledEncodingAttr>(srcEnc);
    if (!enc)
      return emitOptionalError(loc, "expected a #pim.tasklet_tiled encoding");

    // The inverse of `join`: drop the trailing dimension, which `join` gave
    // two elements and one tasklet.
    unsigned rank = enc.getSizePerTasklet().size();
    if (rank < 2)
      return emitOptionalError(loc, "cannot split a rank-1 tensor");
    unsigned last = rank - 1;
    if (enc.getSizePerTasklet()[last] != 2 ||
        enc.getTaskletsPerDpu()[last] != 1)
      return emitOptionalError(
          loc, "split requires the last dimension to hold 2 elements in a "
               "single tasklet");

    dstEnc = TaskletTiledEncodingAttr::get(
        enc.getContext(), dropAt(enc.getSizePerTasklet(), last),
        dropAt(enc.getTaskletsPerDpu(), last),
        dropAt(enc.getDpusPerDevice(), last),
        dropFromOrder(enc.getOrder(), last));
    return success();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    if (!isa<TaskletTiledEncodingAttr>(operandEncodingA) ||
        !isa<TaskletTiledEncodingAttr>(operandEncodingB))
      return op->emitError("expected #pim.tasklet_tiled operand encodings");
    // No tensor core, so no compatibility constraint between the operands.
    return success();
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    // fp4 unpacking is a GPU-specific microscaling path with no PIM equivalent
    // yet; refusing here is better than inventing a layout for it.
    return emitOptionalError(
        loc, "tt.fp4_to_fp is not supported on PIM targets");
  }

private:
  // Shared by inferDefaultJoinOpEncoding: append a 2-element dimension. Unlike
  // the extent-1 dimension `expand_dims` adds, this one holds real data
  // contiguously within a tasklet, so it becomes the fastest-changing axis.
  static void resultEncodingForJoin(TaskletTiledEncodingAttr enc, unsigned rank,
                                    Attribute &dstEnc) {
    SmallVector<unsigned> order;
    order.reserve(rank + 1);
    order.push_back(rank);
    for (unsigned dim : enc.getOrder())
      order.push_back(dim);

    dstEnc = TaskletTiledEncodingAttr::get(
        enc.getContext(), insertAt(enc.getSizePerTasklet(), rank, 2),
        insertAt(enc.getTaskletsPerDpu(), rank, 1),
        insertAt(enc.getDpusPerDevice(), rank, 1), order);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void TritonPIMDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonPIM/IR/AttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonPIM/IR/Ops.cpp.inc"
      >();

  // Shape-changing Triton ops ask the encoding's own dialect how to derive the
  // result encoding, so PIM has to answer for #pim.tasklet_tiled.
  addInterfaces<TritonPIMInferLayoutInterface>();
}

LogicalResult TritonPIMDialect::verifyOperationAttribute(Operation *op,
                                                        NamedAttribute attr) {
  // The hardware description belongs to the module as a whole, not to
  // individual ops inside it.
  if (llvm::is_contained({StringRef(AttrNumDpusName),
                          StringRef(AttrNumTaskletsName),
                          StringRef(AttrWramBytesName),
                          StringRef(AttrMramBytesName),
                          StringRef(AttrDmaAlignName),
                          StringRef(AttrTargetName),
                          StringRef(AttrWramBytesUsedName),
                          StringRef(AttrTileMName),
                          StringRef(AttrTileNName),
                          StringRef(AttrTileKName),
                          StringRef(AttrTileWramBytesName)},
                         attr.getName().strref()) &&
      !isa<ModuleOp>(op)) {
    return op->emitOpError("has unexpected attribute ")
           << attr.getName() << " which is expected only on `module` ops";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Hierarchy lookups
//===----------------------------------------------------------------------===//

// Reads an integer module attribute from `op` itself (when `op` is the
// module) or from the module enclosing `op`. `getParentOfType` only walks
// strictly above `op`, so passes that call this with the `ModuleOp` they are
// running on (e.g. `getOperation()` in a module pass) need the op-itself case
// too, or the lookup always misses.
static std::optional<int64_t> lookupModuleIntAttr(Operation *op,
                                                  StringRef name) {
  auto mod = isa<ModuleOp>(op) ? cast<ModuleOp>(op)
                                : op->getParentOfType<ModuleOp>();
  if (!mod)
    return std::nullopt;
  if (auto attr = mod->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

int mlir::triton::pim::lookupNumTasklets(Operation *op) {
  return lookupModuleIntAttr(op, AttrNumTaskletsName)
      .value_or(kDefaultNumTasklets);
}

int mlir::triton::pim::lookupNumDpus(Operation *op) {
  return lookupModuleIntAttr(op, AttrNumDpusName).value_or(kDefaultNumDpus);
}

std::optional<int64_t> mlir::triton::pim::maybeLookupWramBytes(Operation *op) {
  return lookupModuleIntAttr(op, AttrWramBytesName);
}

std::optional<int64_t> mlir::triton::pim::maybeLookupMramBytes(Operation *op) {
  return lookupModuleIntAttr(op, AttrMramBytesName);
}

std::optional<int64_t> mlir::triton::pim::maybeLookupDmaAlign(Operation *op) {
  return lookupModuleIntAttr(op, AttrDmaAlignName);
}

//===----------------------------------------------------------------------===//
// Layout helpers
//===----------------------------------------------------------------------===//

TaskletTiledEncodingAttr mlir::triton::pim::getDefaultTaskletTiledEncoding(
    MLIRContext *context, ArrayRef<int64_t> shape, int numTasklets,
    int numDpus) {
  unsigned rank = shape.size();
  // One element per tasklet, row-major: the fastest-changing dimension is the
  // last one, so it comes first in `order`.
  SmallVector<unsigned> sizePerTasklet(rank, 1);
  SmallVector<unsigned> order(rank);
  for (unsigned i = 0; i < rank; ++i)
    order[i] = rank - 1 - i;

  return TaskletTiledEncodingAttr::get(context, shape, sizePerTasklet, order,
                                       numTasklets, numDpus);
}

std::optional<int64_t>
mlir::triton::pim::getTensorSizeInBytes(RankedTensorType type) {
  Type elemTy = type.getElementType();
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  if (bits % 8 != 0)
    return std::nullopt;

  if (!type.hasStaticShape())
    return std::nullopt;

  int64_t elems = 1;
  for (int64_t dim : type.getShape())
    elems *= dim;
  return elems * (bits / 8);
}
