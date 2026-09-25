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
// QuantSpecAttr / DatapathAttr / WindowAttr
//===----------------------------------------------------------------------===//

// fp16's largest finite value. The clamped role bounds to exactly this, and a
// different bound would be a different operation.
static constexpr double kF16Max = 65504.0;

LogicalResult
QuantSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      QuantGranularity granularity, int64_t axis,
                      int64_t groupSize, DataExtension dataExt, QuantRole role,
                      TypeAttr fpDtype, ArrayAttr range, bool spc,
                      int64_t spcAxis, bool spg, int64_t spgAxis,
                      int64_t spgGroupSize) {
  if (range) {
    if (range.size() != 2)
      return emitError() << "range is [min, max]; got " << range.size()
                         << " entries";
    auto lo = dyn_cast<FloatAttr>(range[0]);
    auto hi = dyn_cast<FloatAttr>(range[1]);
    if (!lo || !hi)
      return emitError() << "range entries must be floats";
    if (lo.getValueAsDouble() > hi.getValueAsDouble())
      return emitError() << "range min must not exceed max";
  }

  // A placeholder is not really quantized, so a value range would describe a
  // bound that is not applied -- and a consumer reading it would emit
  // quantization fields for a tensor that has none.
  if (role == QuantRole::Transparent && range)
    return emitError() << "a transparent spec is a placeholder and takes no "
                          "range";

  // The clamped role exists precisely because the element type cannot express
  // the bound: the type stays f16 and the limit is real. So the bound has to be
  // there, and it has to be fp16's own.
  if (role == QuantRole::FpClamp) {
    if (!range)
      return emitError() << "role fp_clamp needs an explicit range; the "
                            "element type does not imply the bound";
    double lo = cast<FloatAttr>(range[0]).getValueAsDouble();
    double hi = cast<FloatAttr>(range[1]).getValueAsDouble();
    if (lo != -kF16Max || hi != kF16Max)
      return emitError() << "role fp_clamp clamps to fp16's range ["
                         << -kF16Max << ", " << kF16Max << "]; got [" << lo
                         << ", " << hi << "]";
  }

  if (fpDtype && !isa<FloatType>(fpDtype.getValue()))
    return emitError() << "fpDtype names a floating-point type; got "
                       << fpDtype.getValue();

  // 逐通道 / 逐组是决策本身，粒度必须与它一致：per_tensor 没有通道可分，
  // 只有 per_group 才谈得上分组。
  bool wantSpc = granularity != QuantGranularity::PerTensor;
  bool wantSpg = granularity == QuantGranularity::PerGroup;
  if (spc != wantSpc)
    return emitError() << "spc is " << (spc ? "on" : "off")
                       << ", which disagrees with granularity "
                       << stringifyQuantGranularity(granularity)
                       << "; per_tensor has no channel to scale per";
  if (spg != wantSpg)
    return emitError() << "spg is " << (spg ? "on" : "off")
                       << ", which disagrees with granularity "
                       << stringifyQuantGranularity(granularity)
                       << "; grouping is what per_group means";
  if (wantSpg && spgGroupSize != groupSize)
    return emitError() << "spgGroupSize " << spgGroupSize
                       << " must equal groupSize " << groupSize
                       << "; the two name the same grouping";
  if (!wantSpg && (spgAxis != -1 || spgGroupSize != 0))
    return emitError() << "grouping is off, so spgAxis must be -1 and "
                          "spgGroupSize 0";

  return verifyLayout(emitError, granularity, axis, groupSize);
}

// The layout half, split out so the role checks above read as one block.
LogicalResult
QuantSpecAttr::verifyLayout(function_ref<InFlightDiagnostic()> emitError,
                            QuantGranularity granularity, int64_t axis,
                            int64_t groupSize) {
  // A single scale for the whole tensor has no axis and no group, so carrying
  // either would describe a layout the granularity does not have.
  if (granularity == QuantGranularity::PerTensor) {
    if (axis != 0 || groupSize != 0)
      return emitError() << "per_tensor takes no axis or groupSize";
    return success();
  }

  if (axis < 0)
    return emitError() << "axis must be non-negative; got " << axis;

  if (granularity == QuantGranularity::PerGroup) {
    if (groupSize <= 0)
      return emitError() << "per_group requires a positive groupSize";
  } else if (groupSize != 0) {
    return emitError() << "groupSize is only meaningful for per_group";
  }

  return success();
}

namespace mlir::triton::pim {

// spc/spg as each hardware block writes them for one quantization decision.
//
// The per-channel and per-group flags are not three independent settings: the
// fixed-point unit rescales with them, the pooling unit reduces groups of them
// and the elementwise-multiply unit dequantizes them. One decision, projected
// onto three blocks -- so they are derived from the spec rather than looked up,
// because a table holds exactly one configuration and would answer for the
// wrong one silently.
//
// Each block has its own axis numbering, and it is *not* the spec's tensor
// axis: the fixed-point unit numbers the channel axis 1, pooling and the
// elementwise-multiply unit number the channel axis 2 and the group axis 3.
// The two numberings must not be mixed. The table, with the target's GML field
// each row was measured from:
//
//   | hardware block           | spc axis | spg axis            | source |
//   | fixed-point unit (FPSU)  | 1        | top level writes no groupSize | TOP_LEVEL |
//   | pooling unit             | 2        | 3                   | DQ phase 0 |
//   | elementwise mul (KANTOR) | 2        | 3                   | DQ phase 3 |
//
// The fixed-point unit never groups -- it only applies the per-channel scale --
// so `spg` is never set for it (`fpsu_spg` is 0 in every node of the target's
// graph), and the group fields it does write spell -1.
SpcSpg deriveHardwareAxes(QuantSpecAttr spec, HardwareBlock block) {
  int64_t spcAxis = 1;
  int64_t spgAxis = -1;
  switch (block) {
  case HardwareBlock::Fpsu:
    break;
  case HardwareBlock::Pooling:
  case HardwareBlock::KantorA:
    spcAxis = 2;
    spgAxis = 3;
    break;
  }

  // A single scale for the whole tensor means the unit does not index its
  // constants per channel, so `spc` is off for `per_tensor`.
  bool spc = spec.getGranularity() != QuantGranularity::PerTensor;

  // Grouping is a property of the spec *and* of the block: only the pooling and
  // elementwise-multiply units reduce along groups.
  bool spg = spec.getGranularity() == QuantGranularity::PerGroup &&
             block != HardwareBlock::Fpsu;

  return SpcSpg{spc, spcAxis, spg, spg ? spgAxis : -1,
                spg ? spec.getGroupSize() : -1};
}

LogicalResult
PhaseSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      int64_t index, int64_t bytes, FunctionalUnit unit,
                      bool forceConsecutive, ArrayAttr reads) {
  if (index < 0)
    return emitError() << "phase index must be non-negative; got " << index;

  // A phase that streams nothing occupies no traversal, so it is not a phase --
  // admitting one would inflate every consumer's phase count.
  if (bytes <= 0)
    return emitError() << "phase bytes must be positive; got " << bytes;

  // `reads` describes the fan-out topology, which is a DAG: a phase reads only
  // strictly earlier phases. A self-reference or a forward reference would
  // describe a cycle the hardware cannot schedule, and getting this backwards
  // is silent -- dynamic quantization's serialized form is off by 256x.
  if (reads) {
    for (Attribute entry : reads) {
      auto read = dyn_cast<IntegerAttr>(entry);
      if (!read)
        return emitError() << "reads must contain only integer phase indices";
      if (read.getInt() < 0 || read.getInt() >= index)
        return emitError() << "phase " << index << " cannot read phase "
                           << read.getInt()
                           << "; reads must name a strictly earlier phase";
    }
  }

  return success();
}

LogicalResult
FpsuSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError, FpsuMode mode,
                     bool spc, int64_t spcAxis, bool spg, int64_t spgAxis,
                     int64_t spgGroupSize) {
  // The flags say the unit indexes its constants per channel / per group, so the
  // axis they index along has to be a real one. The rank is not known here (an
  // attribute has no tensor), so this is the part that can be checked in
  // isolation; whether the axis exists is the op's business.
  if (spc && spcAxis < 0)
    return emitError() << "spcAxis must be non-negative when spc is set; got "
                       << spcAxis;

  if (spg) {
    if (spgAxis < 0)
      return emitError() << "spgAxis must be non-negative when spg is set; got "
                         << spgAxis;
    if (spgGroupSize <= 0)
      return emitError() << "spgGroupSize must be positive when spg is set; got "
                         << spgGroupSize;
  }

  return success();
}

LogicalResult
ContractionAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        ContractionForm form, StringAttr blockName,
                        StringAttr innerOp, StringAttr actKind,
                        StringAttr flagName) {
  // The two forms are mutually exclusive: a nested block that also carries a
  // flat flag, or a flag that also carries a block name, describes a fusion the
  // graph format cannot write.
  if (form == ContractionForm::Named) {
    if (!blockName || !innerOp || !actKind)
      return emitError() << "form named needs blockName, innerOp and actKind";
    if (flagName)
      return emitError() << "form named takes no flagName; that belongs to the "
                            "flat form";
  } else {
    if (!flagName)
      return emitError() << "form flat needs flagName";
    if (blockName || innerOp || actKind)
      return emitError() << "form flat takes no blockName/innerOp/actKind; "
                            "those belong to the named form";
  }
  return success();
}

LogicalResult WeightBindingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, WeightFormat format,
    WeightRole role, int64_t elemBits, int64_t groupSize, int64_t sfMultiplier,
    StringAttr contentHash) {
  // int4 is stored one byte per element, so there is no packing width to get
  // wrong -- but there are exactly two widths a weight path operand has.
  if (elemBits != 4 && elemBits != 8)
    return emitError() << "elemBits is 4 (model weight) or 8 (activation used "
                          "as a weight); got "
                       << elemBits;

  if (role == WeightRole::ModelWeight && elemBits != 4)
    return emitError() << "a model weight is int4; got elemBits = " << elemBits;
  if (role == WeightRole::ActivationAsWeight && elemBits != 8)
    return emitError()
           << "an activation used as a weight is int8; got elemBits = "
           << elemBits;

  if (groupSize <= 0)
    return emitError() << "groupSize must be positive; got " << groupSize;

  // The guard is undone by a shift, so a non-power-of-two cannot be expressed
  // and the weight would come out scaled by the wrong amount.
  if (sfMultiplier <= 0 || (sfMultiplier & (sfMultiplier - 1)) != 0)
    return emitError() << "sfMultiplier must be a power of two; got "
                       << sfMultiplier;

  // The digest is a sha256 in hex. A raw 32-byte digest would be half this
  // length, which is the mistake worth catching here.
  if (contentHash) {
    StringRef digest = contentHash.getValue();
    if (digest.size() != 64)
      return emitError() << "contentHash must be a 64-character hex digest; got "
                         << digest.size() << " characters";
    if (!llvm::all_of(digest, [](char c) {
          return llvm::isHexDigit(c) && !llvm::isUpper(c);
        }))
      return emitError() << "contentHash must be lowercase hexadecimal";
  }

  return success();
}

} // namespace mlir::triton::pim

LogicalResult KantorSpecAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, KantorMode mode,
    int64_t cardValue, ArrayAttr blocks) {
  if (cardValue < -1)
    return emitError() << "cardValue must be -1 (use the mode's own value) or "
                          "a non-negative card value; got "
                       << cardValue;

  // Each physical block can be configured once; two entries naming the same one
  // would be two conflicting claims about one piece of hardware. The two blocks
  // are deliberately *not* required to carry the same fields -- the RoPE chain
  // has a scale on one and not on the other.
  if (blocks) {
    SmallPtrSet<StringAttr, 4> seen;
    for (Attribute entry : blocks) {
      auto block = dyn_cast<KantorBlockAttr>(entry);
      if (!block)
        return emitError() << "blocks must contain only #pim.kantor_block "
                              "entries";
      auto id = StringAttr::get(block.getContext(), block.getId());
      if (!seen.insert(id).second)
        return emitError() << "block " << block.getId()
                           << " is configured more than once";
    }
  }

  return success();
}

LogicalResult ActSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ActivationKind kind, LutMode mode,
                                  FloatAttr alpha, FloatAttr clipMin,
                                  FloatAttr clipMax) {
  // Only the bounded and sloped activations read these; carrying one on an
  // activation that ignores it would mislead a reader.
  if (kind != ActivationKind::ReluX && (clipMin || clipMax))
    return emitError() << "clip bounds are only meaningful for relu_x";

  if (kind != ActivationKind::LeakyRelu && alpha)
    return emitError() << "alpha is only meaningful for leaky_relu";

  if (clipMin && clipMax &&
      clipMin.getValue().convertToDouble() >
          clipMax.getValue().convertToDouble())
    return emitError() << "clipMin must not exceed clipMax";

  return success();
}

LogicalResult PoolSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   PoolKind kind, WindowAttr window) {
  // A global reduction covers the whole spatial extent, so a window would be
  // ignored; a windowed one has no geometry without it.
  bool global = kind == PoolKind::GlobalAverage;
  if (global && window)
    return emitError() << "global_average takes no window";
  if (!global && !window)
    return emitError() << "a windowed pool requires a window";

  return success();
}

LogicalResult
KantorBlockAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        StringRef id, KantorMode mode, QuantSpecAttr spec) {
  if (id.empty())
    return emitError() << "a kantor block must name the physical block it "
                          "configures";

  // A bypassed block reads no parameters, so a layout for them would describe
  // something that never happens.
  if (mode == KantorMode::Off && spec)
    return emitError() << "spec is meaningless when mode is off";

  return success();
}

LogicalResult DatapathAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   NumericMode nmuMode, NumericMode scaleMode,
                                   QuantSpecAttr scaleSpec,
                                   ArrayAttr kantorBlocks,
                                   bool groupDequantAccum, int64_t groupSize) {
  // The group size is what the accumulator dequantizes at, so it has to be
  // there -- and it has to be absent otherwise, or it would describe a grouping
  // the accumulator does not apply.
  if (groupDequantAccum && groupSize <= 0)
    return emitError() << "groupDequantAccum needs a positive groupSize; it is "
                          "the granularity the accumulator dequantizes at";
  if (!groupDequantAccum && groupSize != 0)
    return emitError() << "groupSize is only meaningful with "
                          "groupDequantAccum";

  // `fixed2float` describes a conversion the accumulator performs on its way
  // out; the scaling block that follows has nothing to convert.
  if (scaleMode == NumericMode::Fixed2Float)
    return emitError() << "scaleMode cannot be fixed2float";

  // Each physical block can only be configured once; two entries naming the
  // same one would be two conflicting claims about one piece of hardware.
  if (kantorBlocks) {
    if (kantorBlocks.empty())
      return emitError() << "kantorBlocks must not be an empty array; omit it "
                            "when no block is engaged";

    SmallPtrSet<StringAttr, 4> seen;
    for (Attribute entry : kantorBlocks) {
      auto block = dyn_cast<KantorBlockAttr>(entry);
      if (!block)
        return emitError() << "kantorBlocks must contain only "
                              "#pim.kantor_block entries";
      auto id = StringAttr::get(block.getContext(), block.getId());
      if (!seen.insert(id).second)
        return emitError() << "kantor block " << block.getId()
                           << " is configured more than once";
    }
  }

  return success();
}

LogicalResult BroadcastSpecAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, DenseI64ArrayAttr axesAttr,
    DenseI64ArrayAttr repeatsAttr) {
  if (!axesAttr || !repeatsAttr)
    return emitError() << "axes 与 repeats 都必填";
  ArrayRef<int64_t> axes = axesAttr.asArrayRef();
  ArrayRef<int64_t> repeats = repeatsAttr.asArrayRef();
  if (axes.empty())
    return emitError() << "axes 不能为空";
  if (axes.size() != repeats.size())
    return emitError() << "axes 与 repeats 必须一一对应";
  for (int64_t repeat : repeats)
    if (repeat <= 0)
      return emitError() << "repeats 的每一项必须为正，收到 " << repeat;
  return success();
}

LogicalResult WindowAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 DenseI64ArrayAttr kernelAttr,
                                 DenseI64ArrayAttr stridesAttr,
                                 DenseI64ArrayAttr padsAttr,
                                 DenseI64ArrayAttr dilationsAttr, int64_t group,
                                 DenseI64ArrayAttr outputPaddingAttr) {
  if (!kernelAttr || !stridesAttr || !padsAttr || !dilationsAttr)
    return emitError() << "kernel, strides, pads and dilations are all required";

  ArrayRef<int64_t> kernel = kernelAttr.asArrayRef();
  ArrayRef<int64_t> strides = stridesAttr.asArrayRef();
  ArrayRef<int64_t> pads = padsAttr.asArrayRef();
  ArrayRef<int64_t> dilations = dilationsAttr.asArrayRef();
  ArrayRef<int64_t> outputPadding =
      outputPaddingAttr ? outputPaddingAttr.asArrayRef() : ArrayRef<int64_t>{};

  unsigned rank = kernel.size();
  if (rank == 0)
    return emitError() << "kernel must not be empty";

  if (strides.size() != rank || dilations.size() != rank)
    return emitError() << "strides and dilations must have one entry per "
                          "kernel dimension ("
                       << rank << ")";

  // One pad per edge, i.e. two per spatial dimension: [top, right, bottom, left]
  // for the 2-D case.
  if (pads.size() != 2 * rank)
    return emitError() << "pads must have two entries per kernel dimension ("
                       << 2 * rank << ")";

  if (llvm::any_of(kernel, [](int64_t v) { return v <= 0; }))
    return emitError() << "kernel extents must be positive";
  if (llvm::any_of(strides, [](int64_t v) { return v <= 0; }))
    return emitError() << "strides must be positive";
  if (llvm::any_of(dilations, [](int64_t v) { return v <= 0; }))
    return emitError() << "dilations must be positive";
  if (llvm::any_of(pads, [](int64_t v) { return v < 0; }))
    return emitError() << "pads must be non-negative";

  if (group <= 0)
    return emitError() << "group must be positive";

  if (!outputPadding.empty() && outputPadding.size() != 2 * rank)
    return emitError() << "outputPadding, when present, must have two entries "
                          "per kernel dimension ("
                       << 2 * rank << ")";

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
                          StringRef(AttrTileWramBytesName),
                          StringRef(AttrL2BytesName),
                          StringRef(AttrL1BytesName),
                          StringRef(AttrRtlVersionName)},
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

std::optional<int64_t> mlir::triton::pim::maybeLookupL2Bytes(Operation *op) {
  return lookupModuleIntAttr(op, AttrL2BytesName);
}

std::optional<int64_t> mlir::triton::pim::maybeLookupL1Bytes(Operation *op) {
  return lookupModuleIntAttr(op, AttrL1BytesName);
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

LogicalResult
TransposePurposeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             TransposePurpose purpose, int64_t cardValue) {
  // A card value is the addressing mode a phase walks with, and only the two
  // phases that walk per-group scalars carry one. Zero is "no walk of its own":
  // a tensor transpose moves data by axis order, not by a card.
  if (cardValue < 0)
    return emitError() << "cardValue is a card number, so it cannot be "
                          "negative; got "
                       << cardValue;
  if (purpose == TransposePurpose::TensorTranspose && cardValue != 0)
    return emitError() << "a tensor transpose permutes axes rather than walking "
                          "a card, so it carries no cardValue; got "
                       << cardValue;
  return success();
}
