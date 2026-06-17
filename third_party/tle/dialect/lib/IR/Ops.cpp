#include "mlir/IR/AffineExpr.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>
#include <set>
#include <string>

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir::triton::tle {

namespace {
// Triton shared-memory pointers map to LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
// Cluster-shared pointers map to LLVM address space 7 (NVVM shared::cluster).
constexpr int kClusterSharedMemoryAddressSpace = 7;
} // namespace

// ============================================================================
// ExtractTileOp Builder
// ============================================================================
void ExtractTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto resultType = RankedTensorType::get(tileShape, srcType.getElementType(),
                                          srcType.getEncoding());
  state.addOperands(src);
  state.addOperands(index);
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(resultType);
}

// ============================================================================
// ExtractTileOp Verification
//
// For dynamic index (index operand is not arith.constant):
//   - Only check constraints that are known at compile time: tile_shape
//   positivity, divisibility, element type, rank match
//   - Skip out-of-bounds and CTA tile alignment checks (only known at runtime)
//
// For static index: perform full checks (same as original implementation)
// ============================================================================
LogicalResult ExtractTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();

  // ---- Get tile_shape attribute ----
  auto tileShapeRawAttr = getOperation()->getAttr("tile_shape");
  SmallVector<int64_t> tileShape;
  if (auto denseArray64 =
          mlir::dyn_cast<mlir::DenseI64ArrayAttr>(tileShapeRawAttr)) {
    for (auto v : denseArray64.asArrayRef())
      tileShape.push_back(v);
  }

  // ---- Basic checks required for both static and dynamic index ----

  // Check 1: element types must match
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != dstTy.getRank())
    return emitError("result rank must equal source rank");

  // Check 3: tile_shape rank must match source rank
  if (tileShape.size() != srcShape.size())
    return emitOpError("tile_shape rank must match source rank");

  // Check 4: tile_shape must be positive in each dimension, divisible, and dst
  // shape must equal tile_shape
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile_shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile_shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    if (dstShape[i] != tileShape[i])
      return emitOpError("result shape must equal tile_shape at dimension ")
             << i;
  }

  // ---- Determine if index is a static constant ----
  // getDefiningOp<arith::ConstantOp>() returns nullptr for dynamic Value
  auto indexConstOp =
      getOperation()->getOperand(1).getDefiningOp<arith::ConstantOp>();

  if (!indexConstOp) {
    // Dynamic index: skip out-of-bounds and offset alignment checks, handled at
    // lowering stage
    return success();
  }

  // ---- Full checks for static index ----
  int64_t index =
      mlir::cast<mlir::IntegerAttr>(indexConstOp.getValue()).getInt();

  // Compute logical grid shape
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // Out-of-bounds check
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  // Delinearize to per-dimension tile indices (row-major order)
  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  // tile indices -> coordinate-level offsets
  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  // Boundary check
  if (offsets.size() != static_cast<size_t>(srcTy.getRank()))
    return emitError("offsets size must match tensor rank");

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] > srcShape[i])
      return emitOpError(
                 "result shape cannot exceed source shape at dimension ")
             << i;
    if (offsets[i] + dstShape[i] > srcShape[i])
      return emitOpError("invalid offset at dimension ")
             << i << ": offset(" << offsets[i] << ") + shape(" << dstShape[i]
             << ") > source(" << srcShape[i] << ")";
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
  }

  auto encoding = srcTy.getEncoding();
  if (!encoding)
    return success();
  return success();
}

LogicalResult MemDescWGMMAViewOp::verify() {
  auto srcType = getSrc().getType();
  auto resultType = getType();
  auto order = getOrder();

  if (srcType.getRank() != 2 || resultType.getRank() != 2)
    return emitOpError("expects rank-2 source and result memdesc types");
  if (order.size() != static_cast<size_t>(srcType.getRank()))
    return emitOpError("expects order rank to match source rank");
  llvm::SmallBitVector seen(order.size());
  for (int32_t dim : order) {
    if (dim < 0 || dim >= static_cast<int32_t>(order.size()) || seen[dim])
      return emitOpError("expects order to be a permutation");
    seen.set(dim);
  }
  if (srcType.getElementType() != resultType.getElementType())
    return emitOpError("expects source and result element types to match");
  if (srcType.getMemorySpace() != resultType.getMemorySpace())
    return emitOpError("expects source and result memory spaces to match");
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(srcType.getMemorySpace()))
    return emitOpError("expects shared memory descriptors");
  if (mlir::product(srcType.getShape()) != mlir::product(resultType.getShape()))
    return emitOpError("expects source and result to cover the same number of "
                       "elements");
  if (!isa<triton::gpu::NVMMASharedEncodingAttr,
           triton::gpu::SharedLinearEncodingAttr>(resultType.getEncoding()))
    return emitOpError("expects result to use a WGMMA-compatible shared "
                       "encoding");
  return success();
}

LogicalResult WGMMASharedOperandFenceOp::verify() {
  if (getDeps().empty())
    return emitOpError("expects at least one shared-memory operand");

  for (Value dep : getDeps()) {
    auto type = cast<triton::gpu::MemDescType>(dep.getType());
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(type.getMemorySpace()))
      return emitOpError("expects only shared-memory operands");
  }
  return success();
}

static bool isAsciiIdentStart(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool isAsciiIdentChar(char c) {
  return isAsciiIdentStart(c) || (c >= '0' && c <= '9') || c == '_';
}

static bool isValidPublicPipeName(StringRef name) {
  if (name.empty() || name == "fields" || name == "readers" ||
      name.starts_with("_") || !isAsciiIdentStart(name.front()))
    return false;
  return llvm::all_of(name.drop_front(), isAsciiIdentChar);
}

static LogicalResult verifyPipeNameArray(Operation *op, ArrayAttr namesAttr,
                                         StringRef attrName, bool allowEmpty) {
  if (!allowEmpty && namesAttr.empty())
    return op->emitOpError("expects ")
           << attrName << " to contain at least one name";

  llvm::SmallSet<StringRef, 8> seenNames;
  for (Attribute attr : namesAttr) {
    auto nameAttr = dyn_cast<StringAttr>(attr);
    if (!nameAttr)
      return op->emitOpError("expects ")
             << attrName << " to contain only strings";
    StringRef name = nameAttr.getValue();
    if (!isValidPublicPipeName(name))
      return op->emitOpError("expects valid public pipe ")
             << attrName << " names";
    if (!seenNames.insert(name).second)
      return op->emitOpError("expects unique pipe ") << attrName << " names";
  }

  return success();
}

static LogicalResult verifyPipeAttrs(Operation *op, OperandRange fields) {
  auto capacityAttr = op->getAttrOfType<IntegerAttr>("capacity");
  if (!capacityAttr)
    return op->emitOpError("requires capacity attribute");
  int64_t capacity = capacityAttr.getInt();
  if (capacity <= 0)
    return op->emitOpError("requires positive capacity");

  auto scopeAttr = op->getAttrOfType<StringAttr>("scope");
  if (!scopeAttr)
    return op->emitOpError("requires scope attribute");
  if (scopeAttr.getValue() != "cta")
    return op->emitOpError("MVP supports only scope = \"cta\"");

  auto fieldNamesAttr = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNamesAttr)
    return op->emitOpError("requires field_names attribute");
  if (fieldNamesAttr.size() != fields.size())
    return op->emitOpError("expects field_names size to match field operands");

  if (failed(verifyPipeNameArray(op, fieldNamesAttr, "field", false)))
    return failure();

  if (auto readersAttr = op->getAttrOfType<ArrayAttr>("readers")) {
    if (failed(verifyPipeNameArray(op, readersAttr, "reader", false)))
      return failure();
  }

  if (auto readerNameAttr = op->getAttrOfType<StringAttr>("reader_name")) {
    if (!isValidPublicPipeName(readerNameAttr.getValue()))
      return op->emitOpError("expects valid public pipe reader_name");
  }

  if (fields.empty())
    return op->emitOpError("expects at least one pipe field");
  for (Value field : fields) {
    auto type = cast<triton::gpu::MemDescType>(field.getType());
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(type.getMemorySpace()))
      return op->emitOpError("expects only shared-memory pipe fields");
    if (type.getRank() < 2)
      return op->emitOpError("expects pipe fields to have rank >= 2");
    if (type.getShape()[0] != capacity)
      return op->emitOpError("expects field leading dimension to equal "
                             "pipe capacity");
  }
  return success();
}

static LogicalResult verifyPipeStagePhase(Operation *op, Value stage,
                                          Value phase) {
  if (!stage.getType().isInteger(32))
    return op->emitOpError("expects stage to be i32");
  if (!phase.getType().isInteger(1))
    return op->emitOpError("expects phase to be i1");
  return success();
}

static LogicalResult verifyPipeStage(Operation *op, Value stage) {
  if (!stage.getType().isInteger(32))
    return op->emitOpError("expects stage to be i32");
  return success();
}

LogicalResult PipeCreateOp::verify() {
  return verifyPipeAttrs(getOperation(), getFields());
}

LogicalResult PipeWriterAcquireOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStagePhase(getOperation(), getStage(), getPhase());
}

LogicalResult PipeWriterCommitOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStage(getOperation(), getStage());
}

LogicalResult PipeWriterCloseOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStagePhase(getOperation(), getStage(), getPhase());
}

LogicalResult PipeReaderWaitOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  if (failed(verifyPipeStagePhase(getOperation(), getStage(), getPhase())))
    return failure();
  if (!getIsClosed().getType().isInteger(1))
    return emitOpError("expects is_closed result to be i1");
  return success();
}

LogicalResult PipeReaderReleaseOp::verify() {
  if (failed(verifyPipeAttrs(getOperation(), getFields())))
    return failure();
  return verifyPipeStage(getOperation(), getStage());
}

void PipeReaderReleaseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
  MutableOperandRange fields = getFieldsMutable();
  for (unsigned i = 0, e = fields.size(); i < e; ++i)
    effects.emplace_back(MemoryEffects::Free::get(), &fields[i],
                         triton::gpu::SharedMemory::get());
}

static bool isValidPublicTaskGridName(StringRef name) {
  if (name.empty() || name == "commit" || name == "epoch" ||
      name == "fields" || name == "name" || name == "scope" ||
      name == "shape" || name == "tile_id" || name == "type" ||
      name.starts_with("_") || !isAsciiIdentStart(name.front()))
    return false;
  return llvm::all_of(name.drop_front(), isAsciiIdentChar);
}

static LogicalResult verifyTaskGridNameArray(Operation *op,
                                             ArrayAttr namesAttr,
                                             StringRef attrName,
                                             bool allowEmpty) {
  if (!allowEmpty && namesAttr.empty())
    return op->emitOpError("expects ")
           << attrName << " to contain at least one name";

  llvm::SmallSet<StringRef, 8> seenNames;
  for (Attribute attr : namesAttr) {
    auto nameAttr = dyn_cast<StringAttr>(attr);
    if (!nameAttr)
      return op->emitOpError("expects ")
             << attrName << " to contain only strings";
    StringRef name = nameAttr.getValue();
    if (!isValidPublicTaskGridName(name))
      return op->emitOpError("expects valid public task_grid ")
             << attrName << " names";
    if (!seenNames.insert(name).second)
      return op->emitOpError("expects unique task_grid ")
             << attrName << " names";
  }

  return success();
}

static LogicalResult verifyTaskGridAttrs(Operation *op, OperandRange fields) {
  auto scopeAttr = op->getAttrOfType<StringAttr>("scope");
  if (!scopeAttr)
    return op->emitOpError("requires scope attribute");
  if (scopeAttr.getValue() != "device" && scopeAttr.getValue() != "cta")
    return op->emitOpError("supports only scope = \"device\" or \"cta\"");

  auto gridNameAttr = op->getAttrOfType<StringAttr>("grid_name");
  if (!gridNameAttr)
    return op->emitOpError("requires grid_name attribute");
  if (!isValidPublicTaskGridName(gridNameAttr.getValue()))
    return op->emitOpError("expects valid public task_grid grid_name");

  auto fieldNamesAttr = op->getAttrOfType<ArrayAttr>("field_names");
  if (!fieldNamesAttr)
    return op->emitOpError("requires field_names attribute");
  if (fieldNamesAttr.size() != fields.size())
    return op->emitOpError("expects field_names size to match field operands");
  if (failed(
          verifyTaskGridNameArray(op, fieldNamesAttr, "field", false)))
    return failure();

  if (fields.empty())
    return op->emitOpError("expects at least one task_grid field");

  if (auto shapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("shape")) {
    if (shapeAttr.empty())
      return op->emitOpError("expects non-empty task_grid shape");
    for (int64_t dim : shapeAttr.asArrayRef()) {
      if (dim <= 0)
        return op->emitOpError("expects positive task_grid shape dimensions");
    }
  }

  return success();
}

LogicalResult TaskGridCreateOp::verify() {
  return verifyTaskGridAttrs(getOperation(), getFields());
}

LogicalResult TaskGridTileIdOp::verify() {
  if (failed(verifyTaskGridAttrs(getOperation(), getFields())))
    return failure();
  auto shapeAttr = getOperation()->getAttrOfType<DenseI64ArrayAttr>("shape");
  if (shapeAttr && getIds().size() != shapeAttr.size())
    return emitOpError("expects tile_id result count to match shape rank");
  if (!shapeAttr && !getIds().empty())
    return emitOpError("requires shape when tile_id has results");
  return success();
}

LogicalResult TaskGridCommitOp::verify() {
  return verifyTaskGridAttrs(getOperation(), getFields());
}

static LogicalResult verifyTaskDomainShape(Operation *op,
                                           ArrayRef<int64_t> domainShape) {
  if (domainShape.empty())
    return op->emitOpError("expects non-empty task domain_shape");
  for (int64_t dim : domainShape) {
    if (dim <= 0)
      return op->emitOpError("expects positive task domain_shape dimensions");
  }
  return success();
}

static LogicalResult verifyProjectionTaskMap(Operation *op, AffineMap map,
                                             int64_t domainRank,
                                             StringRef mapKind) {
  if (static_cast<int64_t>(map.getNumDims()) != domainRank)
    return op->emitOpError()
           << "expects task " << mapKind
           << " map input rank to match domain rank";
  if (map.getNumSymbols() != 0)
    return op->emitOpError()
           << "expects task " << mapKind << " map to have no symbols";
  for (AffineExpr expr : map.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map results to be dimension projections";
    if (static_cast<int64_t>(dim.getPosition()) >= domainRank)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map result dimensions to reference the task domain";
  }
  return success();
}

static LogicalResult verifyTaskMapEntryKeys(Operation *op, DictionaryAttr dict,
                                            StringRef mapKind) {
  for (NamedAttribute named : dict) {
    StringRef key = named.getName().getValue();
    if (key != "grid" && key != "map" && key != "wildcard_dims")
      return op->emitOpError()
             << "expects task " << mapKind << " map entry key " << key
             << " to be one of grid, map, or wildcard_dims";
  }
  return success();
}

static LogicalResult verifyWildcardDims(Operation *op,
                                        DenseI64ArrayAttr wildcardDims,
                                        StringRef mapKind) {
  llvm::SmallSet<int64_t, 8> seenDims;
  for (int64_t dim : wildcardDims.asArrayRef()) {
    if (dim < 0)
      return op->emitOpError()
             << "expects task " << mapKind
             << " wildcard_dims to be non-negative";
    if (!seenDims.insert(dim).second)
      return op->emitOpError()
             << "expects task " << mapKind
             << " wildcard_dims to be unique";
  }
  return success();
}

static LogicalResult verifyTaskMapEntries(Operation *op, ArrayAttr entries,
                                          int64_t domainRank,
                                          StringRef mapKind,
                                          bool allowWildcards) {
  for (Attribute entryAttr : entries) {
    auto entry = dyn_cast<DictionaryAttr>(entryAttr);
    if (!entry)
      return op->emitOpError()
             << "expects task " << mapKind
             << " maps to contain only dictionaries";
    if (failed(verifyTaskMapEntryKeys(op, entry, mapKind)))
      return failure();

    auto gridAttr = dyn_cast_or_null<StringAttr>(entry.get("grid"));
    if (!gridAttr)
      return op->emitOpError()
             << "expects task " << mapKind << " map entry to contain grid";
    if (!isValidPublicTaskGridName(gridAttr.getValue()))
      return op->emitOpError()
             << "expects valid public task_grid name in task " << mapKind
             << " map";

    auto mapAttr = dyn_cast_or_null<AffineMapAttr>(entry.get("map"));
    if (!mapAttr)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map entry to contain affine map";
    AffineMap map = mapAttr.getValue();
    if (failed(verifyProjectionTaskMap(op, map, domainRank, mapKind)))
      return failure();

    int64_t wildcardCount = 0;
    if (Attribute wildcardAttr = entry.get("wildcard_dims")) {
      if (!allowWildcards)
        return op->emitOpError()
               << "task writes must not use wildcard_dims";
      auto wildcardDims = dyn_cast<DenseI64ArrayAttr>(wildcardAttr);
      if (!wildcardDims)
        return op->emitOpError()
               << "expects task " << mapKind
               << " wildcard_dims to be an i64 dense array";
      wildcardCount = wildcardDims.size();
      if (failed(verifyWildcardDims(op, wildcardDims, mapKind)))
        return failure();
    }

    if (map.getNumResults() + wildcardCount == 0)
      return op->emitOpError()
             << "expects task " << mapKind
             << " map entry to target at least one grid dimension";
  }

  return success();
}

LogicalResult TaskDeclareOp::verify() {
  if (!isValidPublicTaskGridName(getTaskName()))
    return emitOpError("expects valid public task name");
  ArrayRef<int64_t> domainShape = getDomainShape();
  if (failed(verifyTaskDomainShape(getOperation(), domainShape)))
    return failure();
  int64_t domainRank = domainShape.size();

  if (failed(verifyTaskMapEntries(getOperation(), getReads(), domainRank,
                                  "read", /*allowWildcards=*/true)))
    return failure();
  if (getWrites().empty())
    return emitOpError("expects task writes to contain at least one map");
  if (failed(verifyTaskMapEntries(getOperation(), getWrites(), domainRank,
                                  "write", /*allowWildcards=*/false)))
    return failure();
  return success();
}

static FailureOr<int64_t> getSchedulerI64Attr(Operation *op,
                                              StringRef attrName,
                                              bool requirePositive) {
  auto attr = op->getAttrOfType<IntegerAttr>(attrName);
  if (!attr)
    return op->emitOpError("requires ") << attrName << " attribute";
  int64_t value = attr.getInt();
  if (requirePositive && value <= 0)
    return op->emitOpError("expects positive ") << attrName;
  if (!requirePositive && value < 0)
    return op->emitOpError("expects non-negative ") << attrName;
  return value;
}

static FailureOr<DenseI32ArrayAttr>
getSchedulerI32ArrayAttr(Operation *op, StringRef attrName) {
  auto attr = op->getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!attr)
    return op->emitOpError("requires ") << attrName << " attribute";
  return attr;
}

static LogicalResult verifyStringArray(Operation *op, StringRef attrName,
                                       bool allowEmpty) {
  auto array = op->getAttrOfType<ArrayAttr>(attrName);
  if (!array)
    return op->emitOpError("requires ") << attrName << " attribute";
  if (!allowEmpty && array.empty())
    return op->emitOpError("expects ") << attrName << " to be non-empty";
  for (Attribute attr : array) {
    auto value = dyn_cast<StringAttr>(attr);
    if (!value || value.getValue().empty())
      return op->emitOpError("expects ")
             << attrName << " to contain non-empty strings";
  }
  return success();
}

static LogicalResult verifySchedulerDictionaryKeys(Operation *op,
                                                   DictionaryAttr dict,
                                                   ArrayRef<StringRef> keys,
                                                   ArrayRef<StringRef> optionalKeys,
                                                   StringRef attrName) {
  llvm::SmallSet<StringRef, 8> required;
  for (StringRef key : keys)
    required.insert(key);
  llvm::SmallSet<StringRef, 8> optional;
  for (StringRef key : optionalKeys)
    optional.insert(key);
  llvm::SmallSet<StringRef, 8> seen;
  for (NamedAttribute named : dict) {
    StringRef key = named.getName().getValue();
    if (!required.contains(key) && !optional.contains(key))
      return op->emitOpError()
             << "expects " << attrName << " entry key " << key
             << " to be part of the scheduler schema";
    seen.insert(key);
  }
  for (StringRef key : keys) {
    if (!seen.contains(key))
      return op->emitOpError()
             << "expects " << attrName << " entry to contain " << key;
  }
  return success();
}

static LogicalResult verifySchedulerStringList(Operation *op, DictionaryAttr dict,
                                               StringRef dictName,
                                               StringRef key) {
  auto array = dyn_cast_or_null<ArrayAttr>(dict.get(key));
  if (!array)
    return op->emitOpError()
           << "expects " << dictName << " entry " << key
           << " to be a string array";
  for (Attribute attr : array) {
    auto stringAttr = dyn_cast<StringAttr>(attr);
    if (!stringAttr || stringAttr.getValue().empty())
      return op->emitOpError()
             << "expects " << dictName << " entry " << key
             << " to contain non-empty strings";
  }
  return success();
}

LogicalResult TaskGraphSchedulerOp::verify() {
  Operation *op = getOperation();

  FailureOr<int64_t> numTasks =
      getSchedulerI64Attr(op, "num_tasks", /*requirePositive=*/true);
  FailureOr<int64_t> numInstances =
      getSchedulerI64Attr(op, "num_instances", /*requirePositive=*/true);
  FailureOr<int64_t> numEdges =
      getSchedulerI64Attr(op, "num_edges", /*requirePositive=*/false);
  FailureOr<int64_t> queueCapacity =
      getSchedulerI64Attr(op, "queue_capacity", /*requirePositive=*/true);
  if (failed(numTasks) || failed(numInstances) || failed(numEdges) ||
      failed(queueCapacity))
    return failure();

  auto counterType = op->getAttrOfType<StringAttr>("counter_type");
  if (!counterType)
    return emitOpError("requires counter_type attribute");
  if (counterType.getValue() != "i32")
    return emitOpError("MVP supports only counter_type = \"i32\"");

  if (failed(verifyStringArray(op, "task_names", /*allowEmpty=*/false)) ||
      failed(verifyStringArray(op, "initial_ready", /*allowEmpty=*/false)))
    return failure();

  auto taskNames = op->getAttrOfType<ArrayAttr>("task_names");
  auto dispatch = op->getAttrOfType<ArrayAttr>("dispatch");
  auto instances = op->getAttrOfType<ArrayAttr>("instances");
  auto edges = op->getAttrOfType<ArrayAttr>("edges");
  auto initialReady = op->getAttrOfType<ArrayAttr>("initial_ready");
  FailureOr<DenseI32ArrayAttr> taskDomainRanks =
      getSchedulerI32ArrayAttr(op, "task_domain_ranks");
  FailureOr<DenseI32ArrayAttr> instanceTaskIds =
      getSchedulerI32ArrayAttr(op, "instance_task_ids");
  FailureOr<DenseI32ArrayAttr> instanceDepCounts =
      getSchedulerI32ArrayAttr(op, "instance_dep_counts");
  FailureOr<DenseI32ArrayAttr> instanceCoordOffsets =
      getSchedulerI32ArrayAttr(op, "instance_coord_offsets");
  FailureOr<DenseI32ArrayAttr> instanceCoords =
      getSchedulerI32ArrayAttr(op, "instance_coords");
  FailureOr<DenseI32ArrayAttr> initialReadyIds =
      getSchedulerI32ArrayAttr(op, "initial_ready_ids");
  FailureOr<DenseI32ArrayAttr> producerEdgeOffsets =
      getSchedulerI32ArrayAttr(op, "producer_edge_offsets");
  FailureOr<DenseI32ArrayAttr> edgeConsumerIds =
      getSchedulerI32ArrayAttr(op, "edge_consumer_ids");
  if (failed(taskDomainRanks) || failed(instanceTaskIds) ||
      failed(instanceDepCounts) || failed(instanceCoordOffsets) ||
      failed(instanceCoords) || failed(initialReadyIds) ||
      failed(producerEdgeOffsets) || failed(edgeConsumerIds))
    return failure();
  if (!dispatch)
    return emitOpError("requires dispatch attribute");
  if (!instances)
    return emitOpError("requires instances attribute");
  if (!edges)
    return emitOpError("requires edges attribute");

  if (*numTasks != static_cast<int64_t>(taskNames.size()))
    return emitOpError("expects num_tasks to match task_names size");
  if (*numTasks != static_cast<int64_t>(dispatch.size()))
    return emitOpError("expects num_tasks to match dispatch size");
  if (*numInstances != static_cast<int64_t>(instances.size()))
    return emitOpError("expects num_instances to match instances size");
  if (*numEdges != static_cast<int64_t>(edges.size()))
    return emitOpError("expects num_edges to match edges size");
  if (*queueCapacity < static_cast<int64_t>(initialReady.size()))
    return emitOpError("expects queue_capacity to cover initial_ready size");
  if (*queueCapacity < *numInstances)
    return emitOpError("expects queue_capacity to cover all task instances");
  if (static_cast<int64_t>((*taskDomainRanks).size()) != *numTasks)
    return emitOpError("expects task_domain_ranks size to match num_tasks");
  if (static_cast<int64_t>((*instanceTaskIds).size()) != *numInstances)
    return emitOpError("expects instance_task_ids size to match num_instances");
  if (static_cast<int64_t>((*instanceDepCounts).size()) != *numInstances)
    return emitOpError("expects instance_dep_counts size to match num_instances");
  if (static_cast<int64_t>((*instanceCoordOffsets).size()) !=
      *numInstances + 1)
    return emitOpError("expects instance_coord_offsets size to be "
                       "num_instances + 1");
  if (static_cast<int64_t>((*initialReadyIds).size()) !=
      static_cast<int64_t>(initialReady.size()))
    return emitOpError("expects initial_ready_ids size to match initial_ready");
  if (static_cast<int64_t>((*producerEdgeOffsets).size()) !=
      *numInstances + 1)
    return emitOpError("expects producer_edge_offsets size to be "
                       "num_instances + 1");
  if (static_cast<int64_t>((*edgeConsumerIds).size()) != *numEdges)
    return emitOpError("expects edge_consumer_ids size to match num_edges");

  ArrayRef<int32_t> taskRanks = (*taskDomainRanks).asArrayRef();
  ArrayRef<int32_t> taskIds = (*instanceTaskIds).asArrayRef();
  ArrayRef<int32_t> depCounts = (*instanceDepCounts).asArrayRef();
  ArrayRef<int32_t> coordOffsets = (*instanceCoordOffsets).asArrayRef();
  ArrayRef<int32_t> coords = (*instanceCoords).asArrayRef();
  ArrayRef<int32_t> readyIds = (*initialReadyIds).asArrayRef();
  ArrayRef<int32_t> edgeOffsets = (*producerEdgeOffsets).asArrayRef();
  ArrayRef<int32_t> consumerIds = (*edgeConsumerIds).asArrayRef();

  for (int32_t rank : taskRanks) {
    if (rank <= 0)
      return emitOpError("expects positive task_domain_ranks entries");
  }
  if (coordOffsets.front() != 0)
    return emitOpError("expects instance_coord_offsets to start at zero");
  for (int64_t index = 0; index < *numInstances; ++index) {
    int32_t taskId = taskIds[index];
    if (taskId < 0 || taskId >= *numTasks)
      return emitOpError("expects instance_task_ids entries to be in "
                         "task_names range");
    if (depCounts[index] < 0 || depCounts[index] > *numEdges)
      return emitOpError("expects instance_dep_counts entries to be in edge "
                         "range");
    if (coordOffsets[index] > coordOffsets[index + 1])
      return emitOpError("expects instance_coord_offsets to be nondecreasing");
    if (coordOffsets[index + 1] >
        static_cast<int32_t>(coords.size()))
      return emitOpError("expects instance_coord_offsets to stay within "
                         "instance_coords");
    if (coordOffsets[index + 1] - coordOffsets[index] != taskRanks[taskId])
      return emitOpError("expects instance coordinate rank to match task "
                         "domain rank");
  }
  if (coordOffsets.back() != static_cast<int32_t>(coords.size()))
    return emitOpError("expects final instance_coord_offsets entry to match "
                       "instance_coords size");
  for (int32_t coord : coords) {
    if (coord < 0)
      return emitOpError("expects non-negative instance_coords entries");
  }
  for (int32_t readyId : readyIds) {
    if (readyId < 0 || readyId >= *numInstances)
      return emitOpError("expects initial_ready_ids entries to reference "
                         "instances");
    if (depCounts[readyId] != 0)
      return emitOpError("expects initial_ready_ids entries to have zero "
                         "dependency count");
  }
  if (edgeOffsets.front() != 0)
    return emitOpError("expects producer_edge_offsets to start at zero");
  for (int64_t index = 0; index < *numInstances; ++index) {
    if (edgeOffsets[index] > edgeOffsets[index + 1])
      return emitOpError("expects producer_edge_offsets to be nondecreasing");
    if (edgeOffsets[index + 1] > static_cast<int32_t>(consumerIds.size()))
      return emitOpError("expects producer_edge_offsets to stay within "
                         "edge_consumer_ids");
  }
  if (edgeOffsets.back() != static_cast<int32_t>(consumerIds.size()))
    return emitOpError("expects final producer_edge_offsets entry to match "
                       "edge_consumer_ids size");
  for (int32_t consumerId : consumerIds) {
    if (consumerId < 0 || consumerId >= *numInstances)
      return emitOpError("expects edge_consumer_ids entries to reference "
                         "instances");
  }

  llvm::SmallSet<StringRef, 32> taskNameSet;
  for (Attribute attr : taskNames) {
    StringRef taskName = cast<StringAttr>(attr).getValue();
    if (!isValidPublicTaskGridName(taskName))
      return emitOpError("expects valid public task names");
    if (!taskNameSet.insert(taskName).second)
      return emitOpError("expects unique task_names");
  }

  llvm::SmallSet<int64_t, 32> dispatchIds;
  for (Attribute attr : dispatch) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return emitOpError("expects dispatch to contain dictionaries");
    if (failed(verifySchedulerDictionaryKeys(
            op, dict, ArrayRef<StringRef>{"task", "task_id"},
            ArrayRef<StringRef>{"callee"}, "dispatch")))
      return failure();
    auto task = dyn_cast<StringAttr>(dict.get("task"));
    auto taskId = dyn_cast<IntegerAttr>(dict.get("task_id"));
    if (Attribute callee = dict.get("callee")) {
      if (!isa<FlatSymbolRefAttr>(callee))
        return emitOpError("expects dispatch callee to be a symbol reference");
    }
    if (!task || !taskNameSet.contains(task.getValue()))
      return emitOpError("expects dispatch task to reference task_names");
    if (!taskId || taskId.getInt() < 0 || taskId.getInt() >= *numTasks)
      return emitOpError("expects dispatch task_id to be in task_names range");
    if (!dispatchIds.insert(taskId.getInt()).second)
      return emitOpError("expects unique dispatch task_id values");
  }
  if (static_cast<int64_t>(dispatchIds.size()) != *numTasks)
    return emitOpError("expects dispatch task_id values to cover all tasks");

  std::set<std::string> instanceNames;
  for (Attribute attr : instances) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return emitOpError("expects instances to contain dictionaries");
    if (failed(verifySchedulerDictionaryKeys(
            op, dict,
            ArrayRef<StringRef>{"task", "instance", "coord", "dep_count", "deps",
                                "writes"},
            ArrayRef<StringRef>{},
            "instances")))
      return failure();
    auto task = dyn_cast<StringAttr>(dict.get("task"));
    auto instance = dyn_cast<StringAttr>(dict.get("instance"));
    auto depCount = dyn_cast<IntegerAttr>(dict.get("dep_count"));
    if (!task || !taskNameSet.contains(task.getValue()))
      return emitOpError("expects instance task to reference task_names");
    if (!instance || instance.getValue().empty())
      return emitOpError("expects non-empty instance name");
    if (!instanceNames.insert(instance.getValue().str()).second)
      return emitOpError("expects unique task instance names");
    if (!depCount || depCount.getInt() < 0 || depCount.getInt() > *numEdges)
      return emitOpError("expects instance dep_count to be in edge range");
    if (!isa<DenseI64ArrayAttr>(dict.get("coord")))
      return emitOpError("expects instance coord to be an i64 dense array");
    if (failed(verifySchedulerStringList(op, dict, "instances", "deps")) ||
        failed(verifySchedulerStringList(op, dict, "instances", "writes")))
      return failure();
  }

  for (Attribute attr : initialReady) {
    StringRef instance = cast<StringAttr>(attr).getValue();
    if (!instanceNames.count(instance.str()))
      return emitOpError("expects initial_ready entries to reference instances");
  }

  std::set<std::string> seenEdges;
  for (Attribute attr : edges) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return emitOpError("expects edges to contain dictionaries");
    if (failed(verifySchedulerDictionaryKeys(
            op, dict, ArrayRef<StringRef>{"tile", "producer", "consumer"},
            ArrayRef<StringRef>{},
            "edges")))
      return failure();
    auto tile = dyn_cast<StringAttr>(dict.get("tile"));
    auto producer = dyn_cast<StringAttr>(dict.get("producer"));
    auto consumer = dyn_cast<StringAttr>(dict.get("consumer"));
    if (!tile || tile.getValue().empty())
      return emitOpError("expects non-empty edge tile");
    if (!producer || !instanceNames.count(producer.getValue().str()))
      return emitOpError("expects edge producer to reference instances");
    if (!consumer || !instanceNames.count(consumer.getValue().str()))
      return emitOpError("expects edge consumer to reference instances");
    std::string key =
        (tile.getValue() + "|" + producer.getValue() + "->" +
         consumer.getValue())
            .str();
    if (!seenEdges.insert(key).second)
      return emitOpError("expects unique producer/consumer scheduler edges");
  }

  return success();
}

LogicalResult TaskGraphRuntimeStateOp::verify() {
  Operation *op = getOperation();

  FailureOr<int64_t> numInstances =
      getSchedulerI64Attr(op, "num_instances", /*requirePositive=*/true);
  FailureOr<int64_t> queueCapacity =
      getSchedulerI64Attr(op, "queue_capacity", /*requirePositive=*/true);
  FailureOr<int64_t> stateSize =
      getSchedulerI64Attr(op, "state_size_bytes", /*requirePositive=*/true);
  FailureOr<int64_t> alignment =
      getSchedulerI64Attr(op, "alignment_bytes", /*requirePositive=*/true);
  FailureOr<int64_t> counterBytes =
      getSchedulerI64Attr(op, "counter_bytes", /*requirePositive=*/true);
  FailureOr<int64_t> queueElementBytes =
      getSchedulerI64Attr(op, "queue_element_bytes",
                          /*requirePositive=*/true);
  FailureOr<int64_t> initFlagOffset =
      getSchedulerI64Attr(op, "init_flag_offset_bytes",
                          /*requirePositive=*/false);
  FailureOr<int64_t> queueLockOffset =
      getSchedulerI64Attr(op, "queue_lock_offset_bytes",
                          /*requirePositive=*/false);
  FailureOr<int64_t> queueHeadOffset =
      getSchedulerI64Attr(op, "queue_head_offset_bytes",
                          /*requirePositive=*/false);
  FailureOr<int64_t> queueTailOffset =
      getSchedulerI64Attr(op, "queue_tail_offset_bytes",
                          /*requirePositive=*/false);
  FailureOr<int64_t> completedCountOffset =
      getSchedulerI64Attr(op, "completed_count_offset_bytes",
                          /*requirePositive=*/false);
  FailureOr<int64_t> depCountersOffset =
      getSchedulerI64Attr(op, "dep_counters_offset_bytes",
                          /*requirePositive=*/false);
  FailureOr<int64_t> queueStorageOffset =
      getSchedulerI64Attr(op, "queue_storage_offset_bytes",
                          /*requirePositive=*/false);
  if (failed(numInstances) || failed(queueCapacity) || failed(stateSize) ||
      failed(alignment) || failed(counterBytes) ||
      failed(queueElementBytes) || failed(initFlagOffset) ||
      failed(queueLockOffset) || failed(queueHeadOffset) ||
      failed(queueTailOffset) || failed(completedCountOffset) ||
      failed(depCountersOffset) || failed(queueStorageOffset))
    return failure();

  auto counterType = op->getAttrOfType<StringAttr>("counter_type");
  if (!counterType)
    return emitOpError("requires counter_type attribute");
  if (counterType.getValue() != "i32")
    return emitOpError("MVP supports only counter_type = \"i32\"");
  if (*counterBytes != 4 || *queueElementBytes != 4)
    return emitOpError("MVP runtime state uses 4-byte counters and queue "
                       "elements");
  if (*queueCapacity < *numInstances)
    return emitOpError("expects queue_capacity to cover all task instances");
  if ((*alignment & (*alignment - 1)) != 0)
    return emitOpError("expects alignment_bytes to be a power of two");
  if (*alignment < 4)
    return emitOpError("expects alignment_bytes to be at least 4");

  auto checkAligned = [&](int64_t offset, StringRef name) -> LogicalResult {
    if (offset % *counterBytes != 0)
      return emitOpError("expects ") << name << " to be 4-byte aligned";
    return success();
  };
  if (failed(checkAligned(*initFlagOffset, "init_flag_offset_bytes")) ||
      failed(checkAligned(*queueLockOffset, "queue_lock_offset_bytes")) ||
      failed(checkAligned(*queueHeadOffset, "queue_head_offset_bytes")) ||
      failed(checkAligned(*queueTailOffset, "queue_tail_offset_bytes")) ||
      failed(checkAligned(*completedCountOffset,
                          "completed_count_offset_bytes")) ||
      failed(checkAligned(*depCountersOffset,
                          "dep_counters_offset_bytes")) ||
      failed(checkAligned(*queueStorageOffset,
                          "queue_storage_offset_bytes")))
    return failure();

  constexpr int64_t kHeaderWords = 5;
  constexpr int64_t kI32Bytes = 4;
  int64_t max = std::numeric_limits<int64_t>::max();
  if (*numInstances > (max / kI32Bytes) ||
      *queueCapacity > (max / kI32Bytes))
    return emitOpError("runtime state byte size overflows i64");
  int64_t depBytes = *numInstances * kI32Bytes;
  int64_t queueBytes = *queueCapacity * kI32Bytes;
  if (depBytes > max - kHeaderWords * kI32Bytes ||
      queueBytes > max - (kHeaderWords * kI32Bytes + depBytes))
    return emitOpError("runtime state byte size overflows i64");

  int64_t expectedInitFlag = 0;
  int64_t expectedQueueLock = expectedInitFlag + kI32Bytes;
  int64_t expectedQueueHead = expectedQueueLock + kI32Bytes;
  int64_t expectedQueueTail = expectedQueueHead + kI32Bytes;
  int64_t expectedCompletedCount = expectedQueueTail + kI32Bytes;
  int64_t expectedDepCounters = expectedCompletedCount + kI32Bytes;
  int64_t expectedQueueStorage = expectedDepCounters + depBytes;
  int64_t expectedStateSize = expectedQueueStorage + queueBytes;

  if (*initFlagOffset != expectedInitFlag)
    return emitOpError("expects init_flag_offset_bytes to be zero");
  if (*queueLockOffset != expectedQueueLock)
    return emitOpError("expects queue_lock_offset_bytes to follow init flag");
  if (*queueHeadOffset != expectedQueueHead)
    return emitOpError("expects queue_head_offset_bytes to follow queue lock");
  if (*queueTailOffset != expectedQueueTail)
    return emitOpError("expects queue_tail_offset_bytes to follow queue head");
  if (*completedCountOffset != expectedCompletedCount)
    return emitOpError("expects completed_count_offset_bytes to follow queue "
                       "tail");
  if (*depCountersOffset != expectedDepCounters)
    return emitOpError("expects dep_counters_offset_bytes to follow scheduler "
                       "header");
  if (*queueStorageOffset != expectedQueueStorage)
    return emitOpError("expects queue_storage_offset_bytes to follow "
                       "dependency counters");
  if (*stateSize != expectedStateSize)
    return emitOpError("expects state_size_bytes to cover dependency counters "
                       "and queue storage exactly");

  return success();
}

// ============================================================================
// InsertTileOp Type Inference + Verification
// ============================================================================
LogicalResult InsertTileOp::inferReturnTypes(
    [[maybe_unused]] MLIRContext *context,
    [[maybe_unused]] std::optional<Location> location, ValueRange operands,
    [[maybe_unused]] DictionaryAttr attributes,
    [[maybe_unused]] OpaqueProperties properties,
    [[maybe_unused]] RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // insert_tile(src, tile, index) -> result has the same type as src.
  if (operands.size() < 3)
    return failure();

  auto srcTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto tileTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!srcTy || !tileTy)
    return failure();

  // Keep conservative checks here; full diagnostics are handled in verify().
  if (srcTy.getElementType() != tileTy.getElementType() ||
      srcTy.getRank() != tileTy.getRank())
    return failure();

  inferredReturnTypes.clear();
  inferredReturnTypes.push_back(srcTy);
  return success();
}

// ============================================================================
// InsertTileOp Verification
//
// For dynamic index (index operand is not arith.constant):
//   - Only check constraints that are known at compile time: tile_shape
//   positivity, divisibility, element type, rank/result shape match
//   - Skip out-of-bounds and insertion region boundary checks (only known at
//   runtime)
//
// For static index: perform full checks (same as original implementation)
// ============================================================================
LogicalResult InsertTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto tileTy = cast<RankedTensorType>(getTile().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());

  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();
  auto dstShape = dstTy.getShape();

  // --- Basic checks required for both static and dynamic index ---

  // Check 1: element types must match
  if (srcTy.getElementType() != tileTy.getElementType())
    return emitOpError("tile element type must match source element type");
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != tileTy.getRank())
    return emitOpError("tile rank must equal source rank");
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("result rank must equal source rank");

  // Check 3: result shape must equal source shape
  if (dstShape != srcShape)
    return emitOpError("result shape must equal source shape");

  // Check 4: tile_shape must be positive in each dimension and divide source
  // shape
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // Check 5: insert_tile updates values but does not change global layout,
  // result encoding must match source encoding
  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (srcEnc && dstEnc && srcEnc != dstEnc)
    return emitOpError("result encoding must match source encoding");

  // --- Determine if index is a static constant ---
  // insert_tile index is the 3rd operand: (src, tile, index).
  auto idxDef =
      getOperation()->getOperand(2).getDefiningOp<arith::ConstantOp>();
  if (!idxDef) {
    // Dynamic index: skip out-of-bounds and insertion region boundary checks,
    // handled at lowering stage
    return success();
  }

  // --- Full checks for static index ---
  int64_t index = mlir::cast<mlir::IntegerAttr>(idxDef.getValue()).getInt();
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  // Delinearize to per-dimension tile indices (row-major order)
  SmallVector<int64_t> tileIndices(srcShape.size(), 0);
  int64_t remain = index;
  for (int i = static_cast<int>(srcShape.size()) - 1; i >= 0; --i) {
    tileIndices[i] = remain % logicalGridShape[i];
    remain /= logicalGridShape[i];
  }

  // tile indices -> coordinate-level offsets
  SmallVector<int64_t> offsets(srcShape.size(), 0);
  for (size_t i = 0; i < srcShape.size(); ++i)
    offsets[i] = tileIndices[i] * tileShape[i];

  // Boundary check: the full insertion region must be within the source
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (offsets[i] < 0)
      return emitOpError("offset must be non-negative at dimension ") << i;
    else if (offsets[i] + tileShape[i] > srcShape[i])
      return emitOpError("invalid insertion region at dimension ")
             << i << ": offset(" << offsets[i] << ") + tile(" << tileShape[i]
             << ") > source(" << srcShape[i] << ")";
  }

  return success();
}

LogicalResult DSLRegionOp::verify() {
  Region &body = getBody();
  const uint32_t numArguments = body.getNumArguments(),
                 numOperands = getNumOperands();
  if (numArguments != numOperands) {
    return emitOpError() << "expects number of operands (" << numArguments
                         << ") to match number of region arguments ("
                         << numOperands << ")";
  }
  for (auto [arg, operand] : llvm::zip(body.getArguments(), getOperands())) {
    if (arg.getType() != operand.getType()) {
      return emitOpError() << "expects region argument type (" << arg.getType()
                           << ") to match operand type (" << operand.getType()
                           << ")";
    }
  }
  return success();
}

void ExtractSizesOp::build(::mlir::OpBuilder &odsBuilder,
                           ::mlir::OperationState &odsState, size_t num,
                           Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

void ExtractStridesOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState, size_t num,
                             Value tensor) {
  SmallVector<Type> tys(num, odsBuilder.getI64Type());
  build(odsBuilder, odsState, tys, tensor);
}

LogicalResult PackOp::verify() {
  TypedValue<LLVM::LLVMStructType> input = getInput();
  ArrayRef<Type> body = input.getType().getBody();
  if (body.size() < 3 || body.size() % 2 != 1 ||
      !isa<LLVM::LLVMPointerType>(body[0]) ||
      !isa<LLVM::LLVMPointerType>(body[1])) {
    return emitOpError() << "expects input struct to have at least 3 elements, "
                            "with the first two being pointer types.";
  }
  return success();
}

LogicalResult LocalPointersOp::verify() {
  auto memDescTy = dyn_cast<triton::gpu::MemDescType>(getSrc().getType());
  if (!memDescTy)
    return emitOpError() << "expects src operand to be a ttg.memdesc";

  auto resultTensorTy = dyn_cast<RankedTensorType>(getResult().getType());
  auto resultPtrTy = dyn_cast<triton::PointerType>(getResult().getType());
  if (!resultTensorTy && !resultPtrTy)
    return emitOpError()
           << "expects result to be either tensor<tt.ptr<...>> or tt.ptr";

  auto ptrTy =
      resultTensorTy
          ? dyn_cast<triton::PointerType>(resultTensorTy.getElementType())
          : resultPtrTy;
  if (!ptrTy)
    return emitOpError() << "expects result element type to be tt.ptr";

  if (ptrTy.getPointeeType() != memDescTy.getElementType())
    return emitOpError() << "expects pointer pointee type "
                         << ptrTy.getPointeeType()
                         << " to match memdesc element type "
                         << memDescTy.getElementType();

  if (ptrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError() << "expects pointers to live in shared memory";

  auto indices = getIndices();
  if (indices.empty()) {
    if (resultTensorTy) {
      if (resultTensorTy.getShape() != memDescTy.getShape())
        return emitOpError()
               << "zero-index local_pointers expects tensor result shape to "
                  "match buffer shape";
      return success();
    }
    if (!memDescTy.getShape().empty())
      return emitOpError()
             << "zero-index scalar local_pointers is only valid for rank-0 "
                "buffers";
    return success();
  }

  if (indices.size() != memDescTy.getShape().size())
    return emitOpError() << "expects indices count to match buffer rank";

  if (resultTensorTy) {
    auto resultShape = resultTensorTy.getShape();
    Attribute resultEncoding = resultTensorTy.getEncoding();

    ArrayRef<int64_t> indexShape;
    for (Value val : indices) {
      auto indexTy = dyn_cast<RankedTensorType>(val.getType());
      if (!indexTy)
        return emitOpError()
               << "tensor result expects indices to be ranked tensors";
      if (!indexTy.getElementType().isInteger())
        return emitOpError() << "expects indices return tensors to have "
                                "integer element types";
      if (indexShape.empty())
        indexShape = indexTy.getShape();
      else if (indexTy.getShape() != indexShape)
        return emitOpError()
               << "expects indices return tensors to have identical shapes";
      if (resultEncoding && indexTy.getEncoding() &&
          resultEncoding != indexTy.getEncoding())
        return emitOpError()
               << "expects indices return tensors to match result encoding";
    }

    if (indexShape != resultShape)
      return emitOpError()
             << "expects indices return tensor shape to match result shape";
    return success();
  }

  for (Value val : indices) {
    if (auto indexTy = dyn_cast<IntegerType>(val.getType())) {
      if (!indexTy.isSignlessInteger())
        return emitOpError()
               << "expects scalar indices to be signless integers";
      continue;
    }
    return emitOpError() << "scalar result expects scalar integer indices";
  }

  return success();
}

LogicalResult ExclusiveCumsumOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  if (!srcTy)
    return emitOpError() << "expects src to be a ranked tensor";

  auto exclusiveTy = dyn_cast<RankedTensorType>(getExclusive().getType());
  if (!exclusiveTy)
    return emitOpError() << "expects exclusive result to be a ranked tensor";
  if (exclusiveTy != srcTy)
    return emitOpError() << "expects exclusive result type to match src type";

  // Keep semantics aligned with current DeepSeek topk use: scan over a single
  // per-block histogram vector.
  if (srcTy.getRank() != 1)
    return emitOpError() << "currently only rank-1 tensors are supported";
  int64_t axisExtent = srcTy.getShape()[0];
  if (ShapedType::isDynamic(axisExtent) || axisExtent <= 0)
    return emitOpError() << "currently only static, positive axis extent is "
                            "supported";
  if (axisExtent > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return emitOpError() << "axis extent is too large";

  const int64_t rank = srcTy.getRank();
  int64_t axis = static_cast<int64_t>(getAxis());
  if (axis < 0)
    axis += rank;
  if (axis != 0)
    return emitOpError() << "currently only axis=0 is supported";

  if (getTotal().getType() != srcTy.getElementType())
    return emitOpError() << "expects total result type to match src element "
                            "type";

  return success();
}

LogicalResult DistributedBarrierOp::verify() {
  auto *op = getOperation();
  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto rankAttr = op->getAttrOfType<IntegerAttr>("group_rank");
  auto shapeAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_shape");
  auto axesAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
  auto maskAttr = op->getAttrOfType<DenseI32ArrayAttr>("group_mask");

  const bool hasAnyGroupMeta =
      rankAttr || shapeAttr || axesAttr || maskAttr || kindAttr;
  if (!hasAnyGroupMeta)
    return success();

  if (!kindAttr) {
    return emitOpError()
           << "group_kind is required when distributed barrier group metadata "
              "is provided";
  }

  StringRef kind = kindAttr.getValue();
  if (kind != "cluster" && kind != "submesh" && kind != "grid") {
    return emitOpError()
           << "group_kind must be 'cluster', 'submesh', or 'grid', got '"
           << kind << "'";
  }

  if (kind == "cluster" || kind == "grid") {
    if (rankAttr || shapeAttr || axesAttr || maskAttr) {
      return emitOpError()
             << kind
             << " group_kind does not accept "
                "group_rank/group_shape/group_axes/group_mask attrs";
    }
    return success();
  }

  if (!rankAttr || !shapeAttr || !axesAttr) {
    return emitOpError()
           << "submesh group_kind requires group_rank/group_shape/group_axes";
  }
  if (!rankAttr.getType().isInteger(32)) {
    return emitOpError() << "group_rank must be i32";
  }

  int32_t rank = static_cast<int32_t>(rankAttr.getInt());
  if (rank <= 0) {
    return emitOpError() << "group_rank must be > 0";
  }
  if (static_cast<int32_t>(shapeAttr.size()) != rank) {
    return emitOpError() << "group_shape length (" << shapeAttr.size()
                         << ") must match group_rank (" << rank << ")";
  }
  if (static_cast<int32_t>(axesAttr.size()) != rank) {
    return emitOpError() << "group_axes length (" << axesAttr.size()
                         << ") must match group_rank (" << rank << ")";
  }

  llvm::SmallSet<int32_t, 8> seenAxes;
  for (int32_t dim : shapeAttr.asArrayRef()) {
    if (dim <= 0)
      return emitOpError() << "group_shape entries must be > 0";
  }
  for (int32_t axis : axesAttr.asArrayRef()) {
    if (axis < 0)
      return emitOpError() << "group_axes entries must be >= 0";
    if (!seenAxes.insert(axis).second) {
      return emitOpError() << "group_axes entries must be unique";
    }
  }
  if (maskAttr) {
    if (maskAttr.asArrayRef().empty())
      return emitOpError() << "group_mask cannot be empty";
    for (int32_t id : maskAttr.asArrayRef()) {
      if (id < 0)
        return emitOpError() << "group_mask entries must be >= 0";
    }
  }

  return success();
}

LogicalResult RemotePointersOp::verify() {
  Type srcTy = getSrc().getType();
  Type resultTy = getResult().getType();
  auto getPtrInfo = [&](Type ty, triton::PointerType &ptr, bool &isTensor,
                        ArrayRef<int64_t> &shape,
                        Attribute &encoding) -> LogicalResult {
    if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
      ptr = dyn_cast<triton::PointerType>(tensorTy.getElementType());
      if (!ptr)
        return emitOpError()
               << "expects tensor src/result element type to be tt.ptr";
      isTensor = true;
      shape = tensorTy.getShape();
      encoding = tensorTy.getEncoding();
      return success();
    }
    if (auto ptrTy = dyn_cast<triton::PointerType>(ty)) {
      ptr = ptrTy;
      isTensor = false;
      shape = ArrayRef<int64_t>();
      encoding = Attribute();
      return success();
    }
    return emitOpError() << "expects src/result to be tensor<tt.ptr<...>> or "
                            "tt.ptr";
  };

  triton::PointerType srcPtrTy;
  triton::PointerType resultPtrTy;
  bool srcIsTensor = false;
  bool resultIsTensor = false;
  ArrayRef<int64_t> srcShape;
  ArrayRef<int64_t> resultShape;
  Attribute srcEncoding;
  Attribute resultEncoding;
  if (failed(getPtrInfo(srcTy, srcPtrTy, srcIsTensor, srcShape, srcEncoding)) ||
      failed(getPtrInfo(resultTy, resultPtrTy, resultIsTensor, resultShape,
                        resultEncoding)))
    return failure();

  if (srcIsTensor != resultIsTensor)
    return emitOpError() << "expects src/result to both be scalar pointers or "
                            "both be pointer tensors";
  if (srcIsTensor) {
    if (srcShape != resultShape)
      return emitOpError() << "expects src/result pointer tensor shapes to "
                              "match";
    if (srcEncoding && resultEncoding && srcEncoding != resultEncoding)
      return emitOpError() << "expects src/result pointer tensor encodings to "
                              "match";
  }
  if (srcPtrTy.getPointeeType() != resultPtrTy.getPointeeType())
    return emitOpError() << "expects src/result pointer pointee types to "
                            "match";

  if (srcPtrTy.getAddressSpace() != kSharedMemoryAddressSpace)
    return emitOpError()
           << "expects src pointers to live in shared memory (addrspace=3)";
  if (resultPtrTy.getAddressSpace() != kClusterSharedMemoryAddressSpace)
    return emitOpError()
           << "expects result pointers to live in cluster shared memory "
              "(addrspace=7)";

  if (!getShardId().getType().isInteger(32))
    return emitOpError() << "expects shard_id to be i32";

  return success();
}

} // namespace mlir::triton::tle
