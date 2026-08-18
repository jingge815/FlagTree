//===----------------------------------------------------------------------===//
//
// Turns the implicit global memory accesses inherited from TTIR into the
// explicit two-level data movement a near-memory device requires:
//
//   tt.load  -> wram_alloc + dma_load  + barrier + wram_load
//   tt.store -> wram_alloc + wram_store + barrier + dma_store
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMEXPLICITDMA
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

//===----------------------------------------------------------------------===//
// Pointer analysis
//===----------------------------------------------------------------------===//
//
// What a DMA engine needs is a base address plus a stride; what TTIR gives us is
// a tensor of independent pointers, built up by broadcasting and adding index
// arithmetic. This walks that construction backwards far enough to answer one
// question: does the address advance by a constant number of elements per step
// along some dimension?
//
// It is deliberately narrow. It recognizes the shape Triton front-ends actually
// emit for a strided tile -- splat of a base, plus offsets from make_range,
// expand_dims and broadcast -- and gives up on anything else. Giving up is a
// supported outcome: the caller records nothing, and the missing attributes mark
// the transfer as not yet proven DMA-able.

struct AddressPattern {
  // Dimension along which addresses are contiguous (or strided).
  int64_t contiguousDim = -1;
  // Element step along that dimension.
  int64_t elemStride = 0;
  // Index of the tt.func argument the base traces back to, if it does.
  std::optional<int64_t> baseArg;

  bool isProven() const { return contiguousDim >= 0; }
};

// Constant integer behind a value, looking through splat.
static std::optional<int64_t> matchConstantInt(Value v) {
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return cst.getSExtValue();
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    return matchConstantInt(splat.getSrc());
  if (auto cstOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cstOp.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

// Traces a scalar base pointer back to a function argument, looking through the
// pointer bumps a loop induction adds.
static std::optional<int64_t> traceBaseArg(Value v) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      // Only a tt.func's own entry block gives a meaningful argument index; a
      // loop-carried argument does not.
      if (isa_and_nonnull<triton::FuncOp>(arg.getOwner()->getParentOp()))
        return arg.getArgNumber();
      return std::nullopt;
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
      v = addptr.getPtr();
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      v = splat.getSrc();
      continue;
    }
    if (auto bitcast = dyn_cast<triton::BitcastOp>(def)) {
      v = bitcast.getSrc();
      continue;
    }
    return std::nullopt;
  }
}

// Per-dimension element stride of an offset tensor, or nullopt if not a shape
// this analysis understands. Returns a vector with one entry per dimension;
// entry i is the step taken when index i advances by one, and 0 means the
// offset does not vary along i.
static std::optional<SmallVector<int64_t>> analyzeOffsets(Value offsets,
                                                          int64_t rank);

// Offsets built by adding two sub-expressions: strides add per dimension, which
// is how a row offset and a column offset compose into a 2-D tile.
static std::optional<SmallVector<int64_t>>
analyzeAdd(Value lhs, Value rhs, int64_t rank) {
  auto l = analyzeOffsets(lhs, rank);
  if (!l)
    return std::nullopt;
  auto r = analyzeOffsets(rhs, rank);
  if (!r)
    return std::nullopt;
  SmallVector<int64_t> res(rank, 0);
  for (int64_t d = 0; d < rank; ++d) {
    // Two varying contributions to the same dimension would need a more general
    // representation than a single stride, so bail rather than guess.
    if ((*l)[d] != 0 && (*r)[d] != 0)
      return std::nullopt;
    res[d] = (*l)[d] != 0 ? (*l)[d] : (*r)[d];
  }
  return res;
}

static std::optional<SmallVector<int64_t>> analyzeOffsets(Value offsets,
                                                          int64_t rank) {
  // A uniform value contributes no variation along any dimension.
  if (matchConstantInt(offsets))
    return SmallVector<int64_t>(rank, 0);

  Operation *def = offsets.getDefiningOp();
  if (!def)
    return std::nullopt;

  // A relayout moves elements between tasklets but does not change which
  // address each element has, so addresses are unaffected. These appear
  // routinely: the TTIR->PIM type converter inserts them wherever a producer's
  // layout does not match its consumer's.
  if (auto cvt = dyn_cast<ConvertLayoutOp>(def))
    return analyzeOffsets(cvt.getSrc(), rank);

  // tt.make_range: unit step along its (only) dimension.
  if (isa<triton::MakeRangeOp>(def)) {
    auto ty = dyn_cast<RankedTensorType>(offsets.getType());
    if (!ty || ty.getRank() != 1)
      return std::nullopt;
    SmallVector<int64_t> res(rank, 0);
    // A rank-1 range inside a rank-N context varies along the last dimension.
    res[rank - 1] = 1;
    return res;
  }

  // tt.splat of a scalar: uniform.
  if (isa<triton::SplatOp>(def))
    return SmallVector<int64_t>(rank, 0);

  // tt.expand_dims: inserts an extent-1 dimension, shifting the strides of the
  // dimensions after it.
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
    auto srcTy = dyn_cast<RankedTensorType>(expand.getSrc().getType());
    if (!srcTy)
      return std::nullopt;
    auto inner = analyzeOffsets(expand.getSrc(), srcTy.getRank());
    if (!inner)
      return std::nullopt;
    int64_t axis = expand.getAxis();
    SmallVector<int64_t> res;
    res.reserve(rank);
    // Rebuild in the expanded coordinate space.
    for (int64_t d = 0, s = 0; d < rank; ++d) {
      if (d == axis) {
        res.push_back(0); // the new dimension has extent 1
      } else {
        res.push_back(s < (int64_t)inner->size() ? (*inner)[s] : 0);
        ++s;
      }
    }
    return res;
  }

  // tt.broadcast: replicates along the dimensions it widens, so the strides it
  // already had are preserved and the widened ones stay zero.
  if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
    auto srcTy = dyn_cast<RankedTensorType>(bcast.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(bcast.getType());
    if (!srcTy || !dstTy || srcTy.getRank() != dstTy.getRank())
      return std::nullopt;
    auto inner = analyzeOffsets(bcast.getSrc(), srcTy.getRank());
    if (!inner)
      return std::nullopt;
    SmallVector<int64_t> res(rank, 0);
    for (int64_t d = 0; d < srcTy.getRank() && d < rank; ++d) {
      // A dimension being broadcast (extent 1 -> N) contributes no variation.
      if (srcTy.getDimSize(d) == dstTy.getDimSize(d))
        res[d] = (*inner)[d];
    }
    return res;
  }

  // arith.addi: strides add.
  if (auto add = dyn_cast<arith::AddIOp>(def))
    return analyzeAdd(add.getLhs(), add.getRhs(), rank);

  // arith.muli by a uniform constant: strides scale.
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    Value varying = mul.getLhs();
    std::optional<int64_t> factor = matchConstantInt(mul.getRhs());
    if (!factor) {
      factor = matchConstantInt(mul.getLhs());
      varying = mul.getRhs();
    }
    if (!factor)
      return std::nullopt;
    auto inner = analyzeOffsets(varying, rank);
    if (!inner)
      return std::nullopt;
    SmallVector<int64_t> res(rank, 0);
    for (int64_t d = 0; d < rank; ++d)
      res[d] = (*inner)[d] * *factor;
    return res;
  }

  return std::nullopt;
}

// Entry point: what, if anything, can we prove about this pointer tensor?
static AddressPattern analyzePointers(Value ptrs) {
  AddressPattern pat;
  auto ptrTy = dyn_cast<RankedTensorType>(ptrs.getType());
  if (!ptrTy)
    return pat;
  int64_t rank = ptrTy.getRank();

  auto addptr = ptrs.getDefiningOp<triton::AddPtrOp>();
  if (!addptr)
    return pat;

  pat.baseArg = traceBaseArg(addptr.getPtr());

  auto strides = analyzeOffsets(addptr.getOffset(), rank);
  if (!strides)
    return pat;

  // The fastest-changing dimension with unit stride is the one a DMA can walk
  // contiguously; prefer it, and otherwise report the innermost dimension that
  // varies at all.
  for (int64_t d = rank - 1; d >= 0; --d) {
    if ((*strides)[d] == 1) {
      pat.contiguousDim = d;
      pat.elemStride = 1;
      return pat;
    }
  }
  for (int64_t d = rank - 1; d >= 0; --d) {
    if ((*strides)[d] > 0) {
      pat.contiguousDim = d;
      pat.elemStride = (*strides)[d];
      return pat;
    }
  }
  return pat;
}

// Records what the analysis proved, and nothing more.
static void annotate(Operation *op, const AddressPattern &pat) {
  if (!pat.isProven())
    return;
  Builder b(op->getContext());
  op->setAttr("contiguous_dim", b.getI64IntegerAttr(pat.contiguousDim));
  op->setAttr("elem_stride", b.getI64IntegerAttr(pat.elemStride));
  if (pat.baseArg)
    op->setAttr("base_arg", b.getI64IntegerAttr(*pat.baseArg));
}

//===----------------------------------------------------------------------===//
// Buffer allocation
//===----------------------------------------------------------------------===//

// Where a staging buffer for `op` should be allocated: outside every enclosing
// loop, so that one buffer is reserved and reused rather than one per iteration.
// WRAM is far too small to afford the latter.
static OpBuilder::InsertPoint getAllocInsertPoint(Operation *op) {
  Operation *outermost = nullptr;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<triton::FuncOp>(parent))
      break;
    if (isa<LoopLikeOpInterface>(parent))
      outermost = parent;
  }
  if (outermost)
    return OpBuilder::InsertPoint(outermost->getBlock(),
                                  Block::iterator(outermost));
  return OpBuilder::InsertPoint(op->getBlock(), Block::iterator(op));
}

// Allocates a WRAM buffer shaped like `tensorTy`, hoisted out of loops.
static Value createHoistedAlloc(OpBuilder &builder, Operation *anchor,
                                RankedTensorType tensorTy) {
  auto memdescTy = MemDescType::get(tensorTy.getShape(),
                                    tensorTy.getElementType(),
                                    WRAMSpaceAttr::get(builder.getContext()));
  OpBuilder::InsertionGuard guard(builder);
  builder.restoreInsertionPoint(getAllocInsertPoint(anchor));
  return builder.create<WRAMAllocOp>(anchor->getLoc(), memdescTy).getResult();
}

//===----------------------------------------------------------------------===//
// Rewrites
//===----------------------------------------------------------------------===//

static LogicalResult rewriteLoad(triton::LoadOp op) {
  auto resultTy = dyn_cast<RankedTensorType>(op.getType());
  // Scalar loads address a single word; there is no tile to stage.
  if (!resultTy)
    return success();
  auto ptrTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
  if (!ptrTy)
    return success();

  OpBuilder builder(op);
  Value buf = createHoistedAlloc(builder, op, resultTy);

  auto dma = builder.create<DmaLoadOp>(op.getLoc(), op.getPtr(), op.getMask(),
                                       op.getOther(), buf);
  annotate(dma, analyzePointers(op.getPtr()));

  // The DMA is asynchronous with respect to the tasklets; they may not read the
  // buffer until every transfer has landed.
  builder.create<BarrierOp>(op.getLoc());

  auto loaded = builder.create<WRAMLoadOp>(op.getLoc(), resultTy, buf);
  op.getResult().replaceAllUsesWith(loaded.getResult());
  op.erase();
  return success();
}

static LogicalResult rewriteStore(triton::StoreOp op) {
  auto valueTy = dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!valueTy)
    return success();
  auto ptrTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
  if (!ptrTy)
    return success();

  OpBuilder builder(op);
  Value buf = createHoistedAlloc(builder, op, valueTy);

  builder.create<WRAMStoreOp>(op.getLoc(), op.getValue(), buf);
  // Every tasklet's contribution must be in WRAM before the buffer leaves.
  builder.create<BarrierOp>(op.getLoc());

  auto dma = builder.create<DmaStoreOp>(op.getLoc(), buf, op.getPtr(),
                                        op.getMask());
  annotate(dma, analyzePointers(op.getPtr()));

  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct TritonPIMExplicitDMAPass
    : public mlir::triton::pim::impl::TritonPIMExplicitDMABase<
          TritonPIMExplicitDMAPass> {
  using mlir::triton::pim::impl::TritonPIMExplicitDMABase<
      TritonPIMExplicitDMAPass>::TritonPIMExplicitDMABase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect first: the rewrites insert and erase ops, so walking live IR
    // would be unsound.
    SmallVector<triton::LoadOp> loads;
    SmallVector<triton::StoreOp> stores;
    mod.walk([&](Operation *op) {
      if (auto load = dyn_cast<triton::LoadOp>(op))
        loads.push_back(load);
      else if (auto store = dyn_cast<triton::StoreOp>(op))
        stores.push_back(store);
    });

    for (auto load : loads)
      if (failed(rewriteLoad(load)))
        return signalPassFailure();
    for (auto store : stores)
      if (failed(rewriteStore(store)))
        return signalPassFailure();

    // Report the WRAM footprint. Note this counts staging buffers only: values
    // that stay live in registers across the kernel, such as a dot
    // accumulator, are still SSA values at this level and get their storage
    // assigned when the IR is lowered to memrefs.
    int64_t used = 0;
    bool exact = true;
    mod.walk([&](WRAMAllocOp alloc) {
      if (auto bytes = alloc.getType().getSizeInBytes())
        used += *bytes;
      else
        exact = false;
    });

    Builder b(&getContext());
    mod->setAttr(AttrWramBytesUsedName,
                 b.getI32IntegerAttr(static_cast<int32_t>(used)));

    if (auto budget = maybeLookupWramBytes(mod)) {
      if (used > *budget)
        mod.emitWarning() << "WRAM staging buffers total " << used
                          << " bytes, over the " << *budget
                          << " byte budget; the tiles need to be split";
      else if (!exact)
        mod.emitWarning() << "some WRAM allocations have no statically known "
                             "size; the reported total of "
                          << used << " bytes is a lower bound";
    }
  }
};

} // namespace
