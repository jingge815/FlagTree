//===----------------------------------------------------------------------===//
//
// Bridges the flagos-pim-compiler graph compiler to a numpy-executable
// artifact. Lowers a single-DPU `pim-explicit-dma` output, for any
// `pim.num-tasklets >= 1`, into `emitc.*` ops that `-convert-func-to-emitc`
// and `mlir-translate --mlir-to-cpp` turn into plain, ctypes-callable C.
//
// See TritonPIMLowerToEmitC in Passes.td for the rationale. The short
// version: the M dimension of a `tt.dot` is split `num-tasklets` ways (a
// static split, unrolled at pass-compile time -- M and num-tasklets are both
// compile-time constants, there is no runtime `tid` value). `w` is
// snapshotted once, shared read-only across all blocks; each block snapshots
// only its own row range of `a` before writing that same row range of `out`
// -- self-contained per block, which is what keeps the existing
// `out`-aliases-`a` safety property (see snapshotToLocal's comment) intact
// when M is split. The blocks execute strictly in program order --
// sequential because that is the concurrency model flagos-pim-compiler's
// NumPy backend uses (deterministic in-order tasklet simulation, not real
// threads, so results stay bit-reproducible and a hazard checker on the
// NumPy side can still catch a wrong split). `num-tasklets == 1` is simply
// the degenerate case of a single block covering the whole M range -- there
// is no separate single-tasklet path. `tt.dot` is expanded here because
// nothing else in Triton or FlagTree lowers it below the tensor level.
//
//===----------------------------------------------------------------------===//
//
// # Offset analysis
//
// `pim-explicit-dma` records only `contiguous_dim` / `elem_stride` /
// `base_arg` on each DMA -- enough to prove "this is a strided transfer",
// which is all a DMA engine needs. It deliberately does not record the full
// address map, and its own comments leave that to "a later, stronger
// analysis". This pass needs that stronger analysis, because it emits real
// subscripts: it needs each dimension's stride, and it needs to know which
// enclosing loop induction variables shift the tile's base.
//
// `OffsetAnalysis` reduces a DMA's pointer-offset tensor to
//
//     offset(i, j) = constant + rowCoeff*i + colCoeff*j + sum(coeff_v * v)
//
// over enclosing loop induction variables `v`, recognizing exactly the shapes
// Triton front-ends emit for a tile: `tt.make_range`, `tt.expand_dims`,
// `tt.broadcast`, `tt.splat` (of a constant or an induction variable),
// `arith.muli` by a splat constant, `arith.addi`. It gives up (returns
// nullopt, and the pass then errors) on anything else rather than guessing an
// address map and silently computing garbage.
//
// # Tiling
//
// A kernel at llama2-7b scale cannot materialize a whole operand tile as one
// Triton tensor, and cannot fit one in GPU shared memory either:
//
//   - Triton caps a tensor at 2^20 elements, and a 4096x512 weight shard is 2x
//     that, so the contraction dimension must be tiled (`BLOCK_K`).
//   - `tl.dot`'s operand tiles live in shared memory, and o_proj's N=4096
//     exceeds the per-CTA limit even at BLOCK_K=16, so the output dimension
//     must be tiled too (`BLOCK_N`).
//
// So the front-end kernel emits a loop nest -- an outer loop over N tiles,
// an inner loop over K tiles carrying the accumulator:
//
//     for n0 in range(0, N, BLOCK_N):
//       acc = 0
//       for k0 in range(0, K, BLOCK_K):     <- scf.for, tensor iter_arg
//         acc = tt.dot(x[:, k0:...], w[n0:..., k0:...].T, acc)
//       out[:, n0:...] = acc
//
// Both loops exist only to bound the front-end's own tile footprints (Triton
// tensor-size and shared-memory limits); they say nothing about how the DPU
// should split work across tasklets. So rather than mirroring the nest, this
// pass first collapses it back to the full, untiled M/K/N (recovering the
// bounds the front-end had to give up), and only then applies the
// tasklet-level M-split described above.
//
// The rule that makes the collapse general: a loop induction variable appears
// in the address with some coefficient; when that coefficient equals a tile
// dimension's stride, the loop and that dimension index the same axis, and
// together they cover the axis' full extent. Merging them multiplies the tile
// extent by the loop's trip count, recovering the full M/K/N. This works for
// any number of tiled dimensions in any nesting order, so the flat (untiled),
// K-tiled, and N+K-tiled front-end kernels all lower to identical C -- which
// the flagos-pim-compiler side relies on, since it compiles one artifact per
// shape and expects the numerics to be shape-determined.
//
// # Dead address arithmetic
//
// `pim-explicit-dma` rewrites `tt.load`/`tt.store` but leaves the ops that
// *computed* their pointers in place, since the DMA ops it emits still take
// that tensor-of-pointers as an operand. This pass reads the DMA attributes
// and runs its own offset analysis instead, so those ops are dead here and
// are skipped. `tt.trans` / `pim.convert_layout` are the one wrinkle: they
// appear both on that dead chain (over address integers) and legitimately on
// a data tensor between `wram_load` and `tt.dot`. They are told apart by
// whether the operand is a tracked buffer view.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMLOWERTOEMITC
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

//===----------------------------------------------------------------------===//
// Element types
//===----------------------------------------------------------------------===//
//
// Storage may be f16 or f32; arithmetic is always f32. That mirrors
// runtime/kernels.py's linear_kernel, which reads/writes MRAM at the graph's
// dtype (llama2-7b is f16 throughout) but computes in float32 -- and it is
// what `tt.dot` already expresses, taking f16 operands and yielding an f32
// accumulator, with an `arith.truncf` narrowing the result before the store.
//
// C has no portable half type: gcc 11 on this target rejects both `_Float16`
// and `__fp16` (verified). So f16 storage is emitted as `uint16_t` and
// converted with the bit-twiddling helpers below, which were checked against
// numpy over all 65536 f16 bit patterns plus 300k f32 inputs covering
// subnormals, overflow, inf and nan, with round-to-nearest-even.

// Emitted verbatim ahead of the kernel when any operand is f16.
static constexpr const char *kF16Helpers = R"C(
static float pim_f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) { bits = sign; }
    else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f; __builtin_memcpy(&f, &bits, 4); return f;
}
static uint16_t pim_f32_to_f16(float f) {
  uint32_t bits; __builtin_memcpy(&bits, &f, 4);
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
  int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = bits & 0x7FFFFFu;
  if (((bits >> 23) & 0xFFu) == 0xFFu)
    return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t midp = 1u << (shift - 1);
    if (rem > midp || (rem == midp && (half & 1u))) half++;
    return (uint16_t)(sign | half);
  }
  uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
  return half;
}
)C";

// Storage element type -> the C type the emitted pointer parameter uses.
// f16 becomes uint16_t because C has no portable half (see above).
static Type storageTypeFor(OpBuilder &b, Type elemTy) {
  if (elemTy.isF16())
    return b.getI16Type();
  return elemTy; // f32
}

static LogicalResult checkElementType(Operation *op, Type elemTy) {
  if (elemTy.isF32() || elemTy.isF16())
    return success();
  return op->emitError() << "pim-lower-to-emitc only handles f16/f32 "
                            "elements, got "
                         << elemTy;
}

//===----------------------------------------------------------------------===//
// Enclosing tiled loops
//===----------------------------------------------------------------------===//

struct LoopInfo {
  Value iv;
  int64_t tripCount;
};

//===----------------------------------------------------------------------===//
// Affine offset analysis
//===----------------------------------------------------------------------===//

// Which of the tile's two axes a term varies along. A term that is still a
// bare 1-D range (before the enclosing `expand_dims` assigns it an axis) is
// `Unassigned`; `expand_dims` turns it into Row or Col.
enum class Axis { Unassigned, Row, Col };

// One enclosing loop's contribution to an address: which axis it slides, and
// by how much per unit of the induction variable.
struct IVTerm {
  Value iv;
  int64_t coeff = 0;
  Axis axis = Axis::Unassigned;
};

struct AffineOffset {
  int64_t constant = 0;
  int64_t rowCoeff = 0;
  int64_t colCoeff = 0;
  // Induction-variable terms, in the order encountered. Kept as a small
  // vector rather than a map: there are at most a couple of tiled dimensions.
  SmallVector<IVTerm, 4> ivTerms;

  void addIV(Value iv, int64_t coeff, Axis axis = Axis::Unassigned) {
    for (auto &t : ivTerms)
      if (t.iv == iv) {
        t.coeff += coeff;
        if (t.axis == Axis::Unassigned)
          t.axis = axis;
        return;
      }
    ivTerms.push_back({iv, coeff, axis});
  }

  AffineOffset operator+(const AffineOffset &o) const {
    AffineOffset r = *this;
    r.constant += o.constant;
    r.rowCoeff += o.rowCoeff;
    r.colCoeff += o.colCoeff;
    for (auto &t : o.ivTerms)
      r.addIV(t.iv, t.coeff, t.axis);
    return r;
  }

  AffineOffset scaled(int64_t f) const {
    AffineOffset r;
    r.constant = constant * f;
    r.rowCoeff = rowCoeff * f;
    r.colCoeff = colCoeff * f;
    for (auto &t : ivTerms)
      r.addIV(t.iv, t.coeff * f, t.axis);
    return r;
  }

  // `expand_dims` assigns axes: it relabels the incoming 1-D variation as this
  // tile's row or column. Applies to the plain coefficients and to every
  // induction-variable term alike.
  void assignAxes(bool toRow) {
    if (toRow)
      std::swap(rowCoeff, colCoeff);
    for (auto &t : ivTerms)
      if (t.axis == Axis::Unassigned)
        t.axis = toRow ? Axis::Row : Axis::Col;
  }
};

struct OffsetAnalysis {
  // Induction variables of all enclosing tiled loops.
  ArrayRef<LoopInfo> loops;

  static std::optional<int64_t> constantInt(Value v) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto splat = dyn_cast<SplatElementsAttr>(cst.getValue()))
        return splat.getSplatValue<APInt>().getSExtValue();
      if (auto i = dyn_cast<IntegerAttr>(cst.getValue()))
        return i.getInt();
    }
    return std::nullopt;
  }

  bool isLoopIV(Value v) const {
    for (auto &l : loops)
      if (l.iv == v)
        return true;
    return false;
  }

  std::optional<AffineOffset> analyze(Value v) const {
    if (auto c = constantInt(v)) {
      AffineOffset r;
      r.constant = *c;
      return r;
    }

    Operation *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;

    // A range contributes unit stride along its single dimension; the
    // enclosing expand_dims decides whether that is the row or the column.
    if (auto range = dyn_cast<triton::MakeRangeOp>(def)) {
      if (range.getStart() != 0)
        return std::nullopt;
      AffineOffset r;
      r.colCoeff = 1;
      return r;
    }

    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      Value src = splat.getSrc();
      if (isLoopIV(src)) {
        AffineOffset r;
        r.addIV(src, 1);
        return r;
      }
      if (auto c = constantInt(src)) {
        AffineOffset r;
        r.constant = *c;
        return r;
      }
      return std::nullopt;
    }

    if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
      auto inner = analyze(expand.getSrc());
      if (!inner)
        return std::nullopt;
      AffineOffset r = *inner;
      // axis 0 -> 1xN, variation along the column; axis 1 -> Nx1, along the row.
      r.assignAxes(/*toRow=*/expand.getAxis() == 1);
      return r;
    }

    // Broadcast replicates along a size-1 dimension: coefficients unchanged.
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def))
      return analyze(bcast.getSrc());

    // A relayout does not change which address an element has.
    if (auto cvt = dyn_cast<ConvertLayoutOp>(def))
      return analyze(cvt.getSrc());

    if (auto add = dyn_cast<arith::AddIOp>(def)) {
      auto l = analyze(add.getLhs());
      if (!l)
        return std::nullopt;
      auto r = analyze(add.getRhs());
      if (!r)
        return std::nullopt;
      return *l + *r;
    }

    if (auto mul = dyn_cast<arith::MulIOp>(def)) {
      // A product of two varying terms is not affine; one side must be a
      // splat constant.
      if (auto c = constantInt(mul.getRhs())) {
        auto l = analyze(mul.getLhs());
        return l ? std::optional<AffineOffset>(l->scaled(*c)) : std::nullopt;
      }
      if (auto c = constantInt(mul.getLhs())) {
        auto r = analyze(mul.getRhs());
        return r ? std::optional<AffineOffset>(r->scaled(*c)) : std::nullopt;
      }
      return std::nullopt;
    }

    return std::nullopt;
  }

  // A DMA op's pointer operand is `tt.addptr(splat(base_arg), offsets)`.
  std::optional<AffineOffset> analyzePtr(Value ptrs) const {
    auto addptr = ptrs.getDefiningOp<triton::AddPtrOp>();
    if (!addptr)
      return std::nullopt;
    return analyze(addptr.getOffset());
  }
};

//===----------------------------------------------------------------------===//
// Buffer views
//===----------------------------------------------------------------------===//

// A tracked tensor SSA value: a 2-D tile of `rows x cols` elements in the
// MRAM buffer reached through `ptr`, addressed as
// `rowStride*row + colStride*col + constant`. `rows`/`cols` are the *full*
// extents once any enclosing tiled loops have been merged in (see "Tiling" in
// the file header). `transposed` flips the two logical indices without moving
// data, which is what `tt.trans` means here.
struct BufferView {
  Value ptr;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t rowStride = 0;
  int64_t colStride = 1;
  int64_t constant = 0;
  bool transposed = false;
  // MRAM storage element type (f16 or f32). Arithmetic is always f32; f16
  // buffers are read/written through the conversion helpers.
  Type elemTy;

  bool isF16() const { return elemTy && elemTy.isF16(); }

  int64_t logicalRows() const { return transposed ? cols : rows; }
  int64_t logicalCols() const { return transposed ? rows : cols; }

  Value elementOffset(OpBuilder &b, Location loc, Value row, Value col) const {
    auto i32 = b.getI32Type();
    auto cst = [&](int64_t v) -> Value {
      return b.create<emitc::ConstantOp>(loc, i32, b.getI32IntegerAttr(v));
    };
    Value physRow = transposed ? col : row;
    Value physCol = transposed ? row : col;
    Value off = b.create<emitc::MulOp>(loc, i32, physRow, cst(rowStride));
    if (colStride != 1)
      physCol = b.create<emitc::MulOp>(loc, i32, physCol, cst(colStride));
    off = b.create<emitc::AddOp>(loc, i32, off, physCol);
    if (constant != 0)
      off = b.create<emitc::AddOp>(loc, i32, off, cst(constant));
    return off;
  }
};

// Builds a full-extent view from a tile's shape plus its analyzed offset, by
// merging each enclosing loop into whichever dimension shares its stride.
static FailureOr<BufferView>
makeView(Operation *op, Value ptr, int64_t tileRows, int64_t tileCols,
         Type elemTy, const AffineOffset &off, ArrayRef<LoopInfo> loops) {
  BufferView v;
  v.ptr = ptr;
  v.rows = tileRows;
  v.cols = tileCols;
  v.rowStride = off.rowCoeff;
  v.colStride = off.colCoeff;
  v.constant = off.constant;
  v.elemTy = elemTy;

  for (auto &term : off.ivTerms) {
    // Find this induction variable's trip count.
    int64_t trip = 0;
    for (auto &l : loops)
      if (l.iv == term.iv)
        trip = l.tripCount;
    if (trip == 0)
      return op->emitError() << "pim-lower-to-emitc: address depends on "
                                "a value that is not an enclosing tiled loop's "
                                "induction variable";
    // Which axis the loop slides comes from the IR (the `expand_dims` that
    // placed it), not from matching its coefficient against the strides: those
    // can coincide. o_proj is exactly that case -- w is (N=4096, K=512), so
    // w's row stride is 512, and the N-tiling loop with BLOCK_N=512 also
    // contributes 512, since its induction variable is scaled by K on the way
    // into the row position. Guessing by coefficient swapped N and K there and
    // made every o_proj launch wrong (caught by comparing against the NumPy
    // kernel launch-by-launch in the real decode loop).
    switch (term.axis) {
    case Axis::Row:
      v.rows *= trip;
      break;
    case Axis::Col:
      v.cols *= trip;
      break;
    case Axis::Unassigned:
      return op->emitError()
            << "pim-lower-to-emitc: a tiled loop's induction variable "
               "reaches the address without passing through an expand_dims, "
               "so which axis it slides is unknown";
    }
  }
  return v;
}

struct LoweringState {
  llvm::DenseMap<Value, BufferView> tensorToBuffer;

  const BufferView *tryLookup(Value v) const {
    auto it = tensorToBuffer.find(v);
    return it == tensorToBuffer.end() ? nullptr : &it->second;
  }
  void record(Value v, BufferView buf) { tensorToBuffer[v] = buf; }
};

// `base_arg` indexes the *original* tt.func argument list; the rewritten
// func.func keeps the same arity and order, so the index carries over.
static FailureOr<Value> resolveBaseArg(Operation *dmaOp, IntegerAttr baseArg,
                                       func::FuncOp newFunc) {
  if (!baseArg)
    return dmaOp->emitError()
          << "pim-lower-to-emitc requires a proven DMA address pattern "
             "(missing base_arg); pim-explicit-dma could not express this "
             "transfer as a strided DMA";
  int64_t idx = baseArg.getInt();
  if (idx < 0 || idx >= (int64_t)newFunc.getNumArguments())
    return dmaOp->emitError() << "base_arg " << idx << " out of range";
  return newFunc.getArgument(idx);
}

static FailureOr<std::pair<int64_t, int64_t>> get2DShape(Operation *op,
                                                         ArrayRef<int64_t> s) {
  if (s.size() != 2 || ShapedType::isDynamicShape(s))
    return op->emitError() << "pim-lower-to-emitc requires statically "
                              "shaped 2-D buffers, got shape "
                           << s;
  return std::make_pair(s[0], s[1]);
}

static std::optional<int64_t> tripCountOf(scf::ForOp forOp) {
  auto lb = OffsetAnalysis::constantInt(forOp.getLowerBound());
  auto ub = OffsetAnalysis::constantInt(forOp.getUpperBound());
  auto st = OffsetAnalysis::constantInt(forOp.getStep());
  if (!lb || !ub || !st || *st <= 0 || *lb != 0)
    return std::nullopt;
  if (*ub % *st != 0)
    return std::nullopt; // a partial tile would need masking
  return *ub / *st;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct TritonPIMLowerToEmitCPass
    : public mlir::triton::pim::impl::TritonPIMLowerToEmitCBase<
          TritonPIMLowerToEmitCPass> {
  using mlir::triton::pim::impl::TritonPIMLowerToEmitCBase<
      TritonPIMLowerToEmitCPass>::TritonPIMLowerToEmitCBase;

  // Set when any function argument is f16, so the conversion helpers get
  // emitted once ahead of the kernel.
  bool needsF16Helpers = false;

  // Degree to which `tt.dot`'s M dimension is split across tasklets. Missing
  // attribute defaults to 1 (single block, the degenerate case -- no separate
  // single-tasklet path). Read once in runOnOperation and used by
  // emitDotLoops for every `tt.dot` in the module.
  int64_t numTasklets = 1;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    if (auto nt = mod->getAttrOfType<IntegerAttr>(AttrNumTaskletsName)) {
      if (nt.getInt() < 1) {
        mod.emitError() << "pim-lower-to-emitc requires " << AttrNumTaskletsName
                        << " >= 1, got " << nt.getInt();
        return signalPassFailure();
      }
      numTasklets = nt.getInt();
    }

    SmallVector<triton::FuncOp> funcs;
    mod.walk([&](triton::FuncOp f) { funcs.push_back(f); });
    for (auto f : funcs)
      if (failed(lowerFunc(f)))
        return signalPassFailure();
  }

  LogicalResult lowerFunc(triton::FuncOp oldFunc) {
    MLIRContext *ctx = &getContext();
    OpBuilder b(ctx);
    Location loc = oldFunc.getLoc();

    SmallVector<Type> newArgTypes;
    for (Type argTy : oldFunc.getArgumentTypes()) {
      auto ptrTy = dyn_cast<triton::PointerType>(argTy);
      if (!ptrTy)
        return oldFunc.emitError()
              << "pim-lower-to-emitc only handles !tt.ptr<...> function "
                 "arguments, got "
              << argTy;
      if (failed(checkElementType(oldFunc, ptrTy.getPointeeType())))
        return failure();
      newArgTypes.push_back(emitc::PointerType::get(
          storageTypeFor(b, ptrTy.getPointeeType())));
      if (ptrTy.getPointeeType().isF16())
        needsF16Helpers = true;
    }

    b.setInsertionPoint(oldFunc);
    // f16 storage needs the conversion helpers ahead of the kernel; C has no
    // portable half type (see "Element types" at the top of this file).
    if (needsF16Helpers)
      b.create<emitc::VerbatimOp>(loc, b.getStringAttr(kF16Helpers));
    auto newFunc = b.create<func::FuncOp>(
        loc, oldFunc.getName(), FunctionType::get(ctx, newArgTypes, {}));
    Block *entry = newFunc.addEntryBlock();
    b.setInsertionPointToStart(entry);

    LoweringState state;
    SmallVector<LoopInfo, 4> loops;
    if (failed(lowerRegion(oldFunc.getBody().front(), newFunc, state, b, loops)))
      return failure();

    b.setInsertionPointToEnd(entry);
    b.create<func::ReturnOp>(loc);
    oldFunc.erase();
    return success();
  }

  // Walks a block in order. `scf.for` nests are descended into with their
  // induction variables recorded, so the offset analysis can recognize them;
  // nothing is emitted for the loops themselves (see "Tiling" in the header).
  LogicalResult lowerRegion(Block &block, func::FuncOp newFunc,
                            LoweringState &state, OpBuilder &b,
                            SmallVectorImpl<LoopInfo> &loops) {
    for (Operation &op : block) {
      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        auto trip = tripCountOf(forOp);
        if (!trip)
          return forOp.emitError()
                << "pim-lower-to-emitc only handles tiling loops with "
                   "static bounds starting at 0 and an evenly dividing step "
                   "(a partial tile would need masking, which pim-explicit-dma "
                   "cannot prove as a strided DMA)";
        loops.push_back({forOp.getInductionVar(), *trip});
        if (failed(lowerRegion(*forOp.getBody(), newFunc, state, b, loops)))
          return failure();
        loops.pop_back();

        // A loop carrying an accumulator forwards it: the dot expansion has
        // already written the full reduction into the output buffer, so the
        // loop's result is that buffer.
        if (forOp.getNumResults() == 1) {
          auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          if (auto *v = state.tryLookup(yield.getOperand(0)))
            state.record(forOp.getResult(0), *v);
        }
        continue;
      }
      if (isa<scf::YieldOp>(&op))
        continue;
      if (failed(lowerOp(&op, newFunc, state, b, loops)))
        return failure();
    }
    return success();
  }

  LogicalResult lowerOp(Operation *op, func::FuncOp newFunc,
                        LoweringState &state, OpBuilder &b,
                        ArrayRef<LoopInfo> loops) {
    Location loc = op->getLoc();

    // pim-explicit-dma's own WRAM staging (sized for the whole, untiled
    // operand -- it runs before this pass's M-split) is elided here: this
    // pass does its own staging in emitDotLoops/snapshotToLocal, split per
    // tasklet block rather than once for the whole operand, so the original
    // wram_alloc/barrier ops carry no numeric effect on top of that.
    if (isa<WRAMAllocOp>(op) || isa<BarrierOp>(op))
      return success();

    if (auto dmaLoad = dyn_cast<DmaLoadOp>(op)) {
      auto base = resolveBaseArg(dmaLoad, dmaLoad.getBaseArgAttr(), newFunc);
      if (failed(base))
        return failure();
      auto shape =
          get2DShape(dmaLoad, dmaLoad.getResult().getType().getShape());
      if (failed(shape))
        return failure();
      OffsetAnalysis analysis{loops};
      auto off = analysis.analyzePtr(dmaLoad.getPtr());
      if (!off)
        return dmaLoad.emitError()
              << "pim-lower-to-emitc could not reduce this transfer's "
                 "address computation to an affine form (see the offset "
                 "analysis in LowerPIMToEmitC.cpp)";
      auto view = makeView(dmaLoad, *base, shape->first, shape->second,
                           dmaLoad.getResult().getType().getElementType(), *off,
                           loops);
      if (failed(view))
        return failure();
      state.record(dmaLoad.getResult(), *view);
      return success();
    }

    if (auto wramLoad = dyn_cast<WRAMLoadOp>(op)) {
      const BufferView *buf = nullptr;
      for (auto &use : wramLoad.getSrc().getUses())
        if (auto dma = dyn_cast<DmaLoadOp>(use.getOwner()))
          if (auto *v = state.tryLookup(dma.getResult())) {
            buf = v;
            break;
          }
      if (!buf)
        return wramLoad.emitError() << "pim-lower-to-emitc could not "
                                       "trace wram_load back to a dma_load";
      state.record(wramLoad.getResult(), *buf);
      return success();
    }

    // Real only when the operand is a tracked data buffer; otherwise part of
    // the dead address chain.
    if (auto trans = dyn_cast<triton::TransOp>(op)) {
      const BufferView *src = state.tryLookup(trans.getSrc());
      if (!src)
        return success();
      ArrayRef<int32_t> order = trans.getOrder();
      if (order.size() != 2 || order[0] != 1 || order[1] != 0)
        return trans.emitError() << "pim-lower-to-emitc only handles a "
                                    "2-D transpose (order = [1, 0])";
      BufferView dst = *src;
      dst.transposed = !dst.transposed;
      state.record(trans.getResult(), dst);
      return success();
    }
    if (auto cvt = dyn_cast<ConvertLayoutOp>(op)) {
      if (const BufferView *src = state.tryLookup(cvt.getSrc()))
        state.record(cvt.getResult(), *src);
      return success();
    }

    // Precision conversions between the f32 accumulator and f16 storage
    // (`arith.truncf` before a store, `arith.extf` after a load) need no code
    // of their own: the emitted C converts at each element access, driven by
    // the buffer's storage type. Forward the view, and let the *destination*
    // buffer's own type decide the conversion -- so a truncf feeding a store
    // keeps pointing at the same MRAM buffer.
    if (isa<arith::TruncFOp, arith::ExtFOp>(op)) {
      if (const BufferView *src = state.tryLookup(op->getOperand(0)))
        state.record(op->getResult(0), *src);
      return success();
    }

    if (auto dot = dyn_cast<triton::DotOp>(op)) {
      const BufferView *a = state.tryLookup(dot.getA());
      const BufferView *w = state.tryLookup(dot.getB());
      if (!a || !w)
        return dot.emitError() << "pim-lower-to-emitc could not resolve "
                                  "tt.dot's operands to MRAM buffers";
      if (failed(checkElementType(
              dot,
              cast<RankedTensorType>(dot.getA().getType()).getElementType())))
        return failure();

      int64_t M = a->logicalRows(), K = a->logicalCols(), N = w->logicalCols();
      if (w->logicalRows() != K)
        return dot.emitError()
              << "pim-lower-to-emitc: tt.dot operands disagree on the K "
                 "dimension (" << K << " vs " << w->logicalRows() << ")";

      // The accumulator seed. `tt.dot`'s C operand is either a splat constant
      // (the common case, including a tiled loop's hoisted initializer) or a
      // loop-carried value whose reduction this expansion subsumes.
      std::optional<APFloat> seed;
      Value c = dot.getC();
      if (auto barg = dyn_cast<BlockArgument>(c)) {
        // Loop-carried accumulator: trace to the loop's init operand.
        if (auto forOp = dyn_cast<scf::ForOp>(barg.getOwner()->getParentOp())) {
          unsigned idx = barg.getArgNumber();
          if (idx >= 1 && idx - 1 < forOp.getInitArgs().size())
            c = forOp.getInitArgs()[idx - 1];
        }
      }
      if (auto cst = c.getDefiningOp<arith::ConstantOp>())
        if (auto splat = dyn_cast<SplatElementsAttr>(cst.getValue()))
          seed = splat.getSplatValue<APFloat>();
      if (!seed)
        return dot.emitError() << "pim-lower-to-emitc only handles a "
                                  "splat-constant tt.dot accumulator seed";

      auto out = resolveDotOutput(dot, newFunc, loops);
      if (failed(out))
        return failure();

      emitDotLoops(b, loc, *a, *w, *out, M, K, N, *seed);
      state.record(dot.getResult(), *out);
      return success();
    }

    if (auto wramStore = dyn_cast<WRAMStoreOp>(op)) {
      if (const BufferView *src = state.tryLookup(wramStore.getSrc()))
        state.record(wramStore.getDst(), *src);
      return success();
    }
    if (isa<DmaStoreOp>(op))
      return success(); // written in place by the dot expansion

    if (auto tid = dyn_cast<TaskletIdOp>(op)) {
      tid.getResult().replaceAllUsesWith(b.create<emitc::ConstantOp>(
          loc, b.getI32Type(), b.getI32IntegerAttr(0)));
      return success();
    }
    if (auto did = dyn_cast<DpuIdOp>(op)) {
      did.getResult().replaceAllUsesWith(b.create<emitc::ConstantOp>(
          loc, b.getI32Type(), b.getI32IntegerAttr(0)));
      return success();
    }

    if (isa<triton::ReturnOp>(op))
      return success(); // lowerFunc emits func::ReturnOp

    // Everything else is dead address-computation scaffolding.
    return success();
  }

  //   for m, n: acc = seed; for k: acc += a[m,k]*b[k,n]; out[m,n] = acc
  // Reads one element of `buf` as f32, converting from f16 storage when
  // needed. Shared by the MRAM->local snapshot and (for the now-local
  // operand views) the dot loops themselves.
  static Value loadElem(OpBuilder &b, Location loc, const BufferView &buf,
                        Value row, Value col) {
    Type storage = storageTypeFor(b, buf.elemTy);
    Value off = buf.elementOffset(b, loc, row, col);
    Value lv = b.create<emitc::SubscriptOp>(
        loc, emitc::LValueType::get(storage), buf.ptr, ValueRange{off});
    Value raw = b.create<emitc::LoadOp>(loc, storage, lv);
    if (!buf.isF16())
      return raw;
    return b.create<emitc::CallOpaqueOp>(loc, TypeRange{b.getF32Type()},
                                         "pim_f16_to_f32", ValueRange{raw})
        .getResult(0);
  }

  // Copies `src` (an MRAM operand view, `rows x cols` logical) into a fresh
  // heap-allocated f32 buffer, and returns a BufferView over it. The caller
  // is responsible for `free`ing `local.ptr` (via `emitc.call_opaque "free"`)
  // once the dot loops are done with it.
  //
  // Two independent reasons this exists, both found by testing, not by
  // inspection:
  //
  // (1) Correctness: `mem_planner` reuses a dead activation's MRAM for this
  // op's own output once it has no more readers -- `x`'s address and `out`'s
  // can be, and in the real decode loop for o_proj *are*, the same bytes. The
  // hand-written NumPy kernel (`runtime/kernels.py`) is safe because
  // `_read_tensor_args` copies each operand out before computing; reading
  // `a`/`w` straight from MRAM inside the (m, n) loop nest below is not,
  // because by the time a later `n` iteration re-reads `x` the first
  // iteration's write to `out` may have already overwritten it. Caught by
  // comparing every real decode-loop launch against the NumPy kernel:
  // o_proj's error was ~100% relative, not float16 rounding noise.
  //
  // (2) Why heap and not a stack array: `w`'s local copy can be several MiB
  // (o_proj's is 4096*512*4 = 8MiB) and this kernel runs inside
  // `NumpyBackend`'s per-DPU `ThreadPoolExecutor` worker threads, whose
  // pthread stacks are the platform default (commonly 8MiB on Linux,
  // independent of the *process*'s `ulimit -s`) -- a `float buf[N]` that size
  // reliably segfaults there even though the same allocation is fine on the
  // main thread. Reproduced directly: an 8MiB `volatile float buf[...]` in a
  // C function called from a `ThreadPoolExecutor` worker crashes every time.
  //
  // Snapshots MRAM rows `[rowStart, rowStart+rowCount)` of `src` (all
  // columns) into a fresh heap buffer, addressed by *absolute* row index --
  // `local.constant` is offset by `-rowStart*cols` so that passing the same
  // absolute row value used elsewhere (e.g. to index `out`, which spans the
  // full M range) lands at the right place inside this smaller buffer.
  // `rowStart=0, rowCount=src.logicalRows()` reproduces the old
  // whole-operand snapshot exactly (constant stays 0); a tasklet block calls
  // this with just its own row range (see emitDotLoops below).
  static BufferView snapshotToLocal(OpBuilder &b, Location loc,
                                    const BufferView &src, int64_t rowStart,
                                    int64_t rowCount) {
    // Always copy in *logical* (post-transpose) row-major order: the local
    // buffer becomes an untransposed view, so the copy loop's own (row, col)
    // matches loadElem(src, row, col) via elementOffset -- which already
    // accounts for src.transposed. Taking rows/cols as separate caller
    // arguments (the earlier version of this function) let a caller pass
    // them in the wrong order relative to `src`'s own shape; that silently
    // walked out of bounds on `w` after a tt.trans and corrupted the heap
    // (found as a segfault, not a wrong-answer -- see the mismatch between
    // rows=N,cols=K passed in vs. src.logicalRows()==K,logicalCols()==N).
    int64_t cols = src.logicalCols();
    auto f32 = b.getF32Type();
    auto i32 = b.getI32Type();
    auto i64 = b.getI64Type();
    auto ptrTy = emitc::PointerType::get(f32);
    Value nbytes = b.create<emitc::ConstantOp>(
        loc, i64, b.getI64IntegerAttr(rowCount * cols * 4));
    Value buf =
        b.create<emitc::CallOpaqueOp>(loc, TypeRange{ptrTy}, "malloc",
                                      ValueRange{nbytes})
            .getResult(0);
    BufferView local;
    local.ptr = buf;
    local.rows = rowCount;
    local.cols = cols;
    local.rowStride = cols;
    local.colStride = 1;
    local.constant = -rowStart * cols;
    local.elemTy = f32; // already converted to f32 during the copy below

    auto cst = [&](int64_t v) -> Value {
      return b.create<emitc::ConstantOp>(loc, i32, b.getI32IntegerAttr(v));
    };
    Value c0 = cst(0), c1 = cst(1), cCols = cst(cols);
    Value cRowStart = cst(rowStart), cRowEnd = cst(rowStart + rowCount);
    b.create<emitc::ForOp>(
        loc, cRowStart, cRowEnd, c1, [&](OpBuilder &rb, Location loc, Value r) {
          rb.create<emitc::ForOp>(
              loc, c0, cCols, c1, [&](OpBuilder &cb, Location loc, Value c) {
                Value v = loadElem(cb, loc, src, r, c);
                Value off = local.elementOffset(cb, loc, r, c);
                Value lv = cb.create<emitc::SubscriptOp>(
                    loc, emitc::LValueType::get(f32), buf, ValueRange{off});
                cb.create<emitc::AssignOp>(loc, lv, v);
                cb.create<emitc::YieldOp>(loc);
              });
          rb.create<emitc::YieldOp>(loc);
        });
    return local;
  }

  // Emits `numTasklets` sequential, statically-bounded blocks splitting the M
  // dimension -- see the pass-level rationale in Passes.td. `w` has no
  // aliasing risk with `out` (never written), so it is snapshotted once,
  // shared read-only by every block; each block snapshots only its own row
  // range of `a` before writing that same row range of `out`, which is what
  // keeps the out-aliases-a safety property (see snapshotToLocal's comment)
  // intact when M is split.
  void emitDotLoops(OpBuilder &b, Location loc, const BufferView &aMram,
                    const BufferView &wMram, const BufferView &out,
                    int64_t M, int64_t K, int64_t N, APFloat seed) {
    auto i32 = b.getI32Type();
    auto f32 = b.getF32Type();
    auto cst = [&](int64_t v) -> Value {
      return b.create<emitc::ConstantOp>(loc, i32, b.getI32IntegerAttr(v));
    };
    Value c0 = cst(0), c1 = cst(1), cN = cst(N), cK = cst(K);

    BufferView w = snapshotToLocal(b, loc, wMram, /*rowStart=*/0,
                                   /*rowCount=*/wMram.logicalRows());

    auto loadAt = [&](OpBuilder &lb, const BufferView &buf, Value row,
                      Value col) -> Value {
      return loadElem(lb, loc, buf, row, col);
    };

    // Writes one f32 element, converting to f16 storage when needed.
    auto storeAt = [&](OpBuilder &sb, const BufferView &buf, Value row,
                       Value col, Value f32Val) {
      Type storage = storageTypeFor(sb, buf.elemTy);
      Value off = buf.elementOffset(sb, loc, row, col);
      Value lv = sb.create<emitc::SubscriptOp>(
          loc, emitc::LValueType::get(storage), buf.ptr, ValueRange{off});
      Value toStore = f32Val;
      if (buf.isF16())
        toStore = sb.create<emitc::CallOpaqueOp>(loc, TypeRange{storage},
                                                 "pim_f32_to_f16",
                                                 ValueRange{f32Val})
                      .getResult(0);
      sb.create<emitc::AssignOp>(loc, lv, toStore);
    };

    // Static split, unrolled here: M and numTasklets are both compile-time
    // constants, so there is no runtime `tid` value anywhere in the emitted
    // C. Mirrors runtime/kernels.py::tasklet_linear_kernel's
    // `rows_per_tasklet = ceil(m / num_tasklets)` exactly, including the
    // trailing-tasklet-does-nothing case when numTasklets > M.
    int64_t rowsPerTasklet = (M + numTasklets - 1) / numTasklets;
    for (int64_t tid = 0; tid < numTasklets; ++tid) {
      int64_t rowStart = tid * rowsPerTasklet;
      if (rowStart >= M)
        break;
      int64_t rowCount = std::min(rowsPerTasklet, M - rowStart);

      BufferView a = snapshotToLocal(b, loc, aMram, rowStart, rowCount);

      Value cRowStart = cst(rowStart), cRowEnd = cst(rowStart + rowCount);
      b.create<emitc::ForOp>(
          loc, cRowStart, cRowEnd, c1, [&](OpBuilder &mb, Location loc, Value m) {
            mb.create<emitc::ForOp>(
                loc, c0, cN, c1, [&](OpBuilder &nb, Location loc, Value n) {
                  Value acc = nb.create<emitc::VariableOp>(
                      loc, emitc::LValueType::get(f32),
                      emitc::OpaqueAttr::get(nb.getContext(), ""));
                  nb.create<emitc::AssignOp>(
                      loc, acc,
                      nb.create<emitc::ConstantOp>(loc, f32,
                                                   nb.getFloatAttr(f32, seed)));
                  nb.create<emitc::ForOp>(
                      loc, c0, cK, c1,
                      [&](OpBuilder &kb, Location loc, Value k) {
                        Value av = loadAt(kb, a, m, k);
                        Value bv = loadAt(kb, w, k, n);
                        Value prod = kb.create<emitc::MulOp>(loc, f32, av, bv);
                        Value cur = kb.create<emitc::LoadOp>(loc, f32, acc);
                        kb.create<emitc::AssignOp>(
                            loc, acc,
                            kb.create<emitc::AddOp>(loc, f32, cur, prod));
                        kb.create<emitc::YieldOp>(loc);
                      });
                  storeAt(nb, out, m, n,
                          nb.create<emitc::LoadOp>(loc, f32, acc));
                  nb.create<emitc::YieldOp>(loc);
                });
            mb.create<emitc::YieldOp>(loc);
          });

      b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "free", ValueRange{a.ptr});
    }

    b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "free", ValueRange{w.ptr});
  }

  // Walks forward from a dot result -- through any accumulator-carrying loops
  // it feeds -- to the wram_store/dma_store pair pim-explicit-dma inserts
  // around a value written back to MRAM, and builds the output's view from
  // that dma_store's own address analysis.
  FailureOr<BufferView> resolveDotOutput(triton::DotOp dot,
                                         func::FuncOp newFunc,
                                         ArrayRef<LoopInfo> loops) {
    SmallVector<Value, 4> worklist{dot.getResult()};
    SmallPtrSet<Value, 4> seen;
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      for (auto &use : v.getUses()) {
        Operation *user = use.getOwner();
        // scf.yield forwards the accumulator to the loop's result.
        if (auto yield = dyn_cast<scf::YieldOp>(user)) {
          if (auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp()))
            worklist.push_back(forOp.getResult(use.getOperandNumber()));
          continue;
        }
        // A precision conversion on the way to the store (f32 accumulator ->
        // f16 storage) is transparent here: the emitted C converts per
        // element, driven by the destination buffer's own type.
        if (isa<arith::TruncFOp, arith::ExtFOp>(user)) {
          worklist.push_back(user->getResult(0));
          continue;
        }
        auto wramStore = dyn_cast<WRAMStoreOp>(user);
        if (!wramStore)
          continue;
        for (auto &dstUse : wramStore.getDst().getUses()) {
          auto dmaStore = dyn_cast<DmaStoreOp>(dstUse.getOwner());
          if (!dmaStore)
            continue;
          auto base =
              resolveBaseArg(dmaStore, dmaStore.getBaseArgAttr(), newFunc);
          if (failed(base))
            return failure();
          auto shape = get2DShape(
              dmaStore, dmaStore.getMemDescType().getShape());
          if (failed(shape))
            return failure();
          OffsetAnalysis analysis{loops};
          auto off = analysis.analyzePtr(dmaStore.getPtr());
          if (!off)
            return dmaStore.emitError()
                  << "pim-lower-to-emitc could not reduce the output "
                     "transfer's address computation to an affine form";
          return makeView(dmaStore, *base, shape->first, shape->second,
                          dmaStore.getMemDescType().getElementType(), *off,
                          loops);
        }
      }
    }
    return dot.emitError()
          << "pim-lower-to-emitc could not resolve tt.dot's output "
             "storage (expected wram_store -> dma_store -> base_arg)";
  }
};

} // namespace
