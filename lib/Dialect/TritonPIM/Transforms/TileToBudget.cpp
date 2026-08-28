#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include <tuple>

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMTILETOBUDGET
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

struct TileShape {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

static bool isPowerOfTwo(int64_t value) {
  return value > 0 && ((value & (value - 1)) == 0);
}

static std::optional<int64_t> constantInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    if (auto i = dyn_cast<IntegerAttr>(cst.getValue()))
      return i.getInt();
  }
  return std::nullopt;
}

static std::optional<TileShape> inferVisibleTile(triton::DotOp dot) {
  auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto dTy = dyn_cast<RankedTensorType>(dot.getType());
  if (!aTy || !bTy || !dTy)
    return std::nullopt;
  if (aTy.getRank() != 2 || bTy.getRank() != 2 || dTy.getRank() != 2)
    return std::nullopt;
  return TileShape{aTy.getShape()[0], dTy.getShape()[1], aTy.getShape()[1]};
}

// Reads a loop's [lb, ub) as a full extent: requires a constant, zero lower
// bound and a positive step (the step itself -- the tile size along that
// axis -- doesn't matter here, only the upper bound, which is the untiled
// extent by construction in the `for i0 in range(0, EXTENT, TILE)` pattern
// every front-end tile loop in this kernel follows).
static std::optional<int64_t> loopFullExtent(scf::ForOp forOp) {
  auto lb = constantInt(forOp.getLowerBound());
  auto ub = constantInt(forOp.getUpperBound());
  auto step = constantInt(forOp.getStep());
  if (!lb || !ub || !step || *lb != 0 || *step <= 0)
    return std::nullopt;
  return *ub;
}

// Recovers the untiled M/N/K a `tt.dot` sits inside, given the front-end may
// have wrapped it in up to two void tiling loops (M outer, N inner -- or just
// one, or neither) plus at most one K-reduction loop carrying the
// accumulator. Unlike a shape-matching heuristic, the K-reduction loop is
// identified structurally: it is the loop whose region iter-arg *is* the
// dot's own accumulator operand, so it can never be confused with an M- or
// N-tiling loop even when a tile's visible shape happens to already equal
// the untiled M/N (as happens for the real llama2-7b o_proj shape, where the
// K-reduction loop's per-iteration result shape is 1x512 -- already equal to
// the visible tile's (M, N) -- so a shape-based comparison silently never
// touches K, which is the bug this replaced).
static std::optional<TileShape> inferFullShape(triton::DotOp dot) {
  auto visible = inferVisibleTile(dot);
  if (!visible)
    return std::nullopt;

  int64_t m = visible->m;
  int64_t n = visible->n;
  int64_t k = visible->k;

  scf::ForOp kLoop;
  if (auto barg = dyn_cast<BlockArgument>(dot.getC()))
    kLoop = dyn_cast_or_null<scf::ForOp>(barg.getOwner()->getParentOp());

  // Void (non-reduction) tiling loops on the way out, innermost first.
  SmallVector<scf::ForOp, 2> voidLoops;
  for (Operation *parent = dot->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp)
      continue;
    if (forOp == kLoop) {
      auto extent = loopFullExtent(forOp);
      if (!extent)
        return std::nullopt;
      k = *extent;
    } else if (forOp.getNumResults() == 0) {
      voidLoops.push_back(forOp);
    } else {
      // A loop-carried loop that isn't the dot's own accumulator loop: not a
      // pattern this pass understands, don't guess.
      return std::nullopt;
    }
  }

  // Known front-end nesting is outermost-to-innermost M, then N, then K, so
  // `voidLoops` (collected innermost-first) is [N-loop] or [N-loop, M-loop].
  if (voidLoops.size() > 2)
    return std::nullopt;
  if (voidLoops.size() >= 1) {
    auto extent = loopFullExtent(voidLoops[0]);
    if (!extent)
      return std::nullopt;
    n = *extent;
  }
  if (voidLoops.size() == 2) {
    auto extent = loopFullExtent(voidLoops[1]);
    if (!extent)
      return std::nullopt;
    m = *extent;
  }
  return TileShape{m, n, k};
}

static std::optional<int64_t> inferDtypeSize(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty)
    return std::nullopt;
  Type elemTy = ty.getElementType();
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  if (bits % 8 != 0)
    return std::nullopt;
  return bits / 8;
}

// A dimension the rewrite may or may not loop over. When `tile < full`, the
// dimension is split into `full/tile` iterations of a real `scf.for`; when
// `tile == full`, no loop is built at all and the whole extent is handled in
// one shot -- this is what keeps the M=1 (llama2-7b decode) and pre-existing
// N/K-tiled-but-already-in-budget cases producing byte-identical IR to
// today, since `needM`/`needN`/`needK` are all false when the front-end tile
// already satisfies the budget (see the `!needRewrite` early-out in
// `runOnOperation`).
struct TileDim {
  int64_t tile;
  int64_t full;
  bool needsLoop() const { return tile < full; }
};

// One dimension's contribution to the rebuilt kernel: either a loop (`forOp`
// set, `iv` is its induction variable) or a flat range covering the whole
// extent (`forOp` null, `iv` null -- callers use the dimension's `tile`/
// `full` directly as the make_range bounds).
struct BuiltDim {
  TileDim dim;
  scf::ForOp forOp;
  Value iv; // valid only when forOp is set
};

// Rebuilds the `linear` kernel's N/K (and, when needed, M) tiling loops and
// the `tt.dot` inside them, replacing the front-end's original loop nest
// with one that uses `newTile` instead. Mirrors `kernel_src.py::
// linear_kernel`'s op sequence one-for-one (see that file's docstring for
// why offsets are computed as one combined tensor before a single
// `tt.addptr`, and why K is always tiled at the TTIR level): this pass only
// ever changes tile *sizes*, never the shape of the address-computation
// pattern `pim-explicit-dma`/`pim-lower-to-emitc` already know how to trace.
class LinearKernelRewriter {
public:
  LinearKernelRewriter(triton::DotOp dot, TileShape full, TileShape newTile,
                      int numTasklets, int numDpus)
      : dot(dot), full(full), newTile(newTile), numTasklets(numTasklets),
        numDpus(numDpus), ctx(dot.getContext()) {}

  // Traces a tensor-of-pointers value back to the scalar function argument
  // it was `tt.splat`'d from, optionally through one `tt.addptr` (the
  // kernel_src.py shape always has one; a hand-written test tile that
  // already covers the whole tensor may `tt.load`/`tt.store` the splat
  // directly, with no addressing on top).
  static std::optional<Value> resolveBasePtr(Value ptrs) {
    if (auto addptr = ptrs.getDefiningOp<triton::AddPtrOp>())
      ptrs = addptr.getPtr();
    auto splat = ptrs.getDefiningOp<triton::SplatOp>();
    if (!splat)
      return std::nullopt;
    return splat.getSrc();
  }

  // Erases `op` and then walks its (former) operands, erasing each one in
  // turn if it has become unused and would be trivially dead if so (no
  // remaining side effects that matter -- a `tt.load` with no other uses
  // and not volatile qualifies: it only reads global memory, and nothing
  // in this kernel's contract depends on unread values being read anyway).
  // Needed because plain `op->erase()` only removes `op` itself: the
  // address-computation chain that fed the tile this rewrite is replacing
  // (`tt.load`, `tt.addptr`, `tt.expand_dims`, ...) would otherwise survive
  // as dead-but-present ops, and `pim-explicit-dma` counts every
  // `tt.load`/`tt.store` in the module toward WRAM usage regardless of
  // whether their results are used.
  // `roots` may overlap (one may be an ancestor of another, or they may
  // share operands transitively) -- all of them are erased first, as a
  // batch, before any operand-chain sweeping starts. Doing this in two
  // phases (erase every named root, *then* sweep) rather than one
  // `eraseTreeIfDead`-per-root call is what makes overlapping roots safe:
  // sweeping after the first root's erase could otherwise reach into a
  // second root that hasn't been erased yet (double-erase, UB) or leave a
  // root's own now-dead operand chain unswept because the sweep only ever
  // started from one root at a time.
  static void eraseTreesIfDead(ArrayRef<Operation *> roots) {
    // A `Value` reaches the worklist once per *use* it had among the ops
    // erased so far, so a value shared by two dead ops (a `tt.make_range`
    // feeding both a dead x-offset and w-offset computation, say) can be
    // pushed twice. Once the first occurrence's defining op is erased, a
    // second occurrence of the same `Value` handle is dangling --
    // re-deriving its defining op (`v.getDefiningOp()`) dereferences freed
    // memory (this crashed inside ~Operation before this dedup was added).
    // `enqueued` gates on `Value` identity *before* ever popping a
    // duplicate, so the same value is never inspected twice regardless of
    // how many dead ops shared it.
    //
    // Separately: `roots` are all force-erased unconditionally (they are
    // known-dead by construction, not checked via `wouldOpBeTriviallyDead`)
    // -- one root can be another root's operand (`store` consumes
    // `truncOp`'s result, `truncOp` consumes `dot`'s result, and `dot` is
    // `root`). If a later root's result got enqueued while collecting an
    // earlier root's operands, that `Value` handle would go dangling the
    // moment its defining op (itself a root, erased later in this same
    // loop) is destroyed -- popping it afterwards is the same
    // already-freed-memory crash. So results of other roots must never
    // reach the sweep worklist at all; only genuinely external operands
    // (the real producers feeding into this whole dead region: `a`, `w`,
    // the accumulator seed, the top-level `tt.make_range`s, ...) should.
    llvm::SmallPtrSet<Operation *, 4> rootSet(roots.begin(), roots.end());
    llvm::SmallPtrSet<void *, 16> enqueued;
    SmallVector<Value> worklist;
    auto enqueue = [&](Value v) {
      Operation *def = v.getDefiningOp();
      if (def && rootSet.contains(def))
        return;
      if (enqueued.insert(v.getAsOpaquePointer()).second)
        worklist.push_back(v);
    };
    for (Operation *root : roots) {
      for (Value v : root->getOperands())
        enqueue(v);
      root->erase();
    }
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      Operation *def = v.getDefiningOp();
      if (!def || !def->use_empty() || !wouldOpBeTriviallyDead(def))
        continue;
      for (Value operand : def->getOperands())
        enqueue(operand);
      def->erase();
    }
  }

  // Locates the store(s) this dot's result eventually reaches, the pointer
  // arguments involved, and the outermost op of the region this rewrite
  // needs to replace. Returns failure (with a diagnostic already emitted) if
  // the surrounding IR isn't the shape this pass understands.
  LogicalResult analyze() {
    auto aTy = cast<RankedTensorType>(dot.getA().getType());
    elemTy = aTy.getElementType();

    // x[m, k] = x_ptr + (m*K + k); recovered by walking the one `tt.addptr`
    // feeding the dot's `a` operand through its `tt.load`. `pim.
    // convert_layout` is a same-shape/same-type passthrough
    // `convert-triton-to-pim` inserts wherever a value's layout doesn't
    // already match its consumer (e.g. after `tt.trans`, whose result
    // layout is a transposed version of its input's); skip through it like
    // `LowerPIMToEmitC.cpp`'s own analysis does.
    Value bVal = dot.getB();
    while (auto cvt = bVal.getDefiningOp<ConvertLayoutOp>())
      bVal = cvt.getSrc();
    Value aVal = dot.getA();
    while (auto cvt = aVal.getDefiningOp<ConvertLayoutOp>())
      aVal = cvt.getSrc();
    auto xLoad = aVal.getDefiningOp<triton::LoadOp>();
    auto wLoad = bVal.getDefiningOp<triton::TransOp>();
    if (!xLoad)
      return dot.emitError() << "pim-tile-to-budget rewrite expects a's tt.load"
                                 " directly feeding tt.dot";
    if (!wLoad)
      return dot.emitError() << "pim-tile-to-budget rewrite expects a tt.trans"
                                 " directly feeding tt.dot's b operand";
    auto wLoadOp = wLoad.getSrc().getDefiningOp<triton::LoadOp>();
    if (!wLoadOp)
      return dot.emitError() << "pim-tile-to-budget rewrite expects tt.trans's"
                                 " operand to be a tt.load";
    auto xPtrOpt = resolveBasePtr(xLoad.getPtr());
    auto wPtrOpt = resolveBasePtr(wLoadOp.getPtr());
    if (!xPtrOpt)
      return dot.emitError() << "pim-tile-to-budget rewrite expects tt.dot's a"
                                 " operand to trace back to a tt.splat of a"
                                 " function argument, through at most one"
                                 " tt.addptr";
    if (!wPtrOpt)
      return dot.emitError() << "pim-tile-to-budget rewrite expects tt.dot's b"
                                 " operand to trace back to a tt.splat of a"
                                 " function argument, through at most one"
                                 " tt.addptr";
    xPtr = *xPtrOpt;
    wPtr = *wPtrOpt;

    // Walk from the dot up to the outermost enclosing op that is still part
    // of this kernel's loop nest (an `scf.for`, possibly nested), and find
    // the store(s) fed by the (possibly loop-yielded) dot result.
    Operation *outer = dot;
    while (isa<scf::ForOp>(outer->getParentOp()))
      outer = outer->getParentOp();
    root = outer;

    // The dot's result reaches a store either directly or through
    // scf.yield -> loop result -> (optionally) arith.truncf -> tt.store.
    Value result = dot.getResult();
    if (auto forOp = dyn_cast<scf::ForOp>(dot->getParentOp())) {
      // dot feeds scf.yield inside a K-reduction loop; find which result.
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      unsigned idx = 0;
      bool found = false;
      for (auto [i, operand] : llvm::enumerate(yield.getOperands()))
        if (operand == result) {
          idx = i;
          found = true;
          break;
        }
      if (!found)
        return dot.emitError() << "pim-tile-to-budget rewrite could not find"
                                   " tt.dot's result in its loop's scf.yield";
      result = forOp.getResult(idx);
    }

    Value storeVal = result;
    if (result.hasOneUse()) {
      Operation *trunc = *result.getUsers().begin();
      if (isa<arith::TruncFOp>(trunc)) {
        truncOp = trunc;
        storeVal = trunc->getResult(0);
      }
    }
    if (!storeVal.hasOneUse())
      return dot.emitError() << "pim-tile-to-budget rewrite expects the dot"
                                 " result (or its truncf) to have one use";
    store = dyn_cast<triton::StoreOp>(*storeVal.getUsers().begin());
    if (!store)
      return dot.emitError() << "pim-tile-to-budget rewrite expects the dot"
                                 " result to reach exactly one tt.store";
    auto oAddPtr = store.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!oAddPtr)
      return dot.emitError() << "pim-tile-to-budget rewrite expects tt.store's"
                                 " pointer to come from a tt.addptr";
    auto oSplat = oAddPtr.getPtr().getDefiningOp<triton::SplatOp>();
    if (!oSplat)
      return dot.emitError() << "pim-tile-to-budget rewrite expects tt.store's"
                                 " tt.addptr base to be a tt.splat";
    outPtr = oSplat.getSrc();
    truncToStorage = storeVal != result;
    return success();
  }

  // Builds the replacement nest and erases `root`. On success, `root` (and
  // everything under it, plus the old store) is gone; the caller must not
  // reference it again.
  void rewrite() {
    OpBuilder b(root);
    Location loc = dot.getLoc();

    TileDim mDim{newTile.m, full.m};
    TileDim nDim{newTile.n, full.n};
    TileDim kDim{newTile.k, full.k};

    // M is outermost, then N, then K -- matching kernel_src.py's nesting
    // (`for n0 in range(N): for k0 in range(K): ...`; M is never looped by
    // the current front-end, but this rewrite treats it the same way N/K
    // are treated so a future front-end that does emit an M loop needs no
    // change here).
    BuiltDim mBuilt = buildOuterDim(b, loc, mDim);
    OpBuilder nBody = mBuilt.forOp ? OpBuilder::atBlockBegin(mBuilt.forOp.getBody())
                                   : b;
    Value offsM = makeOffsets(nBody, loc, mDim, mBuilt.iv);

    BuiltDim nBuilt = buildOuterDim(nBody, loc, nDim);
    OpBuilder kBody = nBuilt.forOp ? OpBuilder::atBlockBegin(nBuilt.forOp.getBody())
                                   : nBody;
    Value offsN = makeOffsets(kBody, loc, nDim, nBuilt.iv);

    Value acc = buildDotAndK(kBody, loc, kDim, offsM, offsN, mDim, nDim);

    // Close the K loop (if any): yield the accumulator.
    if (nBuilt.forOp && kDim.needsLoop()) {
      // acc already belongs to the K loop's body via buildDotAndK; nothing
      // extra needed here, buildDotAndK emits the scf.yield itself.
    }

    // Write out: this happens once per (M,N) tile, i.e. inside the M/N loops
    // but outside the K loop.
    emitStore(kBody, loc, acc, offsM, offsN, mDim, nDim);

    // If N (or M) was a real loop, buildDotAndK already placed everything
    // inside its body; nothing else to hook up -- the loops themselves are
    // already inserted at the right point by buildOuterDim.

    // `root`, `store`, and `truncOp` (if any) are three separate roots of
    // now-dead code that need erasing:
    //  - `root->erase()` alone would only remove `root` itself (which, when
    //    the front-end fully unrolled its loops, is just the old `dot`, not
    //    an `scf.for`) and leave the *entire* old address-computation chain
    //    feeding it (`tt.load`/`tt.addptr`/`tt.expand_dims`/...) behind as
    //    dead-but-present ops: `pim-explicit-dma` counts every `tt.load`/
    //    `tt.store` in the module toward WRAM usage regardless of whether
    //    their results are used, so stale loads left behind would silently
    //    inflate that check with garbage from the tile just replaced.
    //  - `store` (and `truncOp`) have their *own* independent operand chain
    //    (the old `o_off`/`m2o`/... address computation, which draws from
    //    the same top-level `tt.make_range`s as the load side but is not
    //    reachable from `dot`'s operands) that plain `store->erase()` would
    //    leave dangling for the same reason.
    //  - `store`/`truncOp` are not always descendants of `root`: only when
    //    an M or N tiling loop happens to enclose them (the real llama2-7b
    //    shape) does `root`'s subtree already contain them; when the
    //    front-end fully unrolled (small test shapes) they sit as `root`'s
    //    siblings instead.
    // `eraseTreesIfDead` erases all applicable roots as one batch before
    // sweeping any operands, so it is safe even when one root turns out to
    // be an ancestor of another (erasing an already-erased op is
    // undefined behavior, which erasing them one at a time risked).
    // Order matters: `store` consumes `truncOp` (or `dot`'s result
    // directly), and `truncOp` consumes `dot`'s result, so consumers must
    // be erased before the producer they reference -- `eraseTreesIfDead`
    // erases every entry in `roots` before sweeping any of their operands,
    // so listing `root` (which may itself be `dot`) first would erase a
    // still-referenced op out from under `store`/`truncOp`.
    SmallVector<Operation *> roots;
    if (!root->isAncestor(store))
      roots.push_back(store);
    if (truncOp && !root->isAncestor(truncOp) && truncOp != store)
      roots.push_back(truncOp);
    roots.push_back(root);
    eraseTreesIfDead(roots);
  }

private:
  triton::DotOp dot;
  TileShape full;
  TileShape newTile;
  int numTasklets;
  int numDpus;
  MLIRContext *ctx;

  Type elemTy;
  Value xPtr, wPtr, outPtr;
  triton::StoreOp store;
  Operation *truncOp = nullptr; // set only when truncToStorage
  bool truncToStorage = false;
  Operation *root = nullptr;

  RankedTensorType tensorTy(ArrayRef<int64_t> shape, Type elem) {
    auto enc = getDefaultTaskletTiledEncoding(ctx, shape, numTasklets, numDpus);
    return RankedTensorType::get(shape, elem, enc);
  }

  Value constI32(OpBuilder &b, Location loc, int32_t v) {
    return b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(v));
  }

  // Inserts a `pim.convert_layout` from `v`'s current encoding to the
  // default one for `shape`, unless it already has that encoding (in which
  // case `v` is returned unchanged -- avoids a redundant identity convert
  // on every offset computation, matching what `convert-triton-to-pim`'s
  // own materialization hook does: convert only when the layouts actually
  // differ).
  Value convertToDefaultLayout(OpBuilder &b, Location loc, Value v,
                               ArrayRef<int64_t> shape) {
    auto ty = cast<RankedTensorType>(v.getType());
    auto wantTy = tensorTy(shape, ty.getElementType());
    if (ty == wantTy)
      return v;
    return b.create<ConvertLayoutOp>(loc, wantTy, v);
  }

  // Builds (if `dim.needsLoop()`) an `scf.for 0 to full step tile` at `b`'s
  // current insertion point, leaving the builder's insertion point
  // unchanged (the loop is a sibling of whatever comes next at this level);
  // callers get a fresh builder positioned inside the loop body via the
  // returned `BuiltDim`. When no loop is needed, returns a `BuiltDim` with
  // `forOp` null and `iv` null; the dimension's offsets then span its full
  // extent starting at 0.
  BuiltDim buildOuterDim(OpBuilder &b, Location loc, TileDim dim) {
    if (!dim.needsLoop())
      return {dim, nullptr, nullptr};
    Value lb = constI32(b, loc, 0);
    Value ub = constI32(b, loc, static_cast<int32_t>(dim.full));
    Value step = constI32(b, loc, static_cast<int32_t>(dim.tile));
    auto forOp = b.create<scf::ForOp>(loc, lb, ub, step, ValueRange{});
    return {dim, forOp, forOp.getInductionVar()};
  }

  // `tt.make_range(0, dim.tile) [+ tt.splat(iv) + arith.addi]`, i.e. the
  // per-tile offsets along one axis -- `offs_n`/`offs_k`/`offs_m` in
  // kernel_src.py. When there is no loop, this is just the flat range over
  // the whole extent (`dim.tile == dim.full` in that case).
  Value makeOffsets(OpBuilder &b, Location loc, TileDim dim, Value iv) {
    auto rangeTy = tensorTy({dim.tile}, b.getI32Type());
    Value range = b.create<triton::MakeRangeOp>(
        loc, rangeTy, static_cast<uint32_t>(0), static_cast<uint32_t>(dim.tile));
    if (!iv)
      return range;
    Value splatIv = b.create<triton::SplatOp>(loc, rangeTy, iv);
    return b.create<arith::AddIOp>(loc, splatIv, range);
  }

  // Builds the K loop (if needed) and the dot body inside it (or, with no K
  // loop, the single dot directly), returning the final accumulator value.
  // `offsM`/`offsN` are already-built 1-D offset tensors for this (M,N)
  // tile; `mDim`/`nDim` give their tile sizes for shaping intermediate
  // tensors.
  Value buildDotAndK(OpBuilder &b, Location loc, TileDim kDim, Value offsM,
                     Value offsN, TileDim mDim, TileDim nDim) {
    auto accTy = tensorTy({mDim.tile, nDim.tile}, b.getF32Type());
    Value zero = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(accTy, b.getF32FloatAttr(0.0f)));

    if (!kDim.needsLoop()) {
      Value offsK = makeOffsets(b, loc, kDim, nullptr);
      return buildDotBody(b, loc, offsM, offsN, offsK, mDim, nDim, kDim, zero);
    }

    Value lb = constI32(b, loc, 0);
    Value ub = constI32(b, loc, static_cast<int32_t>(kDim.full));
    Value step = constI32(b, loc, static_cast<int32_t>(kDim.tile));
    auto forOp = b.create<scf::ForOp>(loc, lb, ub, step, ValueRange{zero});
    OpBuilder body = OpBuilder::atBlockBegin(forOp.getBody());
    Value offsK = makeOffsets(body, loc, kDim, forOp.getInductionVar());
    Value newAcc = buildDotBody(body, loc, offsM, offsN, offsK, mDim, nDim,
                                kDim, forOp.getRegionIterArgs()[0]);
    body.create<scf::YieldOp>(loc, ValueRange{newAcc});
    return forOp.getResult(0);
  }

  // x_off = offsM[:,None]*K + offsK[None,:]; w_off = offsN[:,None]*K +
  // offsK[None,:]; load both, transpose w, dot into `acc`. One-to-one with
  // kernel_src.py's inner-loop body.
  Value buildDotBody(OpBuilder &b, Location loc, Value offsM, Value offsN,
                     Value offsK, TileDim mDim, TileDim nDim, TileDim kDim,
                     Value acc) {
    Value xOff = buildOperandOffset(b, loc, offsM, offsK, mDim.tile, kDim.tile,
                                    full.k);
    Value wOff = buildOperandOffset(b, loc, offsN, offsK, nDim.tile, kDim.tile,
                                    full.k);

    auto xPtrTy = tensorTy({mDim.tile, kDim.tile}, triton::PointerType::get(
                                                       elemTy, /*addrspace=*/1));
    Value xSplat = b.create<triton::SplatOp>(loc, xPtrTy, xPtr);
    Value xAddr = b.create<triton::AddPtrOp>(loc, xPtrTy, xSplat, xOff);
    Value xBlk = b.create<triton::LoadOp>(loc, xAddr, triton::CacheModifier::NONE,
                                          triton::EvictionPolicy::NORMAL,
                                          /*isVolatile=*/false);

    auto wPtrTy = tensorTy({nDim.tile, kDim.tile}, triton::PointerType::get(
                                                       elemTy, /*addrspace=*/1));
    Value wSplat = b.create<triton::SplatOp>(loc, wPtrTy, wPtr);
    Value wAddr = b.create<triton::AddPtrOp>(loc, wPtrTy, wSplat, wOff);
    Value wBlk = b.create<triton::LoadOp>(loc, wAddr, triton::CacheModifier::NONE,
                                          triton::EvictionPolicy::NORMAL,
                                          /*isVolatile=*/false);

    Value wBlkT = b.create<triton::TransOp>(loc, wBlk, ArrayRef<int32_t>{1, 0});
    return b.create<triton::DotOp>(loc, xBlk, wBlkT, acc,
                                   triton::InputPrecision::IEEE, 0u);
  }

  // `rowOffs[:,None]*stride + colOffs[None,:]`, matching kernel_src.py's
  // `offs_m[:, None] * K + offs_k[None, :]` (and the symmetric `w_off`
  // computation with N in place of M). `rowOffs` gets axis=0, `colOffs`
  // axis=0 -- matching kernel_src.py exactly (`offs_m[:, None] * K +
  // offs_k[None, :]`): the row index is expand_dims'd along axis 1 (shape
  // `rows x 1`), scaled by `stride` *before* broadcasting (so the multiply
  // is on the small `rows x 1` tensor, not the full `rows x cols` one --
  // this op-for-op shape match matters because `pim-explicit-dma`'s
  // `traceBaseArg`/offset analysis and `pim-lower-to-emitc`'s
  // `OffsetAnalysis` both walk this exact op sequence and give up on
  // anything else), while the column index is expand_dims'd along axis 0
  // (shape `1 x cols`) and broadcasts unscaled.
  Value buildOperandOffset(OpBuilder &b, Location loc, Value rowOffs,
                           Value colOffs, int64_t rows, int64_t cols,
                           int64_t stride) {
    Value rowExp = b.create<triton::ExpandDimsOp>(loc, rowOffs, 1u);
    Value colExp = b.create<triton::ExpandDimsOp>(loc, colOffs, 0u);

    // `tt.broadcast` requires SameOperandsAndResultEncoding: the *encoding
    // Attribute itself* (not just the shape) must be identical between
    // operand and result, so broadcasting `rowExp`/`colExp` up to
    // `{rows, cols}` keeps each one's own (possibly different) encoding --
    // there is no single `{rows, cols}` type both can share yet. `cst`
    // needs the same encoding as `rowExp` for the same reason (`arith.
    // muli` has the same trait). Only the final `arith.addi` needs its two
    // operands reconciled to one encoding, via a `pim.convert_layout` --
    // exactly the pattern `convert-triton-to-pim` itself produces for this
    // same op sequence on the real llama2-7b IR (verified against
    // `TritonPIMConversion.cpp`'s target-materialization hook).
    auto rowExpTy = cast<RankedTensorType>(rowExp.getType());
    Value strideCst = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(rowExpTy, b.getI32IntegerAttr(
                                                  static_cast<int32_t>(stride))));
    Value rowScaled = b.create<arith::MulIOp>(loc, rowExp, strideCst);
    auto rowFullTy = RankedTensorType::get({rows, cols}, b.getI32Type(),
                                           rowExpTy.getEncoding());
    Value rowBcast = b.create<triton::BroadcastOp>(loc, rowFullTy, rowScaled);

    auto colExpTy = cast<RankedTensorType>(colExp.getType());
    auto colFullTy = RankedTensorType::get({rows, cols}, b.getI32Type(),
                                           colExpTy.getEncoding());
    Value colBcast = b.create<triton::BroadcastOp>(loc, colFullTy, colExp);

    Value rowBcastNorm = convertToDefaultLayout(b, loc, rowBcast, {rows, cols});
    Value colBcastNorm = convertToDefaultLayout(b, loc, colBcast, {rows, cols});
    return b.create<arith::AddIOp>(loc, rowBcastNorm, colBcastNorm);
  }

  // `out[m, n] = acc` (optionally narrowed to storage dtype first) -- one
  // store per (M,N) tile, placed at whatever insertion point `b` currently
  // has (inside the M/N loops, outside the K loop).
  void emitStore(OpBuilder &b, Location loc, Value acc, Value offsM,
                 Value offsN, TileDim mDim, TileDim nDim) {
    Value oOff = buildOperandOffset(b, loc, offsM, offsN, mDim.tile, nDim.tile,
                                    full.n);
    auto oPtrTy = tensorTy({mDim.tile, nDim.tile},
                           triton::PointerType::get(elemTy, /*addrspace=*/1));
    Value oSplat = b.create<triton::SplatOp>(loc, oPtrTy, outPtr);
    Value oAddr = b.create<triton::AddPtrOp>(loc, oPtrTy, oSplat, oOff);
    Value toStore = acc;
    if (truncToStorage) {
      auto storageTy = tensorTy({mDim.tile, nDim.tile}, elemTy);
      toStore = b.create<arith::TruncFOp>(loc, storageTy, acc);
    }
    b.create<triton::StoreOp>(loc, oAddr, toStore, triton::CacheModifier::NONE,
                              triton::EvictionPolicy::NORMAL);
  }
};

static LogicalResult validateDot(triton::DotOp dot) {
  auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto dTy = dyn_cast<RankedTensorType>(dot.getType());
  if (!aTy || !bTy || !dTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      dTy.getRank() != 2)
    return dot.emitError() << "pim-tile-to-budget only handles the current"
                           << " 2-D linear tt.dot pattern";
  if (aTy.getShape()[1] != bTy.getShape()[0])
    return dot.emitError() << "tt.dot operands disagree on K";
  if (aTy.getShape()[0] <= 0 || bTy.getShape()[0] <= 0 ||
      dTy.getShape()[0] <= 0 || dTy.getShape()[1] <= 0)
    return dot.emitError() << "tt.dot shapes must be positive";
  if (!isPowerOfTwo(aTy.getShape()[0]) || !isPowerOfTwo(bTy.getShape()[0]) ||
      !isPowerOfTwo(aTy.getShape()[1]) || !isPowerOfTwo(dTy.getShape()[0]) ||
      !isPowerOfTwo(dTy.getShape()[1]))
    return dot.emitError() << "pim-tile-to-budget requires power-of-two tiles";
  return success();
}

static int64_t bytesFor(TileShape tile, int64_t elemBytes) {
  return (tile.m * tile.k + tile.n * tile.k + tile.m * tile.n) * elemBytes;
}

// Whether `tile` fits the WRAM budget and each of the three staged buffers
// (x, w, out) is separately DMA-aligned -- the design's per-buffer alignment
// requirement, stricter than (and a superset of) checking only the summed
// footprint.
static bool fitsBudget(TileShape tile, int64_t elemBytes, int64_t wram,
                       int64_t dma) {
  int64_t xBytes = tile.m * tile.k * elemBytes;
  int64_t wBytes = tile.n * tile.k * elemBytes;
  int64_t oBytes = tile.m * tile.n * elemBytes;
  if (xBytes + wBytes + oBytes > wram)
    return false;
  return xBytes % dma == 0 && wBytes % dma == 0 && oBytes % dma == 0;
}

// Every power-of-two divisor of `full`, descending (largest first): used to
// enumerate candidate tile sizes along one dimension.
static SmallVector<int64_t, 8> powerOfTwoDivisorsDesc(int64_t full) {
  SmallVector<int64_t, 8> vals;
  for (int64_t v = full; v >= 1; v /= 2)
    vals.push_back(v);
  return vals;
}

// Searches for a WRAM/DMA-legal tile no larger than `visible` in any
// dimension. Exhaustive over the power-of-two lattice below `visible`
// (bounded: at most log2(M)+1 * log2(N)+1 * log2(K)+1 candidates, a few
// hundred at the largest realistic shapes) rather than greedily shrinking
// one dimension to its floor before trying another -- a greedy single path
// can dead-end: e.g. shrinking K first to fix a WRAM overage can make x's
// per-tile byte count drop below `dma`, permanently failing the alignment
// check for every smaller K, even though a *different* K paired with a
// smaller N would satisfy both budget and alignment. Every candidate stays
// a power of two dividing the corresponding full extent, since `visible`
// and `full` are both already powers of two (guaranteed by `validateDot`
// and the `_compiled_linear_supports` gate upstream). Symmetric across
// M/N/K on purpose: nothing here assumes M is 1 or unsplit -- if a future
// shape needs the M dimension tiled too, this same search already finds it.
// Among all legal tiles, picks the one with the most elements (fewest loop
// iterations over the untiled extents), preferring to keep K largest, then
// N, then M when multiple tiles tie on element count -- matching the
// design's stated preference order without needing a separate tie-break
// pass. On failure, `*smallestTried` (if non-null) is left holding {1,1,1}
// clamped to `visible`, for a diagnostic.
static std::optional<TileShape> searchTile(TileShape visible, int64_t elemBytes,
                                           int64_t wram, int64_t dma,
                                           TileShape *smallestTried = nullptr) {
  std::optional<TileShape> best;
  for (int64_t m : powerOfTwoDivisorsDesc(visible.m))
    for (int64_t n : powerOfTwoDivisorsDesc(visible.n))
      for (int64_t k : powerOfTwoDivisorsDesc(visible.k)) {
        TileShape tile{m, n, k};
        if (!fitsBudget(tile, elemBytes, wram, dma))
          continue;
        if (!best || m * n * k > best->m * best->n * best->k ||
            (m * n * k == best->m * best->n * best->k &&
             std::tuple(k, n, m) > std::tuple(best->k, best->n, best->m)))
          best = tile;
      }
  if (best)
    return best;
  if (smallestTried)
    *smallestTried = TileShape{1, 1, 1};
  return std::nullopt;
}

class TritonPIMTileToBudget
    : public mlir::triton::pim::impl::TritonPIMTileToBudgetBase<
          TritonPIMTileToBudget> {
public:
  using mlir::triton::pim::impl::TritonPIMTileToBudgetBase<
      TritonPIMTileToBudget>::TritonPIMTileToBudgetBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto wram = maybeLookupWramBytes(mod);
    auto mram = maybeLookupMramBytes(mod);
    auto dma = maybeLookupDmaAlign(mod);
    if (!wram || !mram || !dma) {
      mod.emitError() << "pim-tile-to-budget requires " << AttrWramBytesName
                      << ", " << AttrMramBytesName << " and "
                      << AttrDmaAlignName;
      return signalPassFailure();
    }

    bool sawDot = false;
    SmallVector<triton::DotOp> dots;
    mod.walk([&](triton::DotOp dot) { dots.push_back(dot); });
    if (dots.empty()) {
      mod.emitError() << "pim-tile-to-budget requires at least one tt.dot";
      return signalPassFailure();
    }

    std::optional<TileShape> chosen;
    std::optional<int64_t> elemBytes;
    for (triton::DotOp dot : dots) {
      sawDot = true;
      if (failed(validateDot(dot)))
        return signalPassFailure();
      auto visible = inferVisibleTile(dot);
      auto full = inferFullShape(dot);
      if (!visible || !full) {
        dot.emitError() << "pim-tile-to-budget could not infer the linear"
                           " tile shape";
        return signalPassFailure();
      }
      auto bytes = inferDtypeSize(dot.getA());
      if (!bytes) {
        dot.emitError() << "pim-tile-to-budget only handles byte-aligned"
                           " f16/f32 tiles";
        return signalPassFailure();
      }
      elemBytes = bytes;

      // MRAM is a property of the whole (untiled) operator, independent of
      // how the WRAM-side tile gets chosen below.
      if (bytesFor(*full, *bytes) > *mram) {
        dot.emitError() << "linear footprint " << bytesFor(*full, *bytes)
                        << " exceeds mram-bytes " << *mram;
        return signalPassFailure();
      }

      TileShape tile = *visible;
      bool needsRewrite = !fitsBudget(tile, *bytes, *wram, *dma);
      if (needsRewrite) {
        TileShape smallestTried = tile;
        auto found = searchTile(*visible, *bytes, *wram, *dma, &smallestTried);
        if (!found) {
          dot.emitError()
              << "no legal power-of-two tile fits: M=" << full->m
              << " N=" << full->n << " K=" << full->k << " dtype_bytes=" << *bytes
              << " wram_bytes=" << *wram << " mram_bytes=" << *mram
              << " dma_align=" << *dma << "; smallest tried was M="
              << smallestTried.m << " N=" << smallestTried.n
              << " K=" << smallestTried.k;
          return signalPassFailure();
        }
        tile = *found;
      }

      // Check tile-shape consistency, and emit diagnostics, before the
      // rewrite below erases `dot` -- nothing may reference it afterwards.
      if (!chosen)
        chosen = tile;
      else if (chosen->m != tile.m || chosen->n != tile.n ||
               chosen->k != tile.k) {
        dot.emitError() << "pim-tile-to-budget only handles one consistent"
                           " linear tile shape";
        return signalPassFailure();
      }

      if (needsRewrite) {
        LinearKernelRewriter rewriter(dot, *full, tile, lookupNumTasklets(mod),
                                      lookupNumDpus(mod));
        if (failed(rewriter.analyze()))
          return signalPassFailure();
        rewriter.rewrite();
        // `dot` (and everything under the erased root) is gone; nothing
        // else in this iteration may reference it.
      }
    }

    if (!sawDot || !chosen || !elemBytes)
      return signalPassFailure();

    Builder b(&getContext());
    mod->setAttr(AttrTileMName, b.getI64IntegerAttr(chosen->m));
    mod->setAttr(AttrTileNName, b.getI64IntegerAttr(chosen->n));
    mod->setAttr(AttrTileKName, b.getI64IntegerAttr(chosen->k));
    mod->setAttr(AttrTileWramBytesName,
                 b.getI64IntegerAttr(bytesFor(*chosen, *elemBytes)));
  }
};

} // namespace
