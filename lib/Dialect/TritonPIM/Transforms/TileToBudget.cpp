#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
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

static std::optional<int64_t> constantInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    if (auto i = dyn_cast<IntegerAttr>(cst.getValue()))
      return i.getInt();
  }
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
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
//
// Real autotuned kernels (FlagGems `linear_kernel`) don't fit this
// structural model at all: M/N are split across a 2-D launch grid rather
// than a loop (so there is no M/N `scf.for` to walk), and M/N/K are runtime
// scalar kernel arguments, not IR-visible constants -- `loopFullExtent`'s
// `constantInt` can never see them. `overrideM/N/K` (>= 0 when supplied,
// from the pass's `full-m`/`full-n`/`full-k` options, which the Python
// caller fills with the real values captured at the launch site) bypass
// structural inference for whichever dimension is supplied, so this
// function still works on such kernels; each dimension left unsupplied
// (-1) keeps falling back to the structural walk exactly as before.
static std::optional<TileShape> inferFullShape(triton::DotOp dot,
                                               int64_t overrideM,
                                               int64_t overrideN,
                                               int64_t overrideK) {
  auto visible = inferVisibleTile(dot);
  if (!visible)
    return std::nullopt;

  bool needM = overrideM < 0;
  bool needN = overrideN < 0;
  bool needK = overrideK < 0;

  int64_t m = needM ? visible->m : overrideM;
  int64_t n = needN ? visible->n : overrideN;
  int64_t k = needK ? visible->k : overrideK;

  scf::ForOp kLoop;
  if (auto barg = dyn_cast<BlockArgument>(dot.getC()))
    kLoop = dyn_cast_or_null<scf::ForOp>(barg.getOwner()->getParentOp());

  // Void (non-reduction) tiling loops on the way out, innermost first. The
  // K-reduction loop is still identified structurally (by the block-arg
  // check above) even when `needK` is false: misclassifying it as a void
  // loop would corrupt `voidLoops`' size and hence the M/N assignment below.
  SmallVector<scf::ForOp, 2> voidLoops;
  for (Operation *parent = dot->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp)
      continue;
    if (forOp == kLoop) {
      if (needK) {
        auto extent = loopFullExtent(forOp);
        if (!extent)
          return std::nullopt;
        k = *extent;
      }
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
  if (voidLoops.size() >= 1 && needN) {
    auto extent = loopFullExtent(voidLoops[0]);
    if (!extent)
      return std::nullopt;
    n = *extent;
  }
  if (voidLoops.size() == 2 && needM) {
    auto extent = loopFullExtent(voidLoops[1]);
    if (!extent)
      return std::nullopt;
    m = *extent;
  }
  return TileShape{m, n, k};
}

// 内核是否被 launch grid 切分。FlagGems 的 autotune 内核按 2 维 grid 划分 M/N，
// 一个 program 只算一块，循环不覆盖整算子；图编译器那侧的内核（`kernel_src.py`）
// 是整算子内核，不读 program id。
//
// 这个判据决定要不要重建：重建出来的内核是「一个 program 算完整张输出」，而启动
// 网格不变——对 grid 内核来说语义就变了，按「单 program 成本 × 网格」建模的成本
// 口径也随之失效。所以 grid 内核只按实测分块回写属性，不改写。此时若实测分块
// 超预算，`pim.tile-wram-bytes` 会如实大于 `pim.wram-bytes`，让超支可见。
static bool isGridPartitioned(ModuleOp mod) {
  bool found = false;
  mod.walk([&](triton::GetProgramIdOp) { found = true; });
  return found;
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
    if (!xLoad)
      return dot.emitError() << "pim-tile-to-budget rewrite expects a's tt.load"
                                 " directly feeding tt.dot";
    auto wLoad = bVal.getDefiningOp<triton::TransOp>();
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
    auto newDot = b.create<triton::DotOp>(loc, xBlk, wBlkT, acc,
                                          triton::InputPrecision::IEEE, 0u);
    // 融合折进 dot 的 activation 属性带进重建的 dot，否则链删了属性也没了。
    if (auto activation = dot->getAttr("activation"))
      newDot->setAttr("activation", activation);
    return newDot;
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
  return success();
}

// Bytes the three staged buffers occupy: x (m*k) and w (n*k) hold operands,
// out (m*n) holds the accumulator.
//
// `accumBytes` is separate from `elemBytes` because the accumulator is wider
// than the operands on the real kernel: `tt.dot` takes f16 and produces f32
// (measured on the graph compiler's `linear` -- `tensor<4x32xf16> *
// tensor<32x512xf16> -> tensor<4x512xf32>`). Charging the operand width for all
// three undercounts the output tile by exactly the ratio, which loosens both the
// WRAM and the MRAM check on every mixed-precision kernel -- the common case.
static int64_t bytesFor(TileShape tile, int64_t elemBytes, int64_t accumBytes) {
  return (tile.m * tile.k + tile.n * tile.k) * elemBytes +
         tile.m * tile.n * accumBytes;
}

// How many DPUs the module's placement splits its tensors over, or 1 when there
// is no placement (single-DPU) or the placement does not split (replicate /
// partial both keep the whole shape on every DPU).
//
// Reported back as `pim.placed-shards` so a consumer can compare the split this
// pass saw against the one the graph compiler declared. It does NOT divide the
// footprint: the shapes reaching this pass are already one DPU's share.
//
// It counts the tensor ENCODINGS, not the module placement. Reading the
// placement back would make the value a mirror of the intent: `placed-shards`
// would equal `numDpus` by construction, and the consumer's "intent vs effect"
// comparison could never fire while both the attribute's comment and the design
// describe it as what this pass actually saw. The encodings are what the passes
// below really see, so a pattern that rebuilds a tensor type and drops the split
// shows up here as 1 -- which is the drift the consumer is there to detect.
static int64_t localShardCount(ModuleOp mod) {
  int64_t width = 1;
  mod.walk([&](Operation *op) {
    SmallVector<Type> types(op->getResultTypes());
    for (Value operand : op->getOperands())
      types.push_back(operand.getType());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          types.push_back(arg.getType());
    for (Type ty : types) {
      auto tensorTy = dyn_cast<RankedTensorType>(ty);
      if (!tensorTy)
        continue;
      if (auto enc = dyn_cast_or_null<TaskletTiledEncodingAttr>(
              tensorTy.getEncoding()))
        width = std::max(width, (int64_t)enc.getTotalDpusPerDevice());
    }
  });
  return width;
}

// Extra MRAM a `partial` placement needs on each DPU, in bytes.
//
// `partial` says every DPU holds a whole-shaped *piece* of the sum, and the
// value only exists once those pieces are reduced across DPUs. The reduction
// has to land somewhere: a DPU receiving a peer's piece needs room for it
// alongside its own. One peer's piece is the whole per-DPU output (not a WRAM
// tile), and it is real occupancy -- charge it against the same per-DPU MRAM
// budget as the operands.
//
// This is the point where the `replicate` and `partial` kinds stop being a
// record and start changing a decision. The two are identical in shape (both
// keep the whole shape on every DPU) and were therefore indistinguishable to
// this pass before; they are not identical in cost, because only one of them
// owes a reduction. `shard` and `replicate` return 0: a shard's pieces are
// disjoint and a replicate's are already complete, so neither needs a landing
// buffer.
static int64_t reduceStagingBytes(ModuleOp mod, TileShape full,
                                  int64_t accumBytes) {
  auto placement = mod->getAttrOfType<PlacementSpecAttr>(AttrPlacementName);
  if (!placement || placement.getKind() != PlacementKind::Partial)
    return 0;
  if (placement.getNumDpus() < 2)
    return 0;
  // Charged at the accumulator's width, not the operands': the piece arriving
  // from a peer IS a partial sum, so it has the accumulator's format (f32 where
  // the operands are f16). Using the operand width would undercount it by the
  // same ratio `bytesFor` used to undercount the output tile.
  return full.m * full.n * accumBytes;
}

// Whether `tile` fits the WRAM budget and each of the three staged buffers
// (x, w, out) is separately DMA-aligned -- the design's per-buffer alignment
// requirement, stricter than (and a superset of) checking only the summed
// footprint.
static bool fitsBudget(TileShape tile, int64_t elemBytes, int64_t accumBytes,
                       int64_t wram, int64_t dma) {
  int64_t xBytes = tile.m * tile.k * elemBytes;
  int64_t wBytes = tile.n * tile.k * elemBytes;
  // The output tile holds the accumulator, which is f32 where the operands are
  // f16 -- see `bytesFor`.
  int64_t oBytes = tile.m * tile.n * accumBytes;
  if (xBytes + wBytes + oBytes > wram)
    return false;
  return xBytes % dma == 0 && wBytes % dma == 0 && oBytes % dma == 0;
}

// Candidate tile sizes along one dimension, largest first: every power of two
// up to `full`, then `full` itself when it is not one of them. The powers of
// two guarantee a legal tile exists once WRAM is small enough; `full` keeps the
// whole extent as a candidate so a dimension that already fits is not split.
static SmallVector<int64_t, 8> candidateTilesDesc(int64_t full) {
  SmallVector<int64_t, 8> vals;
  if (full <= 0)
    return vals;
  int64_t top = 1;
  while (top <= full / 2)
    top *= 2;
  for (int64_t v = top; v >= 1; v /= 2)
    vals.push_back(v);
  if (full > top)
    vals.push_back(full);
  return vals;
}

// Searches for a WRAM/DMA-legal tile no larger than `visible` in any
// dimension. Exhaustive over the power-of-two lattice below `visible` plus
// `visible` itself (bounded: at most log2(M)+2 * log2(N)+2 * log2(K)+2
// candidates, a few hundred at the largest realistic shapes) rather than
// greedily shrinking one dimension to its floor before trying another -- a
// greedy single path can dead-end: e.g. shrinking K first to fix a WRAM
// overage can make x's per-tile byte count drop below `dma`, permanently
// failing the alignment check for every smaller K, even though a *different*
// K paired with a smaller N would satisfy both budget and alignment. The
// candidates are the powers of two up to the visible extent plus that extent
// itself, so a dimension that is not a power of two still has the
// whole-extent option and still bottoms out at 1. Symmetric across M/N/K on
// purpose: nothing here assumes M is 1 or unsplit -- if a future shape needs
// the M dimension tiled too, this same search already finds it. Among all
// legal tiles, picks the one with the most elements (fewest loop iterations
// over the untiled extents), preferring to keep K largest, then N, then M
// when multiple tiles tie on element count -- matching the design's stated
// preference order without needing a separate tie-break pass. On failure,
// `*smallestTried` (if non-null) is left holding {1,1,1} clamped to
// `visible`, for a diagnostic.
static std::optional<TileShape>
searchTile(TileShape visible, int64_t elemBytes, int64_t accumBytes,
           int64_t wram, int64_t dma, TileShape *smallestTried = nullptr,
           llvm::function_ref<bool(TileShape)> accept = nullptr) {
  std::optional<TileShape> best;
  for (int64_t m : candidateTilesDesc(visible.m))
    for (int64_t n : candidateTilesDesc(visible.n))
      for (int64_t k : candidateTilesDesc(visible.k)) {
        TileShape tile{m, n, k};
        if (!fitsBudget(tile, elemBytes, accumBytes, wram, dma))
          continue;
        // 调用方的额外约束，B 路用它要求 tile 整除各维。
        if (accept && !accept(tile))
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


// 循环体里有没有张量运算。只有标量运算的循环拆了不减工作集，不处理。
static bool loopBodyHasTensorOp(scf::ForOp forOp) {
  bool found = false;
  forOp.getBody()->walk([&](Operation *op) {
    if (llvm::any_of(op->getOperandTypes(),
                     [](Type t) { return isa<RankedTensorType>(t); }))
      found = true;
  });
  return found;
}

// 把张量类型的形状第一维改成 `extent`，其余维度不动。
static RankedTensorType withLeadingExtent(RankedTensorType ty, int64_t extent) {
  SmallVector<int64_t> shape(ty.getShape());
  shape[0] = extent;
  return RankedTensorType::get(shape, ty.getElementType(), ty.getEncoding());
}

// 循环体按新步长重建：把「基址 + 长度为 1 的偏移」改成「基址 + 一段范围」，
// 一段覆盖 `tile` 个元素。认得出两种寻址：splat 出的偏移直接进 addptr，
// 以及真实 kernel 的行主序，即 splat 之后经过 expand_dims、muli、broadcast、
// addi 才到 addptr。认不出就返回空，调用方据此放弃这条循环。
// `reason` 非空时写入认不出的原因，供调用方给出诊断。
static FailureOr<scf::ForOp> rebuildLoopBody(scf::ForOp forOp, int64_t tile,
                                              int64_t newStep,
                                              std::string *reason = nullptr) {
  auto fail = [&](const char *why) -> FailureOr<scf::ForOp> {
    if (reason)
      *reason = why;
    return failure();
  };
  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();
  // 归纳变量有两种形态。index 型先 index_cast 成 i32 再 splat；真实 kernel
  // 的循环边界就是 i32，归纳变量直接 splat，中间没有 index_cast。
  Value scalar;
  arith::IndexCastOp cast = nullptr;
  if (iv.getType().isIndex()) {
    cast = dyn_cast_or_null<arith::IndexCastOp>(
        iv.hasOneUse() ? iv.use_begin()->getOwner() : nullptr);
    if (!cast || !cast.getResult().hasOneUse())
      return fail("induction variable is not index_cast then splat");
    scalar = cast.getResult();
  } else if (iv.getType().isInteger(32)) {
    scalar = iv;
  } else {
    return fail("induction variable is not index_cast then splat");
  }
  auto splat = dyn_cast_or_null<SplatOp>(
      scalar.hasOneUse() ? scalar.use_begin()->getOwner() : nullptr);
  if (!splat)
    return fail("induction variable is not index_cast then splat");
  auto splatTy = dyn_cast<RankedTensorType>(splat.getType());
  if (!splatTy || splatTy.getRank() < 1 || splatTy.getDimSize(0) != 1)
    return fail("splat offset is not a leading extent of 1");

  // 从 splat 往下找消费它的 addptr。中间只允许 expand_dims、muli、
  // broadcast、addi 这四种不改元素值的运算，穿过它们定位到指针运算。
  SmallVector<AddPtrOp> addptrs;
  SmallVector<Operation *> frontier{splat.getOperation()};
  DenseSet<Operation *> seen;
  bool blocked = false;
  while (!frontier.empty()) {
    Operation *op = frontier.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    for (Operation *user : op->getUsers()) {
      if (isa<AddPtrOp>(user))
        addptrs.push_back(dyn_cast<AddPtrOp>(user));
      else if (isa<ExpandDimsOp, arith::MulIOp, BroadcastOp, arith::AddIOp>(user))
        frontier.push_back(user);
      else
        blocked = true;
    }
  }
  if (blocked || addptrs.empty())
    return fail("addressing is not splat or row-major");
  // 重建只把首维从 1 改成 tile。偏移首维已大于 1 时一轮处理多行，
  // 改首维会换掉这些行指向的地址。
  for (AddPtrOp addptr : addptrs) {
    auto offTy = dyn_cast<RankedTensorType>(addptr.getOffset().getType());
    if (!offTy || offTy.getDimSize(0) != 1)
      return fail("offset leading extent is not 1");
  }

  // 重建只认「行号先乘行距、再加列偏移」。乘法落在加法之后时列偏移也被
  // 乘了，重建会把行距再乘一次，按程序顺序向下传播「已过加法」来识别。
  DenseSet<Value> afterAdd;
  for (Operation &op : body->without_terminator()) {
    if (!seen.count(&op))
      continue;
    bool fed = llvm::any_of(op.getOperands(),
                            [&](Value v) { return afterAdd.contains(v); });
    if (fed && isa<arith::MulIOp>(op))
      return fail("row stride is applied after the column offset");
    if (fed || isa<arith::AddIOp>(op))
      for (Value result : op.getResults())
        afterAdd.insert(result);
  }

  // 行主序的行距是 dense 常量，沿地址链找乘法，重建时照原样乘回去。
  // splat 直连 addptr 的形态没有行距，按 1 处理，覆盖的地址集合与原循环相同。
  DenseMap<Operation *, int64_t> mulStride;
  for (Operation *op : seen) {
    auto mul = dyn_cast<arith::MulIOp>(op);
    if (!mul)
      continue;
    Value other = nullptr;
    for (Value operand : mul->getOperands())
      if (!seen.count(operand.getDefiningOp()))
        other = operand;
    auto cst = other ? other.getDefiningOp<arith::ConstantOp>() : nullptr;
    auto dense = cst ? dyn_cast<DenseElementsAttr>(cst.getValue()) : nullptr;
    if (!dense || !dense.isSplat())
      return fail("row stride is not a constant");
    mulStride[op] = dense.getSplatValue<APInt>().getSExtValue();
  }
  // 每个 addptr 沿自己的地址链反向收集行距。重建只有一份新偏移，
  // 各链行距不同时套用任何一个都会把别的链地址改错，只能放弃。
  std::optional<int64_t> chainStride;
  for (AddPtrOp addptr : addptrs) {
    int64_t s = 1;
    SmallVector<Value> work{addptr.getOffset()};
    DenseSet<Operation *> visited;
    while (!work.empty()) {
      Operation *def = work.pop_back_val().getDefiningOp();
      if (!def || !seen.count(def) || !visited.insert(def).second)
        continue;
      if (mulStride.count(def))
        s *= mulStride.lookup(def);
      for (Value operand : def->getOperands())
        work.push_back(operand);
    }
    if (chainStride && *chainStride != s)
      return fail("address chains use different row strides");
    chainStride = s;
  }
  int64_t stride = chainStride.value_or(1);

  // 重建只把首维为 1 的结果拉长到 tile。归约、变形、控制流这类算子
  // 拉长后语义会变，只放行地址链、加载存储和逐元素运算。
  for (Operation &op : body->without_terminator()) {
    if (seen.count(&op) || &op == cast)
      continue;
    if (isa<AddPtrOp, triton::LoadOp, triton::StoreOp, SplatOp>(op))
      continue;
    if (op.getNumRegions() == 0 && op.hasTrait<OpTrait::Elementwise>())
      continue;
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto ty = dyn_cast<RankedTensorType>(cst.getType());
      if (!ty || ty.getDimSize(0) != 1)
        continue;
    }
    return fail("loop body has an op the rebuild cannot widen");
  }

  // 只有从归纳变量出发的值才随新偏移拉长。其余首维为 1 的张量若在循环内
  // 产生，拉长会改掉它的含义；scalar 的 splat 拉长后值不变，可以放行。
  // 加载存储的指针必须由归纳变量驱动，否则同一地址在一块里被读写多次。
  DenseSet<Value> driven;
  for (Operation &op : body->without_terminator()) {
    if (&op == cast)
      continue;
    bool fed = &op == splat.getOperation() ||
               llvm::any_of(op.getOperands(),
                            [&](Value v) { return driven.contains(v); });
    if (fed) {
      for (Value result : op.getResults())
        driven.insert(result);
      continue;
    }
    if (isa<SplatOp>(op))
      continue;
    bool leadingOne = llvm::any_of(op.getResultTypes(), [](Type t) {
      auto ty = dyn_cast<RankedTensorType>(t);
      return ty && ty.getDimSize(0) == 1;
    });
    if (leadingOne)
      return fail("loop body has a tensor value outside the induction-variable chain");
  }
  for (Operation &op : body->without_terminator()) {
    Value ptr;
    if (auto load = dyn_cast<triton::LoadOp>(op))
      ptr = load.getPtr();
    else if (auto store = dyn_cast<triton::StoreOp>(op))
      ptr = store.getPtr();
    if (ptr && !driven.contains(ptr))
      return fail("loop body has a tensor value outside the induction-variable chain");
  }

  // 原循环带 iter_args 时，重建出的循环没有结果，替换会触发断言。
  // 这种循环不重建，调用方保持原样。
  if (!forOp.getRegionIterArgs().empty())
    return fail("loop carries iter_args");

  // 直接建在原循环前面。重建失败时把新建的循环和步长删掉，
  // 原循环保持不动。
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  // 新步长的类型跟原循环一致：index 型用 index，真实 kernel 的 i32 循环用 i32。
  Type ivTy = forOp.getInductionVar().getType();
  Value stepVal = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(ivTy, newStep));
  auto widened = builder.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                             forOp.getUpperBound(), stepVal);
  OpBuilder inner(widened.getBody(), widened.getBody()->begin());
  // 新偏移沿第一维覆盖 tile 个元素。列偏移可能在循环外、秩比 splat 高，
  // 新偏移要跟它同秩，否则后面的 addi 两侧形状对不上。
  RankedTensorType fullTy = splatTy;
  for (AddPtrOp addptr : addptrs) {
    Value off = addptr.getOffset();
    auto offTy = dyn_cast<RankedTensorType>(off.getType());
    if (offTy && offTy.getRank() > fullTy.getRank())
      fullTy = offTy;
  }
  SmallVector<int64_t> shape(fullTy.getShape());
  shape[0] = tile;
  auto fullOff = RankedTensorType::get(shape, splatTy.getElementType(),
                                        splatTy.getEncoding());
  Value base = widened.getInductionVar();
  if (base.getType().isIndex())
    base = inner.create<arith::IndexCastOp>(loc, builder.getI32Type(), base);
  Value baseTensor = inner.create<SplatOp>(loc, fullOff, base);
  auto rangeTy = RankedTensorType::get({tile}, splatTy.getElementType(),
                                        splatTy.getEncoding());
  Value range = inner.create<MakeRangeOp>(
      loc, rangeTy, 0, static_cast<int32_t>(tile));
  Value expanded = range;
  if (fullOff.getRank() > 1) {
    SmallVector<int64_t> rowShape(fullOff.getRank(), 1);
    rowShape[0] = tile;
    auto rowTy = RankedTensorType::get(rowShape, splatTy.getElementType(),
                                        splatTy.getEncoding());
    expanded = inner.create<mlir::triton::ReshapeOp>(loc, rowTy, range);
    expanded = inner.create<BroadcastOp>(loc, fullOff, expanded);
  }
  Value offs = inner.create<arith::AddIOp>(loc, baseTensor, expanded);
  if (stride != 1) {
    Value strideVal = inner.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(fullOff, builder.getI32IntegerAttr(stride)));
    offs = inner.create<arith::MulIOp>(loc, offs, strideVal);
  }

  IRMapping mapping;
  mapping.map(splat.getResult(), offs);
  // 真实 kernel 的归纳变量是 i32，直接被 splat。新循环的归纳变量也是
  // i32，原 splat 被跳过，所以把原归纳变量映射到新的，克隆才找得到定义。
  if (!cast)
    mapping.map(iv, widened.getInductionVar());
  // 列偏移在循环外，可能隔着 broadcast 才到加法。先收集地址链上
  // 不在链内的操作数，再在新循环内广播到 tile 长。
  DenseSet<Value> replaced;
  SmallVector<Value> outside;
  DenseSet<Operation *> chain(seen.begin(), seen.end());
  for (Operation *op : seen)
    for (Value operand : op->getOperands())
      if (!chain.count(operand.getDefiningOp()) && operand != scalar)
        outside.push_back(operand);
  // 循环不变的 bias、被外提的指针 splat 也要广播到 tile 长，
  // 否则与被拉长的另一侧形状对不上。
  for (Operation &op : body->without_terminator())
    for (Value operand : op.getOperands())
      if (forOp.isDefinedOutsideOfLoop(operand))
        outside.push_back(operand);
  for (Value off : outside) {
    auto offTy = dyn_cast<RankedTensorType>(off.getType());
    if (!offTy || offTy.getDimSize(0) != 1 || mapping.contains(off))
      continue;
    auto wide = withLeadingExtent(offTy, tile);
    mapping.map(off, inner.create<BroadcastOp>(loc, wide, off));
    replaced.insert(off);
  }
  // 从 splat 的结果沿数据流正向走，记录「这一步的结果等于新偏移」的值。
  // 不能靠遍历顺序：muli 的两个操作数都在地址链上时，行距那个可能先被
  // 处理，乘法就判不成由新偏移喂入，行距会被乘第二次。
  DenseSet<Value> isOffs;
  isOffs.insert(splat.getResult());
  for (Operation &op : body->without_terminator()) {
    if (&op == cast || &op == splat || isa<scf::YieldOp>(op))
      continue;
    // 行号到行距乘法这一段已被新偏移取代。expand_dims、muli、broadcast
    // 只搬运行号，跳过；addi 还要加上列偏移，必须留下。
    bool fedByNewOffs = false;
    for (Value operand : op.getOperands())
      if (isOffs.contains(operand))
        fedByNewOffs = true;

    if (fedByNewOffs && isa<ExpandDimsOp, arith::MulIOp, BroadcastOp>(op)) {
      for (Value result : op.getResults())
        isOffs.insert(result);
      for (Value result : op.getResults())
        mapping.map(result, offs);
      continue;
    }
    // 地址链上的加法改用广播后的列偏移，原加法的结果不再被使用。
    if (fedByNewOffs && isa<arith::AddIOp>(op)) {
      Value column = nullptr;
      for (Value operand : op.getOperands())
        if (replaced.contains(operand))
          column = mapping.lookup(operand);
      if (column) {
        Value sum = inner.create<arith::AddIOp>(loc, offs, column);
        for (Value result : op.getResults())
          mapping.map(result, sum);
        continue;
      }
    }
    Operation *cloned = inner.clone(op, mapping);
    // 指针与加载结果的张量长度跟着偏移一起变。
    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults())) {
      auto oldTy = dyn_cast<RankedTensorType>(oldRes.getType());
      if (!oldTy || oldTy.getDimSize(0) != 1)
        continue;
      newRes.setType(withLeadingExtent(oldTy, tile));
    }
  }
  // 新循环的 yield 由 scf.for 的构建器放好，原循环的终止符不克隆。
  // 新循环验证不过就整体撤回，验证的报错不外泄，改由调用方给诊断。
  bool valid;
  {
    ScopedDiagnosticHandler silence(forOp.getContext(),
                                    [](Diagnostic &) { return success(); });
    valid = succeeded(verify(widened));
  }
  if (!valid) {
    widened.erase();
    stepVal.getDefiningOp()->erase();
    return fail("rebuilt loop fails verification");
  }
  return widened;
}

// 循环结果是否不随迭代次数变化：每个 iter_arg 原样被 yield 回去。
// 内存副作用不在这里判断——只改步长的路径不能有副作用，而按新步长
// 重建循环体的路径允许 load 与 store。
static bool stepInvariant(scf::ForOp forOp) {
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  return yield.getOperands() == forOp.getRegionIterArgs();
}

// 循环体里有没有内存副作用。
static bool bodyHasEffect(scf::ForOp forOp) {
  bool hasEffect = false;
  forOp.getBody()->walk([&](Operation *op) {
    if (!isMemoryEffectFree(op))
      hasEffect = true;
  });
  return hasEffect;
}

// 通用循环分块。返回 success 表示这条路径适用（含「预算内不改」），
// failure 表示没有可处理的循环，调用方回退到报错。
static LogicalResult tileGenericLoops(ModuleOp mod, int64_t wram,
                                      bool gridPartitioned) {
  SmallVector<scf::ForOp> loops;
  mod.walk([&](scf::ForOp forOp) {
    if (constantInt(forOp.getLowerBound()) &&
        constantInt(forOp.getUpperBound()) &&
        constantInt(forOp.getStep()) && loopBodyHasTensorOp(forOp))
      loops.push_back(forOp);
  });
  if (loops.empty()) {
    return failure();
  }

  // 矩阵乘已写过 pim.tile-m 时以它为准，tile 不同的循环保持原样，
  // 否则模块头的 tile-m 与 tile-n/k 描述的不是同一个算子。
  std::optional<int64_t> chosen;
  if (auto attr = mod->getAttrOfType<IntegerAttr>(AttrTileMName))
    chosen = attr.getInt();
  // 矩阵乘各自切开却没写模块头时，模块头已描述不了全部算子，循环也不切，
  // 模块头保持原状。
  bool blockedByMatmulTile =
      !chosen && mod.walk([](MatmulOp mm) {
                      return mm.getTileAttr() ? WalkResult::interrupt()
                                              : WalkResult::advance();
                    }).wasInterrupted();
  for (scf::ForOp forOp : loops) {
    int64_t lower = *constantInt(forOp.getLowerBound());
    int64_t upper = *constantInt(forOp.getUpperBound());
    int64_t step = *constantInt(forOp.getStep());
    int64_t extent = upper - lower;
    if (step <= 0 || extent <= 0)
      continue;
    int64_t trips = std::max<int64_t>(extent / step, 1);
    // 一次迭代里被加载或存储的数据张量都要同时待在 WRAM 里，所以按
    // 它们的字节数相加，与 tt.dot 路径把输入、权重、累加三块相加同一口径。
    // 只取最大的一块会少算同时驻留的其余块，切出超预算的步长。
    // 指针张量不参与：它的元素宽度不代表搬运的数据量。
    bool sawTransfer = false;
    int64_t bytes = 0;
    forOp.getBody()->walk([&](Operation *op) {
      Type data;
      if (auto load = dyn_cast<triton::LoadOp>(op))
        data = load.getType();
      else if (auto store = dyn_cast<triton::StoreOp>(op))
        data = store.getValue().getType();
      else
        return;
      sawTransfer = true;
      auto tensor = dyn_cast<RankedTensorType>(data);
      if (tensor && tensor.getElementType().isIntOrFloat())
        bytes += tensor.getNumElements() * tensor.getElementTypeBitWidth() / 8;
    });
    // 没有 load 与 store 时退回统计操作数里的数值张量。这类张量跨全部迭代
    // 共用，所以按迭代次数摊薄，覆盖只搬运 iter_arg 的空转循环。
    if (!sawTransfer)
      forOp.getBody()->walk([&](Operation *op) {
        for (Type type : op->getOperandTypes())
          if (auto tensor = dyn_cast<RankedTensorType>(type))
            if (tensor.getElementType().isIntOrFloat())
              bytes = std::max(bytes, tensor.getNumElements() *
                                          tensor.getElementTypeBitWidth() / 8);
      });
    if (bytes == 0)
      continue;
    int64_t perIter = sawTransfer ? bytes : std::max<int64_t>(bytes / trips, 1);
    // 与 tt.dot 路径同一个评估模型：「原样」是整段循环，整段搬运量装得下
    // 就不切，下面各条不改写的路径都保持静默；装不下再按单轮工作集选
    // 预算内块数最少的 tile。
    if (perIter * trips <= wram)
      continue;
    // 以下都是超预算的循环。切不了就给诊断：超预算却没切与预算内不切的
    // 产物一模一样，不给诊断两者无法区分。
    auto leave = [&](StringRef why) {
      forOp.emitRemark("pim-tile-to-budget left this loop untiled: ") << why;
    };
    if (step >= extent) {
      leave("loop runs a single iteration");
      continue;
    }
    // 结果依赖迭代次数时不改步长：只改步长不改循环体，语义会变。
    if (!stepInvariant(forOp)) {
      leave("loop carries iter_args");
      continue;
    }
    // grid 内核每个 program 只算一块，启动网格不变时重建循环会改掉语义，
    // 与 tt.dot 路径同一条约定（见 `isGridPartitioned`）。
    if (gridPartitioned) {
      leave("grid-partitioned kernel");
      continue;
    }
    // 非 2 的幂的边界只挡「只改步长」那条路径：它不重建循环体，尾块形状
    // 对不齐。重建路径按整除回退选 tile，不在这里跳过。
    bool oddExtent = extent & (extent - 1);
    if (oddExtent && forOp.getInductionVar().use_empty()) {
      leave("loop extent is not a power of two");
      continue;
    }
    int64_t maxTile = 1;
    while (maxTile * 2 <= extent && maxTile * 2 * perIter <= wram)
      maxTile *= 2;
    // 预算内最大的 tile 不整除跨度时向下回退，取第一个能整除的。
    int64_t tile = maxTile;
    while (tile > 1 && extent % (step * tile) != 0)
      tile /= 2;
    int64_t newStep = step * tile;
    // 原步长不是 1 时，连续的 tile 个位置不等于原循环这 tile 轮访问的
    // 位置，重建会换掉访问集合。
    if (step != 1) {
      forOp.emitRemark("pim-tile-to-budget left this loop untiled: ")
          << "loop step is not 1";
      continue;
    }
    // 新步长必须整除跨度。不能整除时最后一轮读过上界，而重建没有掩码
    // 也没有收尾循环。搜索过的 tile 都不整除时回退到 1，在这里报出。
    if (maxTile > 1 && tile == 1) {
      forOp.emitRemark("pim-tile-to-budget left this loop untiled: ")
          << "no searched tile divides the loop extent";
      continue;
    }
    // 超预算却没切与预算内不切的产物一模一样，必须给出诊断。单轮放得下
    // 只是两轮放不下时，原因不是单轮超预算，分开报。
    if (newStep == step) {
      forOp.emitRemark("pim-tile-to-budget left this loop untiled: ")
          << (perIter > wram
                  ? "one iteration exceeds the WRAM budget"
                  : "no tile larger than one iteration fits the WRAM budget");
      continue;
    }
    // 模块只有一个 pim.tile-m，各循环 tile 不同时属性描述不了全部循环，
    // 后面的循环保持原样。
    if (blockedByMatmulTile) {
      leave("module matmuls carry their own tiles");
      continue;
    }
    if (chosen && *chosen != tile) {
      leave("module already tiles another loop with a different tile");
      continue;
    }
    // 循环体不读归纳变量时只改步长，但有副作用就不能只改步长；
    // 读了归纳变量就把循环体按新步长重建，每轮覆盖 tile 个元素。
    // 重建不了就给出诊断：超预算却没切和预算内不切必须能区分。
    if (forOp.getInductionVar().use_empty()) {
      if (bodyHasEffect(forOp)) {
        leave("loop has side effects but does not read its induction variable");
        continue;
      }
      // 步长类型跟随归纳变量，i32 循环用 index 步长会验证失败。
      OpBuilder builder(forOp);
      Type ivTy = forOp.getInductionVar().getType();
      auto stepConst = builder.create<arith::ConstantOp>(
          forOp.getLoc(), ivTy.isIndex()
                              ? builder.getIndexAttr(newStep)
                              : builder.getIntegerAttr(ivTy, newStep));
      forOp.setStep(stepConst);
    } else {
      // 带掩码的加载和存储不能重建：指针被拉长到 tile，掩码仍是原长，
      // 两侧形状不同，tt.load 验证直接失败。
      bool masked = false;
      forOp.getBody()->walk([&](Operation *op) {
        if (auto load = dyn_cast<triton::LoadOp>(op))
          masked |= (bool)load.getMask();
        else if (auto store = dyn_cast<triton::StoreOp>(op))
          masked |= (bool)store.getMask();
      });
      if (masked) {
        forOp.emitRemark("pim-tile-to-budget left this loop untiled: ")
            << "masked load or store";
        continue;
      }
      // 重建只改顶层算子的结果类型，内层循环里的加载不会跟着拉长；
      // 删原循环时内层循环也随之删掉。含嵌套循环的保持原样。
      bool nested = false;
      forOp.getBody()->walk([&](scf::ForOp) { nested = true; });
      if (nested) {
        forOp.emitRemark("pim-tile-to-budget left this loop untiled: ")
            << "loop contains a nested scf.for";
        continue;
      }
      std::string reason;
      auto rebuilt = rebuildLoopBody(forOp, tile, newStep, &reason);
      if (failed(rebuilt)) {
        // 超预算却没切时产物与预算内的循环一模一样，必须给出诊断。
        // 设计明确不支持的形态（带 iter_args、偏移长度不是 1）也记一笔，
        // 否则它和预算内不切无法区分。
        forOp.emitRemark("pim-tile-to-budget left this loop untiled: ")
            << reason;
        continue;
      }
      forOp.replaceAllUsesWith(rebuilt->getResults());
      forOp.erase();
    }
    chosen = tile;
  }
  if (chosen) {
    Builder b(mod.getContext());
    mod->setAttr(AttrTileMName, b.getI64IntegerAttr(*chosen));
  }
  return success();
}

// B 路的整算子矩阵乘没有循环，按形状选 tile，写在算子的 `tile` 属性上，
// 循环由 `-pim-lower-to-emitc` 按它生成。评估模型与 tt.dot 路径相同；
// tile 还要整除 M、N 与组宽，组内分段累加才与不分块逐位一致。
static LogicalResult tileWholeMatmuls(ModuleOp mod, int64_t wram,
                                      int64_t mram) {
  // 输出块放两个 f32 累加器：总数与当前组，各 4 字节。
  constexpr int64_t kAccumBytes = 8;
  // 模块头只有一组 tile 属性，只有模块里每个矩阵乘都切成同一个 tile、
  // 同样字节数时才写；有一个不同（含预算内不切的）就不写，各算子照用自己的 tile。
  std::optional<TileShape> chosen;
  int64_t chosenBytes = 0;
  bool uniform = true;
  WalkResult walked = mod.walk([&](MatmulOp mm) -> WalkResult {
    auto aTy = cast<RankedTensorType>(mm.getA().getType());
    auto bTy = cast<RankedTensorType>(mm.getB().getType());
    auto bytes = inferDtypeSize(mm.getA());
    auto outBytes = inferDtypeSize(mm.getResult());
    // 二维与组宽整除由 verifier 保证。元素宽度不是整字节时评估模型算不了
    // 工作集，保持原样；模块里有矩阵乘没切，模块头就不写。
    if (aTy.getRank() != 2 || bTy.getRank() != 2 || !bytes || !outBytes) {
      mm.emitRemark("pim-tile-to-budget left this matmul untiled: ")
          << "element width is not a whole number of bytes";
      uniform = false;
      return WalkResult::advance();
    }
    int64_t m = aTy.getDimSize(0), k = aTy.getDimSize(1);
    int64_t n = bTy.getDimSize(mm.getTransposeB() ? 0 : 1);
    int64_t group = mm.getDatapath().getGroupDequantAccum()
                        ? mm.getDatapath().getGroupSize()
                        : k;
    if (group <= 0 || k % group != 0) {
      uniform = false;
      return WalkResult::advance();
    }

    int64_t footprint = (m * k + n * k) * *bytes + m * n * *outBytes;
    if (footprint > mram) {
      mm.emitError() << "matmul footprint " << footprint
                     << " exceeds mram-bytes " << mram << " (per DPU)";
      return WalkResult::interrupt();
    }
    // 整块是否要切只比 WRAM 字节数：B 路没有 DMA，对齐不满足不等于超预算。
    if (bytesFor(TileShape{m, n, k}, *bytes, kAccumBytes) <= wram) {
      uniform = false;
      return WalkResult::advance();
    }
    // B 路没有 DMA，候选也不受对齐约束，对齐参数取 1。
    auto found = searchTile(
        TileShape{m, n, group}, *bytes, kAccumBytes, wram, 1, nullptr,
        [&](TileShape t) {
          return m % t.m == 0 && n % t.n == 0 && group % t.k == 0;
        });
    // 超预算却切不了时与预算内不切产物相同，必须给出诊断。
    if (!found) {
      mm.emitRemark("pim-tile-to-budget left this matmul untiled: ")
          << "no tile divides M, N and the group within the WRAM budget";
      uniform = false;
      return WalkResult::advance();
    }
    mm.setTileAttr(DenseI64ArrayAttr::get(mod.getContext(),
                                          {found->m, found->n, found->k}));
    int64_t used = bytesFor(*found, *bytes, kAccumBytes);
    if (!chosen) {
      chosen = found;
      chosenBytes = used;
    } else if (chosen->m != found->m || chosen->n != found->n ||
               chosen->k != found->k || chosenBytes != used) {
      uniform = false;
    }
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();
  if (chosen && !uniform) {
    mod.emitRemark("pim-tile-to-budget wrote no module tile attributes: ")
        << "not every matmul in the module uses the same tile";
    return success();
  }
  if (chosen) {
    Builder b(mod.getContext());
    mod->setAttr(AttrTileMName, b.getI64IntegerAttr(chosen->m));
    mod->setAttr(AttrTileNName, b.getI64IntegerAttr(chosen->n));
    mod->setAttr(AttrTileKName, b.getI64IntegerAttr(chosen->k));
    mod->setAttr(AttrTileWramBytesName, b.getI64IntegerAttr(chosenBytes));
  }
  return success();
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
    // 分片起始对齐比模块级 dma-align 更严时，按更严的那个选分块：每块缓冲
    // 都要落在这个对齐上，否则 DMA 写出去的地址不合分片的对齐要求。
    if (auto placement = mod->getAttrOfType<PlacementSpecAttr>(AttrPlacementName))
      if (placement.getAlignBytes() > (dma ? *dma : 0))
        dma = placement.getAlignBytes();
    if (!wram || !mram || !dma) {
      mod.emitError() << "pim-tile-to-budget requires " << AttrWramBytesName
                      << ", " << AttrMramBytesName << " and "
                      << AttrDmaAlignName;
      return signalPassFailure();
    }

    // grid 切分的内核只报分块，不改写（理由见 `isGridPartitioned`）。
    bool gridPartitioned = isGridPartitioned(mod);

    bool sawDot = false;
    SmallVector<triton::DotOp> dots;
    mod.walk([&](triton::DotOp dot) { dots.push_back(dot); });
    if (dots.empty()) {
      // 没有 tt.dot 时先按形状切整算子矩阵乘，再按循环结构分块。没有可切的
      // 对象就保持原样：纯逐元素 kernel 也要能跑这个 pass。
      if (failed(tileWholeMatmuls(mod, *wram, *mram)))
        return signalPassFailure();
      (void)tileGenericLoops(mod, *wram, gridPartitioned);
      return;
    }

    std::optional<TileShape> chosen;
    std::optional<int64_t> elemBytes;
    // The accumulator's width, kept beside `elemBytes` because the two differ on
    // the real kernel (f16 operands, f32 accumulator) and the tile sizing needs
    // both.
    std::optional<int64_t> accumElemBytes;
    // Footprint of the last dot examined, already one DPU's share: the shapes
    // reaching this pass come from the execution plan's local shape, so nothing
    // is divided on the way out.
    int64_t perDpuFootprint = 0;
    // Bytes this pass set aside on each DPU to stage an incoming piece of a
    // `partial` reduction; 0 for the other two placement kinds. Reported back
    // so the graph compiler can see what the reduction actually cost -- it
    // depends on the tile, which is resolved here, so only this pass knows it.
    int64_t reduceStaging = 0;
    // Bytes per element this pass charged, from the operand type. Reported back
    // so a text-level consumer need not guess the width from a type name.
    int64_t placedElemBytes = 0;
    for (triton::DotOp dot : dots) {
      sawDot = true;
      if (failed(validateDot(dot)))
        return signalPassFailure();
      auto visible = inferVisibleTile(dot);
      auto full = inferFullShape(dot, fullM, fullN, fullK);
      if (!visible || !full) {
        dot.emitError() << "pim-tile-to-budget could not infer the linear"
                           " tile shape";
        return signalPassFailure();
      }
      auto bytes = inferDtypeSize(dot.getA());
      // The accumulator's own width. `tt.dot` on this kernel takes f16 and
      // produces f32, so the output tile is wider than the operands; charging
      // the operand width for it undercounts by that ratio. Falls back to the
      // operand width when the result type is not byte-aligned -- same shape of
      // answer this pass has always given for odd types.
      auto accumBytes = inferDtypeSize(dot.getResult());
      if (!bytes) {
        dot.emitError() << "pim-tile-to-budget only handles byte-aligned"
                           " f16/f32 tiles";
        return signalPassFailure();
      }
      // An accumulator type this pass cannot measure falls back to the operand
      // width -- the answer it gave before the two were distinguished.
      int64_t accum = accumBytes.value_or(*bytes);
      elemBytes = bytes;
      accumElemBytes = accum;

      // MRAM is a property of the whole (untiled) operator, independent of
      // how the WRAM-side tile gets chosen below.
      //
      // `*full` is already the shape of ONE DPU's share, so this footprint is
      // directly comparable to the per-DPU MRAM budget -- no division by the
      // split.
      //
      // That is a property of who writes `pim.placement`: the graph compiler,
      // and it builds the kernel from the execution plan's `local_shape`. A tp2
      // `linear` arrives as weight 32x64 when the global weight is 64x64. An
      // earlier version divided here, reasoning that `*full` was global; that
      // reasoning held only for the GeneSim path, whose `full-m/n/k` overrides
      // do come from launch-side globals -- but that path never writes a
      // placement, so the division could only ever fire where it was wrong. It
      // loosened this budget check by exactly the split factor: measured, a
      // kernel needing 7168 B per DPU passed a 5000 B budget.
      // Per-dot locals: the reported values are bound to the dot that `chosen`
      // describes, further down. Overwriting them every iteration made a
      // two-dot module report tile attributes for the first dot and a footprint
      // for the last -- two different operators in one set of attributes.
      int64_t footprint = bytesFor(*full, *bytes, accum);
      // A `partial` placement owes a cross-DPU reduction, and the incoming
      // piece needs somewhere to land. That buffer occupies the same per-DPU
      // MRAM as the operands, so it belongs in this check.
      int64_t staging = reduceStagingBytes(mod, *full, accum);
      footprint += staging;
      if (footprint > *mram) {
        dot.emitError() << "linear footprint " << footprint
                        << " exceeds mram-bytes " << *mram << " (per DPU"
                        << (staging > 0
                                ? ", including " + std::to_string(staging) +
                                      " bytes staging the partial reduction"
                                : "")
                        << ")";
        return signalPassFailure();
      }

      TileShape tile = *visible;
      bool needsRewrite =
          !gridPartitioned && !fitsBudget(tile, *bytes, accum, *wram, *dma);
      if (needsRewrite) {
        TileShape smallestTried = tile;
        auto found = searchTile(*visible, *bytes, accum, *wram, *dma, &smallestTried);
        if (!found) {
          dot.emitError()
              << "no legal tile fits: M=" << full->m
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
      if (!chosen) {
        chosen = tile;
        // The reported values travel with `chosen`, so all four attributes
        // describe the same dot.
        perDpuFootprint = footprint;
        reduceStaging = staging;
        placedElemBytes = *bytes;
      } else if (chosen->m != tile.m || chosen->n != tile.n ||
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

    if (!sawDot || !chosen || !elemBytes || !accumElemBytes)
      return signalPassFailure();

    Builder b(&getContext());
    mod->setAttr(AttrTileMName, b.getI64IntegerAttr(chosen->m));
    mod->setAttr(AttrTileNName, b.getI64IntegerAttr(chosen->n));
    mod->setAttr(AttrTileKName, b.getI64IntegerAttr(chosen->k));
    mod->setAttr(AttrTileWramBytesName,
                 b.getI64IntegerAttr(bytesFor(*chosen, *elemBytes, *accumElemBytes)));

    // The Placement dimension's return path. The graph compiler decided the
    // split; what it cannot know is what the split actually cost, because the
    // per-DPU footprint depends on the tile -- and the tile is resolved here.
    //
    // Two attributes, and the second is not redundant with the placement's own
    // `numDpus`: it is the split width this pass actually *saw* on the module.
    // A pattern that drops the placement on the way here leaves it at 1 while
    // the intent still says N, which is exactly the drift a consumer needs to be
    // able to detect rather than assume away.
    mod->setAttr(AttrPlacedShardsName,
                 b.getI64IntegerAttr(localShardCount(mod)));
    mod->setAttr(AttrPlacedMramBytesName,
                 b.getI64IntegerAttr(perDpuFootprint));
    // The return path for the `partial` kind: how many bytes the reduction
    // staging took. Written even without a placement -- a consumer reading 0
    // learns "no staging was needed", which is different from "this pass did
    // not say", and the single-DPU case is exactly where the footprint is not
    // divided by anything, so gating these on a placement would silently drop
    // the capacity check's input there.
    mod->setAttr(AttrPlacedReduceBytesName,
                 b.getI64IntegerAttr(reduceStaging));
    // The dtype dimension's return path: the element width this pass charged.
    // Only written when a tile was actually sized -- 0 would be a claim, and
    // "this pass did not reach a dot" is not the same as "elements are 0 bytes".
    if (placedElemBytes > 0)
      mod->setAttr(AttrPlacedElemBytesName,
                   b.getI64IntegerAttr(placedElemBytes));
  }
};

} // namespace
