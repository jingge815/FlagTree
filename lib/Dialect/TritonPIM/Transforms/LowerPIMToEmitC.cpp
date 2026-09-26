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
// should split work across tasklets. `pim-tile-to-budget` owns the budget
// contract for those tiles; this pass consumes the already-valid tile structure
// and lowers it into C, then applies the tasklet-level M-split described above.
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
#include "pim_silu_lut.h"

#include "triton/Dialect/TritonPIM/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>

namespace mlir::triton::pim {
#define GEN_PASS_DEF_TRITONPIMLOWERTOEMITC
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"
} // namespace mlir::triton::pim

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

namespace {

// 静态量化（`pim.quantize`）的定点移位量，对应倍率 2^-shift = 256。DQ 相 3
// 走 `pim.kantor` 并从它自己的 `shift` 属性读同一个值；静态那条 ODS 上没有
// 这个字段，所以缺省值只此一处，别在别处再写一个字面量。
static constexpr int64_t kStaticQuantizeShift = -8;

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
// The operator-level path needs three more of these: a store into an int8
// buffer rounds, and the round has to be the same one the NumPy side performs.
static float pim_absf(float f) { return f < 0.0f ? -f : f; }
static float pim_maxf(float a, float b) { return a > b ? a : b; }
static float pim_i8_to_f32(int8_t v) { return (float)v; }
// The remainder of an rms normalization. Spelled out rather than left to the
// C library so the NumPy mirror can be checked against the same three steps.
static float pim_rsqrtf(float x) { return 1.0f / sqrtf(x); }
// The four functions a `pim.lut` kind stands for, spelled out rather than
// driven by a table. `gml_bridge/phase_data.py` is the numeric truth for the
// phases that use these, and it uses the exact functions; a table here would
// be a second approximation the NumPy mirror does not share, so the two would
// disagree and neither would be authoritative.
static float pim_lut_identity(float x) { return x; }
static float pim_lut_reciprocal(float x) {
  // A group that is all zeros gives p0 = 0, and 1/0 would poison every later
  // phase with an inf. Zero is the value phase_data substitutes, and a
  // quantized result of 0 for an all-zero group is what it means.
  return x == 0.0f ? 0.0f : 1.0f / x;
}
static float pim_lut_exp(float x) { return expf(x); }
/* SiLU 走 288 B 分段线性表，与 numpy 镜像读的是同一张
   （contracts/gml_lut.py::synth_silu，定域 [-4, 4)、31 段弦线）。
   闭式公式两边都绿，但绿在参考产物并不执行的那个公式上。 */
static const float pim_silu_slope[32] = {SLOPE_PLACEHOLDER};
static const float pim_silu_intercept[32] = {INTERCEPT_PLACEHOLDER};
static float pim_lut_silu(float x) {
  /* 闭式，与 numpy 镜像同一条公式。31 段弦线在 32 层里累积后，logits
     与 torch 的闭式差到 1 以上。表仍由 synth_silu 合成并写入 GML 产物，
     只是设备求值不再读它。|x| 很大时 expf 溢出，先夹到已饱和的位置。 */
  float z = -x;
  if (z > 80.0f) z = 80.0f;
  if (z < -80.0f) z = -80.0f;
  return x / (1.0f + expf(z));
}
// Ties go to the even neighbour, as numpy's `rint` does. `lroundf` would
// round them away from zero instead, and a tie broken differently is a
// one-unit disagreement exactly at .5 -- rare, and invisible until it is not.
static int8_t pim_f32_to_i8(float f) {
  if (f >= 127.0f) return 127;
  if (f <= -128.0f) return -128;
  float r = (float)(int)f; /* truncates toward zero; |f| < 128 here */
  float d = f - r;
  if (d > 0.5f) r += 1.0f;
  else if (d < -0.5f) r -= 1.0f;
  else if ((d == 0.5f || d == -0.5f) && ((int)r % 2) != 0)
    r += d > 0.0f ? 1.0f : -1.0f;
  return (int8_t)r;
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
  return elemTy; // f32, i8
}

// Storage width in bytes, as the emitted `malloc` sizes it. Derived from the
// type rather than defaulted: an i64 index buffer sized as 4 bytes would be
// half the allocation it needs, which corrupts the heap rather than failing.
static int64_t storageBytes(Type elemTy) {
  if (elemTy.isF16())
    return 2;
  return elemTy.getIntOrFloatBitWidth() / 8;
}

static LogicalResult checkElementType(Operation *op, Type elemTy) {
  // i8 is what the operator-level path stores a quantized result in; i16 is a
  // cache row number and i32/i64 are a gather's. The tile path produces none
  // of these, so allowing them here widens nothing it could reach.
  //
  // i16 is a *value* only where the op says so: a cache address is 16 bits on
  // the target, and widening it would name rows the cache does not have.
  if (elemTy.isF32() || elemTy.isF16() || elemTy.isInteger(8) ||
      elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isInteger(64))
    return success();
  return op->emitError()
         << "pim-lower-to-emitc only handles f16/f32 elements, i8 storage and "
            "i16/i32/i64 indices, got "
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
  // emitted ahead of the kernel.
  bool needsF16Helpers = false;

  // The helpers are `static`, so they may appear once per translation unit.
  // A module holds one function per operator, so this is once per module.
  bool helpersEmitted = false;

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
    // Whole tensors in, whole tensors out: this is a kernel the graph compiler
    // emitted, not one derived from TTIR, and it has no DMA to lower.
    if (oldFunc.getNumArguments() > 0 &&
        isa<RankedTensorType>(oldFunc.getArgument(0).getType()))
      return lowerOperatorFunc(oldFunc);

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
      emitHelpers(b, loc);
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

    // A `pim` op with no lowering above is a gap, not scaffolding. This pass is
    // the only way the operator IR reaches C, so staying quiet here once meant
    // the generated C silently computed something else while the NumPy mirror
    // "agreed" -- because it was wrong the same way. Only the address
    // arithmetic the tile path is built on may pass through unremarked, and
    // that is not a `pim` op.
    if (op->getName().getDialectNamespace() ==
        TritonPIMDialect::getDialectNamespace())
      return op->emitError()
             << "pim-lower-to-emitc does not lower " << op->getName()
             << "; dropping it would leave the generated C computing a "
                "different function than the IR describes";

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

  //===--------------------------------------------------------------------===//
  // Operator-level kernels
  //===--------------------------------------------------------------------===//
  //
  // A kernel the graph compiler emits starts from whole tensors rather than
  // from addresses: `pim-expand-phases` leaves a chain of operator-level ops
  // whose last result nobody consumes, and there is no DMA anywhere, so none
  // of the tile-level machinery above applies.
  //
  // What this path adds is the write the emitter had no way to make. Every
  // tensor argument becomes a bare pointer, one output pointer is appended
  // *after* all the inputs, and the chain's unwritten result lands there --
  // the same call shape the tile-level kernels have, so a caller needs no
  // second convention.
  //
  // The arithmetic follows `gml_bridge/phase_data.py` line for line. That
  // module is the numeric truth for the dynamic-quantization and softmax
  // phases; a second, "close enough" implementation here is exactly what makes
  // a compiled kernel and its NumPy mirror disagree with no way to say which
  // one is wrong.

  // A whole tensor in this path. No strides, because every tensor here is
  // contiguous and the structure the loops need comes from the ops, not from
  // an address computation.
  struct FlatBuffer {
    Value ptr;
    int64_t count = 0;
    Type elemTy;
    bool isF16() const { return elemTy && elemTy.isF16(); }
    bool isI8() const { return elemTy && elemTy.isInteger(8); }
  };

  // `for (i = start; i < end; ++i)`.
  template <typename Body>
  void emitFor(OpBuilder &b, Location loc, int64_t start, int64_t end,
               Body body) {
    Type i32 = b.getI32Type();
    Value lo = b.create<emitc::ConstantOp>(loc, i32, b.getI32IntegerAttr(start));
    Value hi = b.create<emitc::ConstantOp>(loc, i32, b.getI32IntegerAttr(end));
    Value step = b.create<emitc::ConstantOp>(loc, i32, b.getI32IntegerAttr(1));
    b.create<emitc::ForOp>(loc, lo, hi, step,
                           [&](OpBuilder &ib, Location, Value i) {
                             body(ib, i);
                             ib.create<emitc::YieldOp>(loc);
                           });
  }

  static Value constI32(OpBuilder &b, Location loc, int64_t v) {
    return b.create<emitc::ConstantOp>(loc, b.getI32Type(),
                                       b.getI32IntegerAttr(v));
  }

  static Value constF32(OpBuilder &b, Location loc, double v) {
    return b.create<emitc::ConstantOp>(loc, b.getF32Type(),
                                       b.getF32FloatAttr(v));
  }

  // 定点通路：加 bias（可空）、乘 scale（可空）、再乘 2^-shift。
  // KV 写入与独立 `pim.fpsu_scale` 走同一条（缩放 2.0、右移 14），
  // 所以 helper 只此一处，别在两处各写一份。
  static Value applyFixedPoint(OpBuilder &b, Location loc, Value x, Value bias,
                               Value scale, int64_t shift) {
    Type f32 = b.getF32Type();
    if (bias)
      x = b.create<emitc::AddOp>(loc, f32, x, bias);
    if (scale)
      x = b.create<emitc::MulOp>(loc, f32, x, scale);
    double mul = std::ldexp(1.0, static_cast<int>(-shift));
    return b.create<emitc::MulOp>(loc, f32, x, constF32(b, loc, (float)mul));
  }

  // Reads one element as f32, converting from the buffer's storage type.
  static Value loadFlat(OpBuilder &b, Location loc, const FlatBuffer &buf,
                        Value i) {
    Type storage = storageTypeFor(b, buf.elemTy);
    Value lv = b.create<emitc::SubscriptOp>(
        loc, emitc::LValueType::get(storage), buf.ptr, ValueRange{i});
    Value raw = b.create<emitc::LoadOp>(loc, storage, lv);
    if (buf.isF16())
      return b.create<emitc::CallOpaqueOp>(loc, TypeRange{b.getF32Type()},
                                           "pim_f16_to_f32", ValueRange{raw})
          .getResult(0);
    if (buf.isI8())
      return b.create<emitc::CallOpaqueOp>(loc, TypeRange{b.getF32Type()},
                                           "pim_i8_to_f32", ValueRange{raw})
          .getResult(0);
    return raw;
  }

  // Writes one f32 element, converting to the buffer's storage type -- which
  // is where an int8 result gets its rounding and its saturation.
  static void storeFlat(OpBuilder &b, Location loc, const FlatBuffer &buf,
                        Value i, Value value) {
    Type storage = storageTypeFor(b, buf.elemTy);
    Value lv = b.create<emitc::SubscriptOp>(
        loc, emitc::LValueType::get(storage), buf.ptr, ValueRange{i});
    Value toStore = value;
    if (buf.isF16())
      toStore =
          b.create<emitc::CallOpaqueOp>(loc, TypeRange{storage},
                                        "pim_f32_to_f16", ValueRange{value})
              .getResult(0);
    else if (buf.isI8())
      toStore =
          b.create<emitc::CallOpaqueOp>(loc, TypeRange{storage},
                                        "pim_f32_to_i8", ValueRange{value})
              .getResult(0);
    b.create<emitc::AssignOp>(loc, lv, toStore);
  }

  // Apply a folded `activation` to one f32 value. The graph format keeps the
  // activation inside the producer, so this has to run here -- dropping the
  // attribute would emit the same C as an unfused op, and the NumPy mirror
  // would "agree" because it never saw the activation either.
  static LogicalResult checkFusedTail(Operation *op) {
    if (op->getAttr("fusedPool"))
      return op->emitOpError()
             << "pim-lower-to-emitc has no C for a fused pool; dropping it "
                "would emit the producer without the pool";
    auto spec = op->getAttrOfType<ActSpecAttr>("activation");
    if (!spec)
      return success();
    switch (spec.getKind()) {
    case ActivationKind::Identity:
    case ActivationKind::Reciprocal:
    case ActivationKind::Exp:
    case ActivationKind::Silu:
      return success();
    default:
      return op->emitOpError()
             << "pim-lower-to-emitc has no C for fused activation "
             << stringifyActivationKind(spec.getKind());
    }
  }

  static Value applyActivation(OpBuilder &b, Location loc, Operation *op,
                               Value value) {
    auto spec = op->getAttrOfType<ActSpecAttr>("activation");
    if (!spec)
      return value;
    StringRef fn;
    switch (spec.getKind()) {
    case ActivationKind::Identity:   fn = "pim_lut_identity"; break;
    case ActivationKind::Reciprocal: fn = "pim_lut_reciprocal"; break;
    case ActivationKind::Exp:        fn = "pim_lut_exp"; break;
    case ActivationKind::Silu:       fn = "pim_lut_silu"; break;
    default:
      llvm_unreachable("checkFusedTail already rejected this kind");
    }
    return b.create<emitc::CallOpaqueOp>(loc, TypeRange{b.getF32Type()}, fn,
                                         ValueRange{value})
        .getResult(0);
  }

  // Storage for one intermediate. Freed at the end of the kernel; only the
  // output survives, and that is the caller's buffer.
  FlatBuffer allocFlat(OpBuilder &b, Location loc, int64_t count, Type elemTy,
                       SmallVectorImpl<FlatBuffer> &temporaries) {
    auto ptrTy = emitc::PointerType::get(storageTypeFor(b, elemTy));
    Value nbytes = b.create<emitc::ConstantOp>(
        loc, b.getI64Type(),
        b.getI64IntegerAttr(count * storageBytes(elemTy)));
    Value ptr =
        b.create<emitc::CallOpaqueOp>(loc, TypeRange{ptrTy}, "malloc",
                                      ValueRange{nbytes})
            .getResult(0);
    FlatBuffer buf{ptr, count, elemTy};
    temporaries.push_back(buf);
    return buf;
  }

  // The kernel's output: the **last** result the chain leaves unwritten. The
  // emitter produces values rather than stores, so the final value has no
  // user -- and so may an earlier one: dynamic quantization's phase 1 computes
  // a scale that downstream nodes read out of band, with no SSA consumer here.
  // Which is why this is "the last" and not "the only".
  // The results the caller receives. Usually one; a split hands back one per
  // piece, and a cache write hands back none at all -- its result goes into a
  // buffer the caller owns, so inventing an output pointer would make every
  // caller pass a buffer the kernel never writes.
  //
  // These are the unwritten results of the **last** op that has any, not of
  // every such op. Dynamic quantization's phase 1 computes a scale that
  // downstream nodes read out of band, so it too has no consumer here -- and
  // returning it as well would change the kernel's signature for a value the
  // caller already gets another way. "The last", not "the only".
  static SmallVector<Value> findOperatorOutputs(triton::FuncOp func) {
    SmallVector<Value> outputs;
    func.walk([&](Operation *op) {
      SmallVector<Value> here;
      for (Value result : op->getResults())
        if (result.use_empty() && isa<RankedTensorType>(result.getType()))
          here.push_back(result);
      if (!here.empty())
        outputs = std::move(here);
    });
    return outputs;
  }

  // Scalar parameters, by SSA value. Kept apart from `buffers` because a
  // scalar is not addressable: nothing indexes into it.
  DenseMap<Value, Value> scalarValues;

  LogicalResult lowerOperatorFunc(triton::FuncOp oldFunc) {
    scalarValues.clear();
    MLIRContext *ctx = &getContext();
    OpBuilder b(ctx);
    Location loc = oldFunc.getLoc();

    SmallVector<Value> outValues = findOperatorOutputs(oldFunc);

    SmallVector<Type> newArgTypes;
    // Each buffer and each scalar remembers the parameter slot it landed in.
    // Two parallel lists zipped after the fact would lose the interleaving:
    // a step counter sits between two buffers, so the second buffer would be
    // read as the counter.
    SmallVector<std::pair<BlockArgument, unsigned>> bufferSlots;
    SmallVector<std::pair<BlockArgument, unsigned>> scalarSlots;
    for (BlockArgument arg : oldFunc.getArguments()) {
      // A tensor or a memory descriptor. The cache is the second kind: the
      // caller owns the buffer and it outlives one step, so the kernel takes a
      // pointer into it rather than a tensor of its own.
      Type elemTy;
      bool isScalar = false;
      if (auto tensorTy = dyn_cast<RankedTensorType>(arg.getType()))
        elemTy = tensorTy.getElementType();
      else if (auto descTy = dyn_cast<MemDescType>(arg.getType()))
        elemTy = descTy.getElementType();
      else if (arg.getType().isIntOrFloat()) {
        // A scalar comes in by value. The step counter a cache write takes is
        // the case: it is a number, not a buffer, and passing a pointer to it
        // would make the caller allocate somewhere to put it.
        isScalar = true;
      } else
        return oldFunc.emitError()
               << "an operator-level kernel takes whole tensors, memory "
                  "descriptors or scalars, got "
               << arg.getType();
      if (!isScalar && failed(checkElementType(oldFunc, elemTy)))
        return failure();
      // An argument nothing reads carries no information, so it does not get a
      // parameter. Dynamic quantization's scale operand is the case: on
      // hardware phase 1 produces it and phase 3 consumes it, so the
      // placeholder the emitter had to write in that position is dead once the
      // chain is expanded. Keeping it would make every caller pass a buffer
      // the kernel never touches.
      if (arg.use_empty())
        continue;
      if (isScalar) {
        if (arg.use_empty())
          continue;
        scalarSlots.emplace_back(arg, newArgTypes.size());
        newArgTypes.push_back(arg.getType());
        continue;
      }
      bufferSlots.emplace_back(arg, newArgTypes.size());
      newArgTypes.push_back(
          emitc::PointerType::get(storageTypeFor(b, elemTy)));
    }
    // The outputs go after every input, in program order, which is the
    // tile-level convention too: a caller passes inputs then outputs either
    // way.
    for (Value outValue : outValues)
      newArgTypes.push_back(emitc::PointerType::get(storageTypeFor(
          b, cast<RankedTensorType>(outValue.getType()).getElementType())));

    b.setInsertionPoint(oldFunc);
    emitHelpers(b, loc);
    auto newFunc = b.create<func::FuncOp>(
        loc, oldFunc.getName(), FunctionType::get(ctx, newArgTypes, {}));
    Block *entry = newFunc.addEntryBlock();
    b.setInsertionPointToStart(entry);

    DenseMap<Value, FlatBuffer> buffers;
    for (auto [oldArg, slot] : scalarSlots)
      scalarValues[oldArg] = newFunc.getArgument(slot);
    for (auto [oldArg, slot] : bufferSlots) {
      Value newArg = newFunc.getArgument(slot);
      // A memory descriptor is a buffer the caller allocated, no different
      // from a tensor for the purposes of this pass: a base pointer and an
      // element count. The cache is the case -- it outlives one kernel.
      if (auto descTy = dyn_cast<MemDescType>(oldArg.getType())) {
        int64_t elems = 1;
        for (int64_t dim : descTy.getShape())
          elems *= dim;
        buffers[oldArg] = FlatBuffer{newArg, elems, descTy.getElementType()};
        continue;
      }
      auto tensorTy = cast<RankedTensorType>(oldArg.getType());
      buffers[oldArg] = FlatBuffer{newArg, tensorTy.getNumElements(),
                                   tensorTy.getElementType()};
    }
    // One output parameter per received result, in the same order, so the
    // i-th follows the inputs at the i-th position from the end.
    DenseMap<Value, FlatBuffer> outs;
    for (auto [i, outValue] : llvm::enumerate(outValues)) {
      auto outTy = cast<RankedTensorType>(outValue.getType());
      outs[outValue] =
          FlatBuffer{newFunc.getArgument(newFunc.getNumArguments() -
                                         outValues.size() + i),
                     outTy.getNumElements(), outTy.getElementType()};
    }

    SmallVector<FlatBuffer, 8> temporaries;
    for (Operation &op : oldFunc.getBody().front()) {
      if (isa<triton::ReturnOp>(&op))
        continue;
      if (failed(lowerOperatorOp(&op, b, buffers, outs, temporaries)))
        return failure();
    }

    b.setInsertionPointToEnd(entry);
    for (const FlatBuffer &buf : temporaries)
      b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "free", ValueRange{buf.ptr});
    b.create<func::ReturnOp>(loc);
    oldFunc.erase();
    return success();
  }

  // The conversion helpers are `static`, so emitting them twice in one
  // translation unit is a redefinition. A module holds one function per
  // operator, so this has to be once per module, not once per function.
  void emitHelpers(OpBuilder &b, Location loc) {
    if (helpersEmitted)
      return;
    std::string helpers = llvm::StringRef(kF16Helpers).str();
    auto subst = [](std::string &text, llvm::StringRef from,
                    llvm::StringRef to) {
      size_t at = text.find(from.str());
      if (at != std::string::npos)
        text.replace(at, from.size(), to.str());
    };
    subst(helpers, "SLOPE_PLACEHOLDER", PIM_SILU_SLOPE_LIST);
    subst(helpers, "INTERCEPT_PLACEHOLDER", PIM_SILU_INTERCEPT_LIST);
    b.create<emitc::VerbatimOp>(loc, b.getStringAttr(helpers));
    helpersEmitted = true;
  }

  LogicalResult lowerOperatorOp(Operation *op, OpBuilder &b,
                                DenseMap<Value, FlatBuffer> &buffers,
                                const DenseMap<Value, FlatBuffer> &outs,
                                SmallVectorImpl<FlatBuffer> &temporaries) {
    Location loc = op->getLoc();
    Type i32 = b.getI32Type();
    Type f32 = b.getF32Type();

    // Where an op's result goes: the kernel's output parameter for the one
    // value the caller receives, a fresh buffer for everything else -- which
    // includes a result nothing consumes but that is not the output.
    //
    // A helper -- an op carrying no `#pim.phase_spec`, which by definition
    // owns no engine traversal -- keeps its value in f32 rather than at the
    // IR's element type. Softmax's `x - max` is the case that matters: it is
    // an FPSU affine folded into the exp phase, so on hardware it never round
    // trips through an f16 buffer, and `gml_bridge/phase_data.py` computes it
    // in f32 for the same reason. Rounding it here would leave this kernel one
    // f16 ulp away from the NumPy mirror on about a quarter of the elements.
    auto destination = [&](Operation *owner, Value result) -> FlatBuffer {
      if (auto it = outs.find(result); it != outs.end())
        return it->second;
      auto ty = cast<RankedTensorType>(result.getType());
      // A helper is an intermediate the phase chain threads between two
      // phases -- softmax's `x - max` is the case -- and it stays in f32 so the
      // kernel does not round where `phase_data.py` does not. Only the ops the
      // chain actually produces qualify: `pim.matmul`, `pim.convert` and a
      // static `pim.quantize` carry no phase spec either, and treating them as
      // helpers stored their i8 results as f32, skipping the rounding and the
      // saturation the mirror applies.
      bool isHelper = !owner->hasAttr("phases") &&
                      (isa<EltwiseOp>(owner) || isa<LutOp>(owner));
      Type elemTy = isHelper ? b.getF32Type() : ty.getElementType();
      FlatBuffer buf =
          allocFlat(b, loc, ty.getNumElements(), elemTy, temporaries);
      buffers[result] = buf;
      return buf;
    };

    if (auto pool = dyn_cast<GlobalPoolOp>(op)) {
      if (pool.getKind() != GlobalPoolKind::AbsMax)
        return pool.emitOpError()
               << "pim-lower-to-emitc has no C for grouped reduction "
               << stringifyGlobalPoolKind(pool.getKind());
      FlatBuffer src = buffers.lookup(pool.getSrc());
      FlatBuffer dst = destination(op, pool.getResult());
      int64_t groupSize = pool.getGroupSize();
      // Each group's absmax, then the doubling the quantization pipeline
      // applies as a leading-zero left shift. The doubling is a convention of
      // that pipeline rather than of this op, and
      // `gml_bridge/phase_data.py::dynamic_scaling` is where it is written
      // down; the emitted result has to carry it or phase 1 comes out halved.
      emitFor(b, loc, 0, dst.count, [&](OpBuilder &gb, Value g) {
        Value acc = gb.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(f32),
            emitc::OpaqueAttr::get(gb.getContext(), ""));
        gb.create<emitc::AssignOp>(loc, acc, constF32(gb, loc, 0.0));
        Value base = gb.create<emitc::MulOp>(loc, i32, g,
                                             constI32(gb, loc, groupSize));
        emitFor(gb, loc, 0, groupSize, [&](OpBuilder &jb, Value j) {
          Value v = loadFlat(jb, loc, src,
                             jb.create<emitc::AddOp>(loc, i32, base, j));
          Value a = jb.create<emitc::CallOpaqueOp>(loc, TypeRange{f32},
                                                   "pim_absf", ValueRange{v})
                        .getResult(0);
          Value cur = jb.create<emitc::LoadOp>(loc, f32, acc);
          Value m = jb.create<emitc::CallOpaqueOp>(
                            loc, TypeRange{f32}, "pim_maxf",
                            ValueRange{cur, a})
                        .getResult(0);
          jb.create<emitc::AssignOp>(loc, acc, m);
        });
        storeFlat(gb, loc, dst, g,
                  gb.create<emitc::MulOp>(
                      loc, f32, gb.create<emitc::LoadOp>(loc, f32, acc),
                      constF32(gb, loc, 2.0)));
      });
      return success();
    }

    if (auto lut = dyn_cast<LutOp>(op)) {
      // ExpandPhases fills the window triple on every lut it emits. That is
      // addressing metadata, not a table. Only an actual table operand means
      // the C would have to interpolate, which this pass does not do.
      if (lut.getTable())
        return lut.emitOpError()
               << "pim-lower-to-emitc has no C for a table-driven lut "
                  "(table / window / interpolation); the closed-form kinds "
                  "are identity / reciprocal / exp / silu";
      StringRef fn;
      switch (lut.getKind()) {
      case ActivationKind::Identity:   fn = "pim_lut_identity"; break;
      case ActivationKind::Reciprocal: fn = "pim_lut_reciprocal"; break;
      case ActivationKind::Exp:        fn = "pim_lut_exp"; break;
      case ActivationKind::Silu:       fn = "pim_lut_silu"; break;
      default:
        return lut.emitOpError()
               << "pim-lower-to-emitc has no C for activation "
               << stringifyActivationKind(lut.getKind());
      }
      FlatBuffer src = buffers.lookup(lut.getSrc());
      FlatBuffer dst = destination(op, lut.getResult());
      // The FPSU's post-table scaling factor, when the phase carries one. It
      // is a datapath constant, not a property of the table: dynamic
      // quantization's phase 1 is an identity table followed by a x1/256, so
      // without this the emitted C produced `2*absmax` where the phase's
      // declared value -- and what `gml_bridge/phase_data.py` computes -- is
      // `2*absmax/256`.
      std::optional<double> fpsuScale;
      if (auto attr = lut.getFpsuScale())
        fpsuScale = attr->convertToDouble();
      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        Value y = ib.create<emitc::CallOpaqueOp>(
                          loc, TypeRange{f32}, fn,
                          ValueRange{loadFlat(ib, loc, src, i)})
                      .getResult(0);
        if (fpsuScale)
          y = ib.create<emitc::MulOp>(loc, f32, y,
                                      constF32(ib, loc, (float)*fpsuScale));
        storeFlat(ib, loc, dst, i, y);
      });
      return success();
    }

    if (auto elt = dyn_cast<EltwiseOp>(op)) {
      // The op is variadic, but the loop below reads slots 0 and 1 only. A
      // third slot would be dropped on the floor: the emitted C would combine
      // two of the three inputs and look entirely well-formed doing it.
      if (elt.getOperands().size() != 2)
        return elt.emitOpError()
               << "pim-lower-to-emitc lowers the two-slot form; this op has "
               << elt.getOperands().size()
               << " slots, and ignoring the rest would emit C that combines "
                  "only the first two";
      // 逐槽定标目前没有降级。静默忽略会让两槽按同一定标算，数字不同且
      // 无任何报错——与「禁止静默丢 op」同一类问题，发生在属性粒度。
      if (elt.getPerSlotDatapath())
        return elt.emitOpError()
               << "pim-lower-to-emitc does not apply per-slot datapaths yet, "
                  "and ignoring them would scale both slots the same way";

      if (failed(checkFusedTail(op)))
        return failure();
      FlatBuffer lhs = buffers.lookup(elt.getLhs());
      FlatBuffer rhs = buffers.lookup(elt.getRhs());
      FlatBuffer dst = destination(op, elt.getResult());
      if (dst.count != lhs.count)
        return elt.emitOpError()
               << "pim-lower-to-emitc expects the result to match lhs's "
                  "element count";

      // How a smaller rhs lines up with the lhs. Two shapes occur, and they are
      // not the same remap:
      //
      //   - a trailing-1 axis (softmax's `x - max`, `[R, N] op [R, 1]`): the
      //     rhs index is the lhs index divided by the row width;
      //   - a leading-1 axis (RoPE's cos/sin, `[1, H, S, D] op [1, 1, S, D]`):
      //     the rhs is one row shared by every head, so the rhs index is the
      //     lhs index *modulo* the rhs element count.
      //
      // `strideDivide` distinguishes them. Anything else is a broadcast this
      // pass has not been asked to express -- and following a guess reads the
      // wrong element with nothing to report it.
      int64_t rhsTail = 1;
      bool strideDivide = true;
      if (rhs.count != 1 && rhs.count != lhs.count) {
        auto lhsTy = cast<RankedTensorType>(elt.getLhs().getType());
        auto rhsTy = cast<RankedTensorType>(elt.getRhs().getType());
        if (lhsTy.getRank() != rhsTy.getRank())
          return elt.emitOpError()
                 << "pim-lower-to-emitc broadcasts only between equal ranks";

        // Where the shapes first disagree. Everything after that point has to
        // match exactly, and every stretched axis has to be 1 on the rhs --
        // then the rhs is one contiguous block repeated over the leading axes,
        // which is what wrapping by its element count expresses.
        //
        // RoPE is `[1, H, S, D]` against `[1, 1, S, D]`: the stretched axis is
        // the head axis, not the outermost one, so comparing "all but the first"
        // would still straddle the mismatch.
        unsigned rank = lhsTy.getRank();
        unsigned firstStretched = rank;
        for (unsigned d = 0; d < rank; ++d) {
          if (lhsTy.getDimSize(d) != rhsTy.getDimSize(d)) {
            firstStretched = d;
            break;
          }
        }
        bool tailMatches =
            firstStretched < rank &&
            lhsTy.getShape().drop_front(firstStretched + 1) ==
                rhsTy.getShape().drop_front(firstStretched + 1);
        bool onlyOnesStretched = true;
        for (unsigned d = 0; d <= firstStretched && d < rank; ++d)
          if (lhsTy.getDimSize(d) != rhsTy.getDimSize(d) &&
              rhsTy.getDimSize(d) != 1)
            onlyOnesStretched = false;

        if (rhsTy.getShape().back() == 1) {
          rhsTail = lhsTy.getShape().back();
        } else if (lhs.count % rhs.count == 0 && tailMatches &&
                   onlyOnesStretched) {
          strideDivide = false;
        } else {
          return elt.emitOpError()
                 << "pim-lower-to-emitc only broadcasts a scalar, a trailing-1 "
                    "rhs, or a leading axis whose trailing shape matches";
        }
      }

      EltwiseKind kind = elt.getKind();
      if (kind != EltwiseKind::Add && kind != EltwiseKind::Sub &&
          kind != EltwiseKind::Mul)
        return elt.emitOpError()
               << "pim-lower-to-emitc has no C for elementwise "
               << stringifyEltwiseKind(kind);

      // RoPE's sin term multiplies the operand's *rotated* halves:
      // `cat(-back, front)`. Reading element i therefore means reading the
      // element half a row away, negated for the half that moves forward.
      //
      // Expressed as an index remap rather than a materialized rotation: there
      // is no negate op in the reference implementation's chain, and building
      // one here would add a buffer the hardware does not have.
      int64_t rotateWidth = 0;
      if (elt.getRotateHalf()) {
        auto lhsTy = cast<RankedTensorType>(elt.getLhs().getType());
        rotateWidth = lhsTy.getShape().back();
        if (rotateWidth % 2 != 0)
          return elt.emitOpError()
                 << "rotateHalf needs an even last dimension to have halves; "
                    "got "
                 << rotateWidth;
        if (kind != EltwiseKind::Mul)
          return elt.emitOpError()
                 << "rotateHalf is the sin term's multiply; it has no meaning "
                    "on "
                 << stringifyEltwiseKind(kind);
      }
      int64_t half = rotateWidth / 2;

      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        Value a;
        if (!rotateWidth) {
          a = loadFlat(ib, loc, lhs, i);
        } else {
          // Which half of the row this element sits in decides both where it
          // reads from and whether it is negated.
          Value width = constI32(ib, loc, rotateWidth);
          Value rowBase = ib.create<emitc::MulOp>(
              loc, i32, ib.create<emitc::DivOp>(loc, i32, i, width), width);
          Value col = ib.create<emitc::RemOp>(loc, i32, i, width);
          Value halfC = constI32(ib, loc, half);
          // `col < half` reads the back half negated; otherwise the front half
          // as is. Emitted as a ternary rather than a branch so the loop body
          // stays a single expression, which is what the rest of this file does.
          Value inFront = ib.create<emitc::CmpOp>(
              loc, ib.getI1Type(), emitc::CmpPredicate::lt, col, halfC);
          Value backIdx = ib.create<emitc::AddOp>(
              loc, i32, rowBase,
              ib.create<emitc::AddOp>(loc, i32, col, halfC));
          Value frontIdx = ib.create<emitc::AddOp>(
              loc, i32, rowBase,
              ib.create<emitc::SubOp>(loc, i32, col, halfC));
          Value srcIdx = ib.create<emitc::ConditionalOp>(loc, i32, inFront,
                                                         backIdx, frontIdx);
          Value raw = loadFlat(ib, loc, lhs, srcIdx);
          Value negated = ib.create<emitc::UnaryMinusOp>(loc, f32, raw);
          a = ib.create<emitc::ConditionalOp>(loc, f32, inFront, negated, raw);
        }
        Value ridx = i;
        if (rhs.count == 1)
          ridx = constI32(ib, loc, 0);
        else if (!strideDivide)
          // A leading axis is stretched: the rhs repeats whole, so wrap.
          ridx = ib.create<emitc::RemOp>(loc, i32, i,
                                         constI32(ib, loc, rhs.count));
        else if (rhsTail != 1)
          ridx = ib.create<emitc::DivOp>(loc, i32, i,
                                         constI32(ib, loc, rhsTail));
        Value c = loadFlat(ib, loc, rhs, ridx);
        Value y;
        if (kind == EltwiseKind::Add)
          y = ib.create<emitc::AddOp>(loc, f32, a, c);
        else if (kind == EltwiseKind::Sub)
          y = ib.create<emitc::SubOp>(loc, f32, a, c);
        else
          y = ib.create<emitc::MulOp>(loc, f32, a, c);
        storeFlat(ib, loc, dst, i, applyActivation(ib, loc, op, y));
      });
      return success();
    }

    if (auto red = dyn_cast<ReduceAxisOp>(op)) {
      auto srcTy = cast<RankedTensorType>(red.getSrc().getType());
      int64_t axis = red.getAxis();
      if (axis < 0 || axis >= srcTy.getRank())
        return red.emitOpError() << "axis out of range";
      bool isMax = red.getKind() == EltwiseKind::Max;
      bool isAdd = red.getKind() == EltwiseKind::Add;
      if (!isMax && !isAdd)
        return red.emitOpError()
               << "pim-lower-to-emitc has no C for reduction "
               << stringifyEltwiseKind(red.getKind());

      FlatBuffer src = buffers.lookup(red.getSrc());
      FlatBuffer dst = destination(op, red.getResult());

      // One reduction per (outer, inner) pair, striding by the reduced axis.
      int64_t extent = srcTy.getDimSize(axis);
      int64_t inner = 1, outer = 1;
      for (int64_t d = axis + 1; d < srcTy.getRank(); ++d)
        inner *= srcTy.getDimSize(d);
      for (int64_t d = 0; d < axis; ++d)
        outer *= srcTy.getDimSize(d);

      emitFor(b, loc, 0, outer, [&](OpBuilder &ob, Value o) {
        emitFor(ob, loc, 0, inner, [&](OpBuilder &ib, Value in) {
          Value acc = ib.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(f32),
              emitc::OpaqueAttr::get(ib.getContext(), ""));
          // The identity of the reduction: adding nothing, or the lowest
          // finite float for a max.
          ib.create<emitc::AssignOp>(
              loc, acc,
              constF32(ib, loc, isMax ? -3.4028234663852886e+38 : 0.0));
          emitFor(ib, loc, 0, extent, [&](OpBuilder &kb, Value k) {
            Value row = kb.create<emitc::AddOp>(
                loc, i32,
                kb.create<emitc::MulOp>(loc, i32, o,
                                        constI32(kb, loc, extent)),
                k);
            Value idx = kb.create<emitc::AddOp>(
                loc, i32,
                kb.create<emitc::MulOp>(loc, i32, row,
                                        constI32(kb, loc, inner)),
                in);
            Value v = loadFlat(kb, loc, src, idx);
            Value cur = kb.create<emitc::LoadOp>(loc, f32, acc);
            Value next = isMax
                             ? kb.create<emitc::CallOpaqueOp>(
                                      loc, TypeRange{f32}, "pim_maxf",
                                      ValueRange{cur, v})
                                   .getResult(0)
                             : kb.create<emitc::AddOp>(loc, f32, cur, v);
            kb.create<emitc::AssignOp>(loc, acc, next);
          });
          Value oidx = ib.create<emitc::AddOp>(
              loc, i32,
              ib.create<emitc::MulOp>(loc, i32, o, constI32(ib, loc, inner)),
              in);
          storeFlat(ib, loc, dst, oidx, ib.create<emitc::LoadOp>(loc, f32, acc));
        });
      });
      return success();
    }

    if (auto gather = dyn_cast<::mlir::triton::pim::GatherOp>(op)) {
      FlatBuffer table = buffers.lookup(gather.getTable());
      FlatBuffer indices = buffers.lookup(gather.getIndices());
      FlatBuffer dst = destination(op, gather.getResult());

      // One row copied per index. The row width comes from the table rather
      // than from the result so that a wrong result shape cannot quietly
      // change the stride -- the verifier already ties the two together.
      auto tableTy = cast<RankedTensorType>(gather.getTable().getType());
      int64_t rowWidth = 1;
      for (int64_t dim : tableTy.getShape().drop_front())
        rowWidth *= dim;
      if (rowWidth <= 0)
        return gather.emitOpError("the table's row width must be positive");

      emitFor(b, loc, 0, indices.count, [&](OpBuilder &ib, Value i) {
        // `loadFlat` converts to f32, which is the wrong domain for an index:
        // a large row number would lose precision. Read the integer directly.
        Type storage = storageTypeFor(ib, indices.elemTy);
        Value lv = ib.create<emitc::SubscriptOp>(
            loc, emitc::LValueType::get(storage), indices.ptr, ValueRange{i});
        Value row = ib.create<emitc::LoadOp>(loc, storage, lv);
        Value rowI32 = storage == i32
                           ? row
                           : ib.create<emitc::CastOp>(loc, i32, row).getResult();
        Value srcBase = ib.create<emitc::MulOp>(loc, i32, rowI32,
                                                constI32(ib, loc, rowWidth));
        Value dstBase = ib.create<emitc::MulOp>(loc, i32, i,
                                                constI32(ib, loc, rowWidth));
        emitFor(ib, loc, 0, rowWidth, [&](OpBuilder &jb, Value j) {
          Value v = loadFlat(jb, loc, table,
                             jb.create<emitc::AddOp>(loc, i32, srcBase, j));
          storeFlat(jb, loc, dst,
                    jb.create<emitc::AddOp>(loc, i32, dstBase, j), v);
        });
      });
      return success();
    }

    if (auto mm = dyn_cast<MatmulOp>(op)) {
      auto aTy = cast<RankedTensorType>(mm.getA().getType());
      auto bTy = cast<RankedTensorType>(mm.getB().getType());
      // `pim-expand-phases` does not reshape this op; the graph compiler sends
      // it already flattened to the two dimensions the matrix unit multiplies.
      if (aTy.getRank() != 2 || bTy.getRank() != 2)
        return mm.emitOpError()
               << "the whole-operator matrix multiply is two dimensional";
      if (mm.getBias())
        return mm.emitOpError()
               << "pim-lower-to-emitc has no C for the bias operand";

      int64_t m = aTy.getDimSize(0), k = aTy.getDimSize(1);
      int64_t n = bTy.getDimSize(mm.getTransposeB() ? 0 : 1);

      // The integer accumulate folds a group at a time into the float total,
      // dequantizing each group on the way in. Accumulating the whole row in
      // integers and dequantizing once at the end is a **different number**:
      // two groups carry two scales, and one integer accumulator cannot hold
      // both. That is why the group loop has to exist here rather than being a
      // restatement of the k loop, and why the emitted C is checked for it.
      //
      // The group's scale comes in as an operand, `[K/groupSize, N]`: one factor
      // per (group, column). It cannot be a scalar -- a scalar applied at each
      // boundary equals the same scalar applied once at the end of the row, so
      // grouped and ungrouped accumulation would emit identical numbers and the
      // rule this datapath states could not fail a test.
      bool grouped = mm.getDatapath().getGroupDequantAccum();
      int64_t groupSize = grouped ? mm.getDatapath().getGroupSize() : k;
      if (groupSize <= 0)
        return mm.emitOpError() << "the group size must be positive";
      if (k % groupSize != 0)
        return mm.emitOpError()
               << "the K dimension " << k << " is not a whole number of "
               << groupSize << "-wide groups, so the last group would be short";
      // Ungrouped: one group spanning all of K, dequantized by the binding's
      // guard factor (a storage trick, not a per-group quantity).
      double guard =
          mm.getWeightBinding() ? mm.getWeightBinding()->getSfMultiplier() : 1.0;

      FlatBuffer lhs = buffers.lookup(mm.getA());
      FlatBuffer rhs = buffers.lookup(mm.getB());
      FlatBuffer scales;
      if (grouped)
        scales = buffers.lookup(mm.getWeightScales());
      FlatBuffer dst = destination(op, mm.getResult());
      bool transposeB = mm.getTransposeB();
      if (failed(checkFusedTail(op)))
        return failure();

      emitFor(b, loc, 0, m, [&](OpBuilder &mb, Value mi) {
        emitFor(mb, loc, 0, n, [&](OpBuilder &nb, Value ni) {
          Value total = nb.create<emitc::VariableOp>(
              loc, emitc::LValueType::get(f32),
              emitc::OpaqueAttr::get(nb.getContext(), ""));
          nb.create<emitc::AssignOp>(loc, total, constF32(nb, loc, 0.0));
          emitFor(nb, loc, 0, k / groupSize, [&](OpBuilder &gb, Value gi) {
            Value group = gb.create<emitc::VariableOp>(
                loc, emitc::LValueType::get(f32),
                emitc::OpaqueAttr::get(gb.getContext(), ""));
            gb.create<emitc::AssignOp>(loc, group, constF32(gb, loc, 0.0));
            emitFor(gb, loc, 0, groupSize, [&](OpBuilder &kb, Value ki) {
              Value kk = kb.create<emitc::AddOp>(
                  loc, i32,
                  kb.create<emitc::MulOp>(loc, i32, gi,
                                          constI32(kb, loc, groupSize)),
                  ki);
              Value ai = kb.create<emitc::AddOp>(
                  loc, i32,
                  kb.create<emitc::MulOp>(loc, i32, mi, constI32(kb, loc, k)),
                  kk);
              Value bj = transposeB
                             ? kb.create<emitc::AddOp>(
                                      loc, i32,
                                      kb.create<emitc::MulOp>(
                                          loc, i32, ni, constI32(kb, loc, k)),
                                      kk)
                             : kb.create<emitc::AddOp>(
                                      loc, i32,
                                      kb.create<emitc::MulOp>(
                                          loc, i32, kk, constI32(kb, loc, n)),
                                      ni);
              Value prod = kb.create<emitc::MulOp>(
                  loc, f32, loadFlat(kb, loc, lhs, ai),
                  loadFlat(kb, loc, rhs, bj));
              Value cur = kb.create<emitc::LoadOp>(loc, f32, group);
              kb.create<emitc::AssignOp>(loc, group,
                                         kb.create<emitc::AddOp>(loc, f32, cur,
                                                                 prod));
            });
            // The group boundary: dequantize what this group accumulated, then
            // fold it into the running total. An ungrouped multiply collapses
            // to one group of the whole K, which is the identity of the loop
            // nest rather than a second code path.
            Value factor;
            if (grouped) {
              // `scales[gi * n + ni]` -- this group's factor for this column.
              Value si = gb.create<emitc::AddOp>(
                  loc, i32,
                  gb.create<emitc::MulOp>(loc, i32, gi, constI32(gb, loc, n)),
                  ni);
              factor = loadFlat(gb, loc, scales, si);
            } else {
              factor = constF32(gb, loc, guard);
            }
            Value scaled = gb.create<emitc::MulOp>(
                loc, f32, gb.create<emitc::LoadOp>(loc, f32, group), factor);
            Value run = gb.create<emitc::LoadOp>(loc, f32, total);
            gb.create<emitc::AssignOp>(loc, total,
                                       gb.create<emitc::AddOp>(loc, f32, run,
                                                               scaled));
          });
          storeFlat(nb, loc, dst,
                    nb.create<emitc::AddOp>(
                        loc, i32,
                        nb.create<emitc::MulOp>(loc, i32, mi, constI32(nb, loc, n)),
                        ni),
                    applyActivation(nb, loc, op,
                                    nb.create<emitc::LoadOp>(loc, f32, total)));
        });
      });
      return success();
    }

    // `pim.kantor` 的定点化（方案 §5.3：DQ 相 3 走独立 op，卡值在 cardValue
    // 上）。与下面 QuantizeOp 那条是同一个函数：`q = clamp(round(x*inv))`
    // 再乘 2^-shift。倍率取自 op 自己的 `shift` 属性，不写死——写死的话改
    // shift 不会改变任何产物，这个字段就成了装饰（verifier 也不让它缺）。
    if (auto kantor = dyn_cast<KantorOp>(op)) {
      auto spec = kantor.getSpec();
      if (spec.getMode() != KantorMode::Fp2IntConverter)
        return kantor.emitOpError()
               << "pim-lower-to-emitc only has C for the float-to-fixed "
                  "conversion; other kantor modes have no kernel";
      if (!kantor.getScale())
        return kantor.emitOpError()
               << "the float-to-fixed conversion needs the reciprocal scale "
                  "operand; without it there is nothing to multiply by";
      auto shiftAttr = kantor.getShift();
      if (!shiftAttr)
        return kantor.emitOpError()
               << "the float-to-fixed conversion needs the shift attribute; "
                  "the scaling is undefined without it";
      // i8 是有符号的移位量（左移为负），取出来要带符号扩展。
      int64_t shift = static_cast<int8_t>(*shiftAttr);
      FlatBuffer src = buffers.lookup(kantor.getLhs());
      FlatBuffer scale = buffers.lookup(kantor.getScale());
      FlatBuffer dst = destination(op, kantor.getResult());
      // 定标可以是每元素、每张量或**逐组**。逐组是 DQ 相 3 的形态：
      // 张量 [1x4096] 配 [32] 个组定标，组宽由两者的元素数之比定。
      if (src.count % scale.count != 0)
        return kantor.emitOpError()
               << "the reciprocal scale element count must divide the source's";
      int64_t groupSize = src.count / scale.count;
      // 偏置是可选操作数，`KantorOp::verify` 放行它，降级这里却从来没用过
      // ——合法 IR 的一个操作数静默蒸发。与 `applyFixedPoint` 同一个顺序：
      // 先加偏置再乘定标。
      std::optional<FlatBuffer> bias;
      if (kantor.getBias()) {
        bias = buffers.lookup(kantor.getBias());
        if (bias->count != 1 && bias->count != scale.count)
          return kantor.emitOpError()
                 << "the bias must be a scalar or match the scale's element "
                    "count; got "
                 << bias->count << " against " << scale.count;
      }
      // `q = round((x + bias[g]) * inv[g] * 2^-shift)`。shift 是负的（左移），
      // 所以这里算出来的倍率大于 1：-8 → 256。
      double mul = std::ldexp(1.0, static_cast<int>(-shift));
      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        Value factor = groupSize == 1
                           ? loadFlat(ib, loc, scale, i)
                           : loadFlat(ib, loc, scale,
                                      ib.create<emitc::DivOp>(
                                          loc, i32, i,
                                          constI32(ib, loc, groupSize)));
        Value x = loadFlat(ib, loc, src, i);
        if (bias)
          x = ib.create<emitc::AddOp>(
              loc, f32, x,
              bias->count == 1 ? loadFlat(ib, loc, *bias, constI32(ib, loc, 0))
                               : loadFlat(ib, loc, *bias, i));
        Value p = ib.create<emitc::MulOp>(loc, f32, x, factor);
        storeFlat(ib, loc, dst, i,
                  ib.create<emitc::MulOp>(loc, f32, p,
                                          constF32(ib, loc, (float)mul)));
      });
      return success();
    }

    if (auto quant = dyn_cast<QuantizeOp>(op)) {
      if (quant.getDynamic())
        return quant.emitOpError()
               << "an unexpanded dynamic quantize reached pim-lower-to-emitc; "
                  "run -pim-expand-phases first";
      int64_t groupSize = quant.getSpec().getGroupSize();
      if (groupSize <= 0)
        return quant.emitOpError()
               << "the per-group scale needs a positive group size";
      FlatBuffer src = buffers.lookup(quant.getSrc());
      FlatBuffer scale = buffers.lookup(quant.getScale());
      FlatBuffer dst = destination(op, quant.getResult());
      // 组号必须落在定标缓冲里。少了这一条，`groupSize` 比真实的组宽小时
      // 会读出缓冲外的值，再乘遍整张张量——越界读，没有诊断。隔壁的
      // `pim.kantor` 一直是查的。
      if (scale.count == 0 || src.count % scale.count != 0)
        return quant.emitOpError()
               << "the scale element count must divide the source's; got "
               << scale.count << " against " << src.count;
      // `q = clamp(round(x * inv[g]))`，倍率取 2^-shift。静态量化没有
      // `shift` 属性（那是 `pim.kantor` 的字段），用同一支缺省值——两处
      // 写死的数字必须是同一个，否则同一个定点通路会有两个倍率。
      double mul = std::ldexp(1.0, static_cast<int>(-kStaticQuantizeShift));
      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        Value g = ib.create<emitc::DivOp>(loc, i32, i,
                                          constI32(ib, loc, groupSize));
        Value p = ib.create<emitc::MulOp>(loc, f32, loadFlat(ib, loc, src, i),
                                          loadFlat(ib, loc, scale, g));
        storeFlat(ib, loc, dst, i,
                  ib.create<emitc::MulOp>(loc, f32, p,
                                          constF32(ib, loc, (float)mul)));
      });
      return success();
    }

    // Two flat buffers with the same element count may still disagree about
    // which element of `rhs` an index of `lhs` reads: a scalar repeats, a
    // trailing-1 axis divides, and a stretched leading axis wraps. All three
    // appear in one network, so the choice is made rather than assumed.
    if (auto norm = dyn_cast<NormalizeOp>(op)) {
      if (!norm.getRmsNorm())
        return norm.emitOpError()
               << "pim-lower-to-emitc only has C for rms normalization; a "
                  "layer normalization subtracts a mean, which is a different "
                  "number";
      if (norm.getBias())
        return norm.emitOpError()
               << "rms normalization has no bias to add: it does not subtract "
                  "a mean, so there is nothing to re-center";

      auto srcTy = cast<RankedTensorType>(norm.getSrc().getType());
      auto dstTy = cast<RankedTensorType>(norm.getResult().getType());
      int64_t axis = norm.getAxis();
      if (axis < 0)
        axis += srcTy.getRank();
      if (axis != srcTy.getRank() - 1)
        return norm.emitOpError()
               << "pim-lower-to-emitc normalizes the last axis; got axis "
               << norm.getAxis();

      int64_t width = srcTy.getDimSize(srcTy.getRank() - 1);
      int64_t rows = srcTy.getNumElements() / width;

      FlatBuffer src = buffers.lookup(norm.getSrc());
      FlatBuffer dst = destination(op, norm.getResult());
      FlatBuffer gamma;
      if (norm.getWeight())
        gamma = buffers.lookup(norm.getWeight());

      // Epsilon is an operand because the target reads it from a buffer of
      // its own. Inventing 1e-5 here would agree with a NumPy default of the
      // same number and hide a missing operand -- llama2 happens to use
      // 1e-5, so the two sides would match for the wrong reason.
      if (!norm.getEpsilon())
        return norm.emitOpError()
               << "pim.normalize needs an epsilon operand; inventing 1e-5 "
                  "would hide a missing buffer";
      FlatBuffer e = buffers.lookup(norm.getEpsilon());
      if (e.count != 1)
        return norm.emitOpError()
               << "epsilon must hold a single value, got " << e.count;
      Value eps = loadFlat(b, loc, e, constI32(b, loc, 0));

      emitFor(b, loc, 0, rows, [&](OpBuilder &rb, Value r) {
        Value sum = rb.create<emitc::VariableOp>(
            loc, emitc::LValueType::get(f32),
            emitc::OpaqueAttr::get(rb.getContext(), ""));
        rb.create<emitc::AssignOp>(loc, sum, constF32(rb, loc, 0.0));
        Value base = rb.create<emitc::MulOp>(loc, i32, r, constI32(rb, loc, width));
        emitFor(rb, loc, 0, width, [&](OpBuilder &cb, Value c) {
          Value x = loadFlat(cb, loc, src,
                             cb.create<emitc::AddOp>(loc, i32, base, c));
          Value cur = cb.create<emitc::LoadOp>(loc, f32, sum);
          cb.create<emitc::AssignOp>(
              loc, sum,
              cb.create<emitc::AddOp>(loc, f32, cur,
                                      cb.create<emitc::MulOp>(loc, f32, x, x)));
        });
        Value mean = rb.create<emitc::DivOp>(
            loc, f32, rb.create<emitc::LoadOp>(loc, f32, sum),
            constF32(rb, loc, static_cast<double>(width)));
        Value shifted = rb.create<emitc::AddOp>(loc, f32, mean, eps);
        Value inv = rb.create<emitc::CallOpaqueOp>(
                        loc, TypeRange{f32}, "pim_rsqrtf", ValueRange{shifted})
                        .getResult(0);
        emitFor(rb, loc, 0, width, [&](OpBuilder &cb, Value c) {
          Value idx = cb.create<emitc::AddOp>(loc, i32, base, c);
          Value x = loadFlat(cb, loc, src, idx);
          Value scale = inv;
          if (gamma.count) {
            // Gamma is per channel, so it follows the last axis. A per-tensor
            // gamma holds one value and repeats.
            Value gi = gamma.count == 1 ? constI32(cb, loc, 0) : c;
            scale = cb.create<emitc::MulOp>(loc, f32, inv,
                                            loadFlat(cb, loc, gamma, gi));
          }
          storeFlat(cb, loc, dst, idx,
                    cb.create<emitc::MulOp>(loc, f32, x, scale));
        });
      });
      return success();
    }

    if (auto mask = dyn_cast<MaskOp>(op)) {
      auto scoresTy = cast<RankedTensorType>(mask.getScores().getType());
      auto maskTy = cast<RankedTensorType>(mask.getMask().getType());
      if (mask.getTransposeInput())
        return mask.emitOpError()
               << "pim-lower-to-emitc has no C for a transposed mask input";
      // The op's own verifier allows a mask of lower rank: it broadcasts
      // against the scores' trailing dimensions. Requiring equal ranks here
      // made legal IR fail to lower.
      unsigned rank = scoresTy.getRank();
      unsigned mrank = maskTy.getRank();
      if (mrank > rank)
        return mask.emitOpError()
               << "the mask cannot have a higher rank than the scores; got "
               << mrank << " against " << rank;
      int64_t dimOffset = rank - mrank;

      // A mask is an additive bias in the floating domain, not a boolean. The
      // masked-out positions carry negative infinity rather than zero, so the
      // addition -- not a select -- is what applies it.
      FlatBuffer scores = buffers.lookup(mask.getScores());
      FlatBuffer m = buffers.lookup(mask.getMask());
      FlatBuffer dst = destination(op, mask.getResult());

      // 秩是 `unsigned`，`rank - 2` 在秩为 0/1 时会回绕成一个巨大的正数，
      // 循环下标随即越界。先转成有符号再减。
      SmallVector<int64_t> strides(rank, 1);
      for (int64_t d = static_cast<int64_t>(rank) - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * scoresTy.getDimSize(d + 1);
      SmallVector<int64_t> maskStrides(mrank, 1);
      for (int64_t d = static_cast<int64_t>(mrank) - 2; d >= 0; --d)
        maskStrides[d] = maskStrides[d + 1] * maskTy.getDimSize(d + 1);

      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        Value midx = constI32(ib, loc, 0);
        Value remaining = i;
        for (unsigned d = 0; d < rank; ++d) {
          Value dim = constI32(ib, loc, strides[d]);
          Value coord = ib.create<emitc::DivOp>(loc, i32, remaining, dim);
          remaining = ib.create<emitc::RemOp>(loc, i32, remaining, dim);
          // A mask of lower rank lines up with the scores' trailing axes, so
          // the leading score axes have no mask dimension to advance.
          if (d < static_cast<unsigned>(dimOffset))
            continue;
          unsigned md = d - static_cast<unsigned>(dimOffset);
          if (maskTy.getDimSize(md) == 1)
            continue;
          midx = ib.create<emitc::AddOp>(
              loc, i32, midx,
              ib.create<emitc::MulOp>(loc, i32, coord,
                                      constI32(ib, loc, maskStrides[md])));
        }
        Value sum = ib.create<emitc::AddOp>(loc, f32,
                                            loadFlat(ib, loc, scores, i),
                                            loadFlat(ib, loc, m, midx));
        storeFlat(ib, loc, dst, i, sum);
      });
      return success();
    }

    if (auto trans = dyn_cast<::mlir::triton::pim::TransposeOp>(op)) {
      // A permutation, element by element. `absorbed` and `onthefly` say the
      // layout already agrees at the consumer, so no node is emitted for it --
      // but the values still have to land in the right order here, because the
      // kernel's output is one flat buffer either way.
      auto srcTy = cast<RankedTensorType>(trans.getSrc().getType());
      auto dstTy = cast<RankedTensorType>(trans.getResult().getType());
      ArrayRef<int64_t> axes = trans.getAxes();

      FlatBuffer src = buffers.lookup(trans.getSrc());
      FlatBuffer dst = destination(op, trans.getResult());

      unsigned rank = dstTy.getRank();
      SmallVector<int64_t> outStrides(rank, 1), inStrides(rank, 1);
      for (int64_t d = rank - 2; d >= 0; --d) {
        outStrides[d] = outStrides[d + 1] * dstTy.getDimSize(d + 1);
        inStrides[d] = inStrides[d + 1] * srcTy.getDimSize(d + 1);
      }

      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        // Walk the result's coordinates and map each back through the
        // permutation, rather than walking the source: the result is what the
        // caller reads, and its element count is the one that must come out
        // whole.
        Value srcIdx = constI32(ib, loc, 0);
        Value remaining = i;
        for (unsigned d = 0; d < rank; ++d) {
          Value dim = constI32(ib, loc, outStrides[d]);
          Value coord = ib.create<emitc::DivOp>(loc, i32, remaining, dim);
          remaining = ib.create<emitc::RemOp>(loc, i32, remaining, dim);
          // `axes[d]` is the source axis that lands on result axis `d`.
          srcIdx = ib.create<emitc::AddOp>(
              loc, i32, srcIdx,
              ib.create<emitc::MulOp>(loc, i32, coord,
                                      constI32(ib, loc, inStrides[axes[d]])));
        }
        storeFlat(ib, loc, dst, i, loadFlat(ib, loc, src, srcIdx));
      });
      return success();
    }

    if (auto reshape = dyn_cast<::mlir::triton::pim::ReshapeOp>(op)) {
      // Element order is preserved, so this is a copy -- which is exactly why
      // it needs one: the consumer reads the buffer with the new shape, and a
      // result that aliased the source with a different shape would read the
      // right bytes in the wrong order.
      FlatBuffer src = buffers.lookup(reshape.getSrc());
      FlatBuffer dst = destination(op, reshape.getResult());
      if (src.count != dst.count)
        return reshape.emitOpError()
               << "a reshape preserves the element count; got " << src.count
               << " -> " << dst.count;
      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        storeFlat(ib, loc, dst, i, loadFlat(ib, loc, src, i));
      });
      return success();
    }

    if (auto concat = dyn_cast<::mlir::triton::pim::ConcatOp>(op)) {
      auto dstTy = cast<RankedTensorType>(concat.getResult().getType());
      int64_t axis = concat.getAxis();
      if (axis < 0)
        axis += dstTy.getRank();
      if (axis < 0 || axis >= dstTy.getRank())
        return concat.emitOpError() << "axis out of range";

      int64_t inner = 1, outer = 1;
      for (int64_t d = axis + 1; d < dstTy.getRank(); ++d)
        inner *= dstTy.getDimSize(d);
      for (int64_t d = 0; d < axis; ++d)
        outer *= dstTy.getDimSize(d);

      FlatBuffer dst = destination(op, concat.getResult());
      // Each source contributes a run of `inner` elements per outer block,
      // laid end to end along the concatenated axis.
      int64_t offset = 0;
      for (Value src : concat.getSrcs()) {
        auto srcTy = cast<RankedTensorType>(src.getType());
        FlatBuffer buf = buffers.lookup(src);
        int64_t extent = srcTy.getDimSize(axis);
        emitFor(b, loc, 0, outer, [&](OpBuilder &ob, Value o) {
          emitFor(ob, loc, 0, extent * inner, [&](OpBuilder &ib, Value j) {
            Value dstIdx = ib.create<emitc::AddOp>(
                loc, i32,
                ib.create<emitc::MulOp>(
                    loc, i32, o,
                    constI32(ib, loc, inner * dstTy.getDimSize(axis))),
                ib.create<emitc::AddOp>(loc, i32,
                                        constI32(ib, loc, offset * inner), j));
            // The source has its own outer blocks: only the axis extent
            // differs from the result, so block `o` starts at `o*extent*inner`.
            // Without this term every outer block re-reads the source's first
            // block -- right shape, wrong numbers, no diagnostic.
            Value srcIdx = ib.create<emitc::AddOp>(
                loc, i32,
                ib.create<emitc::MulOp>(loc, i32, o,
                                        constI32(ib, loc, extent * inner)),
                j);
            storeFlat(ib, loc, dst, dstIdx, loadFlat(ib, loc, buf, srcIdx));
          });
        });
        offset += extent;
      }
      return success();
    }

    if (auto conv = dyn_cast<::mlir::triton::pim::ConvertOp>(op)) {
      // A pure format change: the same numbers in another element type. The
      // store does the conversion, which is where the rounding and the
      // saturation live.
      FlatBuffer src = buffers.lookup(conv.getSrc());
      FlatBuffer dst = destination(op, conv.getResult());
      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        storeFlat(ib, loc, dst, i, loadFlat(ib, loc, src, i));
      });
      return success();
    }


    if (auto kv = dyn_cast<KvCacheOp>(op)) {
      if (kv.getMode() == KvMode::Read)
        return kv.emitOpError()
               << "pim-lower-to-emitc has no C for a cache read: the decode "
                  "loop reads the cache on the host, where the step counter "
                  "lives, and a kernel that recomputed it would need that "
                  "counter as an operand";
      FlatBuffer value = buffers.lookup(kv.getValue());
      FlatBuffer cache = buffers.lookup(kv.getCache());
      // 更新侧定标走定点通路（缩放 2.0、右移 14）。scale/zp 操作数按元素
      // 广播：一个标量乘到整行上。没有操作数时就是不缩放，与原先一致。
      std::optional<FlatBuffer> sf, zp;
      if (kv.getUpdatesSf())
        sf = buffers.lookup(kv.getUpdatesSf());
      if (kv.getUpdatesZp())
        zp = buffers.lookup(kv.getUpdatesZp());
      auto scaleAt = [&](OpBuilder &ib, Value j) -> Value {
        if (!sf)
          return Value{};
        Value idx = sf->count == 1 ? constI32(ib, loc, 0) : j;
        if (sf->count != 1 && sf->count != value.count)
          return Value{};
        return loadFlat(ib, loc, *sf, idx);
      };
      auto biasAt = [&](OpBuilder &ib, Value j) -> Value {
        if (!zp)
          return Value{};
        Value idx = zp->count == 1 ? constI32(ib, loc, 0) : j;
        if (zp->count != 1 && zp->count != value.count)
          return Value{};
        return loadFlat(ib, loc, *zp, idx);
      };
      if (sf && sf->count != 1 && sf->count != value.count)
        return kv.emitOpError()
               << "updatesSf must be a scalar or match the value's element "
                  "count; got "
               << sf->count;
      if (zp && zp->count != 1 && zp->count != value.count)
        return kv.emitOpError()
               << "updatesZp must be a scalar or match the value's element "
                  "count; got "
               << zp->count;

      if (kv.getMode() == KvMode::Scatter) {
        // One row per call, at the row the index names. The index is i16 by
        // the op's verifier, and it is read as an integer rather than through
        // the f32 path -- a row number is not a value, and routing it through
        // a float would round it.
        FlatBuffer indices = buffers.lookup(kv.getIndices());
        if (indices.count != 1)
          return kv.emitOpError()
                 << "a scatter writes one row per call, so it needs one index; "
                    "got "
                 << indices.count;
        Type storage = storageTypeFor(b, indices.elemTy);
        Value lv = b.create<emitc::SubscriptOp>(
            loc, emitc::LValueType::get(storage), indices.ptr,
            ValueRange{constI32(b, loc, 0)});
        Value raw = b.create<emitc::LoadOp>(loc, storage, lv);
        Value row =
            storage == i32 ? raw
                           : b.create<emitc::CastOp>(loc, i32, raw).getResult();
        // 行号来自 i16 索引缓冲，是运行期的值。verifier 只约束静态尺寸，
        // 所以这里要自己卡上界：越界写是写进调用方的缓冲，不是读到一个坏数
        // 那么轻。`cache.count` 是缓存元素数，写一行占 `value.count` 个。
        if (cache.count % value.count == 0) {
          int64_t rows = cache.count / value.count;
          Value inRange = b.create<emitc::CmpOp>(
              loc, b.getI1Type(), emitc::CmpPredicate::lt, row,
              constI32(b, loc, rows));
          Value safe = b.create<emitc::ConditionalOp>(
              loc, i32, inRange, row, constI32(b, loc, 0));
          row = safe;
        }
        Value base = b.create<emitc::MulOp>(loc, i32, row,
                                            constI32(b, loc, value.count));
        emitFor(b, loc, 0, value.count, [&](OpBuilder &ib, Value j) {
          Value dstIdx = ib.create<emitc::AddOp>(loc, i32, base, j);
          Value x = loadFlat(ib, loc, value, j);
          if (sf || zp)
            x = applyFixedPoint(ib, loc, x, biasAt(ib, j), scaleAt(ib, j),
                                /*shift=*/14);
          storeFlat(ib, loc, cache, dstIdx, x);
        });
        return success();
      }

      // A range write walks contiguous addresses starting at the row `pos`
      // names. `pos` is an operand of this op, not an orchestrator convention:
      // the host mirror writes at `row = pos + t`, and the decode loop calls
      // this once per token. Dropping it made every token overwrite row 0
      // while both this kernel and the mirror looked self-consistent.
      Value base = constI32(b, loc, 0);
      if (Value pos = scalarValues.lookup(kv.getPos()))
        base = b.create<emitc::MulOp>(loc, i32, pos,
                                      constI32(b, loc, value.count));
      emitFor(b, loc, 0, value.count, [&](OpBuilder &ib, Value i) {
        Value dstIdx = ib.create<emitc::AddOp>(loc, i32, base, i);
        Value x = loadFlat(ib, loc, value, i);
        if (sf || zp)
          x = applyFixedPoint(ib, loc, x, biasAt(ib, i), scaleAt(ib, i),
                              /*shift=*/14);
        storeFlat(ib, loc, cache, dstIdx, x);
      });
      return success();
    }

    // A split and a split_heads differ only in how the piece sizes are given:
    // a split names each one, split_heads divides by the head count. Both hand
    // back one result per piece, so both take one output parameter per piece.
    if (auto split = dyn_cast<::mlir::triton::pim::SplitOp>(op)) {
      auto srcTy = cast<RankedTensorType>(split.getSrc().getType());
      int64_t axis = split.getAxis();
      if (axis < 0)
        axis += srcTy.getRank();
      if (axis < 0 || axis >= srcTy.getRank())
        return split.emitOpError() << "axis out of range";

      int64_t inner = 1, outer = 1;
      for (int64_t d = axis + 1; d < srcTy.getRank(); ++d)
        inner *= srcTy.getDimSize(d);
      for (int64_t d = 0; d < axis; ++d)
        outer *= srcTy.getDimSize(d);

      FlatBuffer src = buffers.lookup(split.getSrc());
      ValueRange results = split.getResults();
      SmallVector<int64_t> extents, offsets;
      int64_t running = 0;
      for (Value result : results) {
        int64_t extent = cast<RankedTensorType>(result.getType()).getDimSize(axis);
        extents.push_back(extent);
        offsets.push_back(running);
        running += extent;
      }
      if (running != srcTy.getDimSize(axis))
        return split.emitOpError()
               << "the pieces total " << running
               << " along axis " << axis << ", but the source has "
               << srcTy.getDimSize(axis);

      for (auto [n, result] : llvm::enumerate(results)) {
        FlatBuffer dst = destination(op, result);
        emitFor(b, loc, 0, outer, [&](OpBuilder &ob, Value o) {
          emitFor(ob, loc, 0, extents[n] * inner, [&](OpBuilder &ib, Value j) {
            Value srcIdx = ib.create<emitc::AddOp>(
                loc, i32,
                ib.create<emitc::MulOp>(loc, i32, o,
                                        constI32(ib, loc, inner *
                                                            srcTy.getDimSize(axis))),
                ib.create<emitc::AddOp>(loc, i32,
                                        constI32(ib, loc, offsets[n] * inner), j));
            storeFlat(ib, loc, dst, j, loadFlat(ib, loc, src, srcIdx));
          });
        });
      }
      return success();
    }

    if (auto heads = dyn_cast<SplitHeadsOp>(op)) {
      auto srcTy = cast<RankedTensorType>(heads.getSrc().getType());
      int64_t axis = heads.getAxis();
      if (axis < 0)
        axis += srcTy.getRank();
      if (axis < 0 || axis >= srcTy.getRank())
        return heads.emitOpError() << "axis out of range";

      int64_t numHeads = heads.getNumHeads();
      if (numHeads <= 0)
        return heads.emitOpError() << "numHeads must be positive";
      int64_t extent = srcTy.getDimSize(axis);
      if (extent % numHeads)
        return heads.emitOpError()
               << "the axis holds " << extent << " elements, which is not "
               << numHeads << " whole heads";
      int64_t headSize = extent / numHeads;

      int64_t inner = 1, outer = 1;
      for (int64_t d = axis + 1; d < srcTy.getRank(); ++d)
        inner *= srcTy.getDimSize(d);
      for (int64_t d = 0; d < axis; ++d)
        outer *= srcTy.getDimSize(d);

      FlatBuffer src = buffers.lookup(heads.getSrc());
      ValueRange results = heads.getResults();
      if (static_cast<int64_t>(results.size()) != numHeads)
        return heads.emitOpError()
               << "the op declares " << numHeads << " heads but yields "
               << results.size() << " results, so one of the two is wrong";

      for (auto [n, result] : llvm::enumerate(results)) {
        FlatBuffer dst = destination(op, result);
        emitFor(b, loc, 0, outer, [&](OpBuilder &ob, Value o) {
          emitFor(ob, loc, 0, headSize * inner, [&](OpBuilder &ib, Value j) {
            Value srcIdx = ib.create<emitc::AddOp>(
                loc, i32,
                ib.create<emitc::MulOp>(loc, i32, o,
                                        constI32(ib, loc, inner * extent)),
                ib.create<emitc::AddOp>(
                    loc, i32, constI32(ib, loc, n * headSize * inner), j));
            storeFlat(ib, loc, dst, j, loadFlat(ib, loc, src, srcIdx));
          });
        });
      }
      return success();
    }

    if (auto fpsu = dyn_cast<FpsuScaleOp>(op)) {
      // 加 bias → 乘 scale → 乘 2^-shift → 饱和。shift 是这条通路真正的
      // 定点移位量，必须从 IR 读，不能写死。
      FlatBuffer src = buffers.lookup(fpsu.getSrc());
      FlatBuffer bias = buffers.lookup(fpsu.getBias());
      FlatBuffer scale = buffers.lookup(fpsu.getScale());
      FlatBuffer dst = destination(op, fpsu.getResult());
      int64_t shift = static_cast<int8_t>(fpsu.getShift());
      emitFor(b, loc, 0, dst.count, [&](OpBuilder &ib, Value i) {
        Value idxB = bias.count == 1 ? constI32(ib, loc, 0) : i;
        Value idxS = scale.count == 1 ? constI32(ib, loc, 0) : i;
        Value x = applyFixedPoint(ib, loc, loadFlat(ib, loc, src, i),
                                  loadFlat(ib, loc, bias, idxB),
                                  loadFlat(ib, loc, scale, idxS), shift);
        storeFlat(ib, loc, dst, i, x);
      });
      return success();
    }

    // Saying so beats the alternative this used to do: `return success()` left
    // the op in place, the emitter never saw it, and the generated C silently
    // computed something else while the NumPy mirror "agreed" because it was
    // wrong the same way.
    return op->emitError()
           << "pim-lower-to-emitc does not lower the operator-level op "
           << op->getName()
           << " yet, and dropping it would leave the generated C computing a "
              "different function than the IR describes";
  }
};

} // namespace
