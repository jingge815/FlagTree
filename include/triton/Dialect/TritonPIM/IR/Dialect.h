#ifndef TRITON_DIALECT_TRITONPIM_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONPIM_IR_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonPIM depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Attributes.h"
#include "triton/Dialect/TritonPIM/IR/Types.h"

#include "triton/Dialect/TritonPIM/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonPIM/IR/Ops.h.inc"

namespace mlir::triton::pim {

//===----------------------------------------------------------------------===//
// Module attributes
//===----------------------------------------------------------------------===//
//
// Written by `convert-triton-to-pim` and read by downstream passes and by the
// GeneSim cost extractor. The counterparts of `ttg.num-warps` and friends.

// Number of DPUs the device exposes.
constexpr static char AttrNumDpusName[] = "pim.num-dpus";
// Number of tasklets each DPU runs.
constexpr static char AttrNumTaskletsName[] = "pim.num-tasklets";
// WRAM budget per DPU, in bytes.
constexpr static char AttrWramBytesName[] = "pim.wram-bytes";
// MRAM budget per DPU, in bytes.
constexpr static char AttrMramBytesName[] = "pim.mram-bytes";
// DMA alignment requirement, in bytes.
constexpr static char AttrDmaAlignName[] = "pim.dma-align";
// Target string, e.g. "pim:v1".
constexpr static char AttrTargetName[] = "pim.target";
// The graph compiler's cross-DPU placement for this kernel's tensors. Written by
// the graph compiler (it is the only party that knows the sharding decision),
// read by `convert-triton-to-pim` and by the tile/DMA passes downstream.
// Absent means single-DPU.
constexpr static char AttrPlacementName[] = "pim.placement";
// Per-DPU MRAM this operator occupies. The shapes reaching `pim-tile-to-budget`
// are already one DPU's share, so this is NOT the global footprint divided by
// the split -- an earlier version did that and loosened the budget by exactly
// the split factor. Written back by that pass: the graph compiler decides the
// split, but only the pass knows the footprint, because only it resolves the
// tile. Read by the GeneSim cost model and the memory planner.
constexpr static char AttrPlacedMramBytesName[] = "pim.placed-mram-bytes";
// The split width `pim-tile-to-budget` actually saw on the module, not a divisor
// it applied (it divides nothing; see above). The placement states the intent;
// this states what the pass saw, so the two can be compared instead of assumed
// equal. A pattern that drops the placement leaves this at 1 while the intent
// still says N, which is the drift a consumer needs to be able to detect.
constexpr static char AttrPlacedShardsName[] = "pim.placed-shards";
// Bytes set aside on each DPU to stage an incoming piece of a `partial`
// reduction; 0 for `shard` and `replicate`, which owe no reduction. Written
// back by `pim-tile-to-budget`: the graph compiler decides that a tensor is
// partial, but the staging size depends on the tile, which only this pass
// resolves. Read by the GeneSim cost model's capacity check.
constexpr static char AttrPlacedReduceBytesName[] = "pim.placed-reduce-bytes";
// Bytes per element this pass actually charged when sizing tiles and the MRAM
// footprint, taken from the operand type (`inferDtypeSize`). Written back by
// `pim-tile-to-budget`: a text-level consumer sees the element *type* but has to
// guess the width from a name table, and guessing wrong scales every byte count
// it derives. This is the dtype dimension's return path.
constexpr static char AttrPlacedElemBytesName[] = "pim.placed-elem-bytes";
// WRAM actually claimed by this kernel's `pim.wram_alloc`s, in bytes. Written
// by `pim-explicit-dma`.
constexpr static char AttrWramBytesUsedName[] = "pim.wram-bytes-used";
constexpr static char AttrTileMName[] = "pim.tile-m";
constexpr static char AttrTileNName[] = "pim.tile-n";
constexpr static char AttrTileKName[] = "pim.tile-k";
constexpr static char AttrTileWramBytesName[] = "pim.tile-wram-bytes";
// L2 budget shared by the units of one NPU, in bytes.
constexpr static char AttrL2BytesName[] = "pim.l2-bytes";
// L1 budget private to one functional unit, in bytes.
constexpr static char AttrL1BytesName[] = "pim.l1-bytes";
// Target revision the emitted graph is written for, e.g. "1.4".
//
// This is the dynamic-quantization family's marker, not a statement that a node
// is phase-shaped: 37 nodes carry it while 69 are phase-shaped, because the 32
// softmax nodes have none. It is also not the manual's revision number -- the
// two are different numbering schemes and must not be converted between.
constexpr static char AttrRtlVersionName[] = "pim.rtl-version";

//===----------------------------------------------------------------------===//
// Memory resources
//===----------------------------------------------------------------------===//

struct WRAM : public SideEffects::Resource::Base<WRAM> {
  StringRef getName() final { return "<WRAM>"; }
};

struct MRAM : public SideEffects::Resource::Base<MRAM> {
  StringRef getName() final { return "<MRAM>"; }
};

// L1 is private to one functional unit; L2 is shared by the units of one NPU.
// They are separate resources from WRAM so that the effect machinery does not
// treat a transfer between two levels as aliasing.
struct L1 : public SideEffects::Resource::Base<L1> {
  StringRef getName() final { return "<L1>"; }
};

struct L2 : public SideEffects::Resource::Base<L2> {
  StringRef getName() final { return "<L2>"; }
};

//===----------------------------------------------------------------------===//
// Hierarchy lookups
//===----------------------------------------------------------------------===//

// Number of tasklets per DPU in scope for `op`, from the enclosing module.
// Falls back to `kDefaultNumTasklets` when the attribute is absent.
int lookupNumTasklets(Operation *op);
// Number of DPUs in scope for `op`. Falls back to 1.
int lookupNumDpus(Operation *op);
// Declared DPU count, or nullopt when the module does not state one. Callers
// that are checking a placement against the hardware must use this: the
// fallback above turns "not declared" into a hardware fact.
std::optional<int64_t> maybeLookupNumDpus(Operation *op);
// WRAM budget in bytes in scope for `op`. Returns nullopt when unset, so
// callers can tell "no budget declared" from "budget of zero".
std::optional<int64_t> maybeLookupWramBytes(Operation *op);
// MRAM budget in bytes in scope for `op`. Returns nullopt when unset.
std::optional<int64_t> maybeLookupMramBytes(Operation *op);
// DMA alignment in bytes in scope for `op`. Returns nullopt when unset.
std::optional<int64_t> maybeLookupDmaAlign(Operation *op);
// L2 budget in bytes in scope for `op`. Returns nullopt when unset.
std::optional<int64_t> maybeLookupL2Bytes(Operation *op);
// L1 budget in bytes in scope for `op`. Returns nullopt when unset.
std::optional<int64_t> maybeLookupL1Bytes(Operation *op);

// Defaults, matching the pass options of `convert-triton-to-pim`. 16 tasklets
// is the UPMEM convention; 64 KiB is an upper bound on the WRAM of a single
// DPU. Both are placeholders until the target hardware is pinned down, which is
// why they are pass options rather than constants in the lowering.
constexpr static int kDefaultNumTasklets = 16;
constexpr static int kDefaultNumDpus = 1;
constexpr static int kDefaultWramBytes = 65536;
constexpr static int64_t kDefaultMramBytes = 4LL * 1024 * 1024 * 1024;
constexpr static int kDefaultDmaAlign = 8;

//===----------------------------------------------------------------------===//
// Quantization spec -> hardware block axes
//===----------------------------------------------------------------------===//

// Which hardware block a quantization decision is projected onto. The three
// see the same decision under their own axis numbering; see
// `deriveHardwareAxes` for the mapping.
enum class HardwareBlock { Fpsu, Pooling, KantorA };

// One block's per-channel / per-group fields. `spcAxis` and `spgAxis` are the
// *block's* numbering, not the quantization spec's tensor axes -- the two must
// not be mixed. -1 means the block has no such axis (-1 is what the target's
// own field carries for the fixed-point unit, which never groups).
struct SpcSpg {
  bool spc;
  int64_t spcAxis;
  bool spg;
  int64_t spgAxis;
  int64_t spgGroupSize;
};

// The spc/spg `block` writes for the quantization decision `spec` describes.
// One decision, three projections -- not three independent configurations.
SpcSpg deriveHardwareAxes(QuantSpecAttr spec, HardwareBlock block);

//===----------------------------------------------------------------------===//
// Layout helpers
//===----------------------------------------------------------------------===//

// Default encoding for a tensor of `shape`: one element per tasklet, tasklets
// spread starting from the fastest-changing dimension, row-major order. The
// counterpart of `triton::gpu::getDefaultBlockedEncoding`.
TaskletTiledEncodingAttr getDefaultTaskletTiledEncoding(MLIRContext *context,
                                                        ArrayRef<int64_t> shape,
                                                        int numTasklets,
                                                        int numDpus);

// Same, but with the cross-DPU split taken from a graph-level placement. This
// is the only way a non-all-ones `dpusPerDevice` gets written: the default
// builder's `numDpus` is a hardware count, not a sharding decision.
TaskletTiledEncodingAttr
getPlacedTaskletTiledEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                              int numTasklets, PlacementSpecAttr placement);

// Whether a module-level placement is consistent with the hardware it is being
// compiled for. Returns failure (after emitting) when it is not; a null
// placement is the single-DPU case and always succeeds.
LogicalResult verifyModulePlacement(Operation *mod, PlacementSpecAttr placement,
                                    int numDpus);

// Whether every PIM-layout tensor type in `mod` records the same cross-DPU split
// that `placement` declares. The two carriers of the Placement dimension can
// drift -- a pattern that rebuilds a tensor type can drop the split -- and a
// dropped split costs downstream as if the tensor were unsharded.
LogicalResult verifyLayoutsMatchPlacement(Operation *mod,
                                          PlacementSpecAttr placement);

// Bytes a tensor of this type occupies when staged in WRAM whole, or nullopt if
// that cannot be determined statically.
std::optional<int64_t> getTensorSizeInBytes(RankedTensorType type);

} // namespace mlir::triton::pim

#endif // TRITON_DIALECT_TRITONPIM_IR_DIALECT_H_
