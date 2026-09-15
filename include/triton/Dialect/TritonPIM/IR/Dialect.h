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
// Layout helpers
//===----------------------------------------------------------------------===//

// Default encoding for a tensor of `shape`: one element per tasklet, tasklets
// spread starting from the fastest-changing dimension, row-major order. The
// counterpart of `triton::gpu::getDefaultBlockedEncoding`.
TaskletTiledEncodingAttr getDefaultTaskletTiledEncoding(MLIRContext *context,
                                                        ArrayRef<int64_t> shape,
                                                        int numTasklets,
                                                        int numDpus);

// Bytes a tensor of this type occupies when staged in WRAM whole, or nullopt if
// that cannot be determined statically.
std::optional<int64_t> getTensorSizeInBytes(RankedTensorType type);

} // namespace mlir::triton::pim

#endif // TRITON_DIALECT_TRITONPIM_IR_DIALECT_H_
