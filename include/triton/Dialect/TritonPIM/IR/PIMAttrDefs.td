#ifndef TRITONPIM_ATTRDEFS
#define TRITONPIM_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/TritonPIM/IR/PIMDialect.td"

class TritonPIM_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<TritonPIM_Dialect, name, traits> {
  let attrName = "triton.pim." # attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Tensor layout encoding
//===----------------------------------------------------------------------===//

def TTPIM_TaskletTiledEncodingAttr
    : TritonPIM_Attr<"TaskletTiledEncoding", "tasklet_tiled"> {
  let mnemonic = "tasklet_tiled";

  let description = [{
    Describes how a tensor is decomposed across the PIM execution hierarchy.
    This is the PIM counterpart of `#ttg.blocked`, with the GPU's
    (thread, warp, CTA) hierarchy replaced by (tasklet, DPU).

    Three per-dimension tuples, from innermost outwards:

      - `sizePerTasklet`: how many elements along each dimension a single
        tasklet owns;
      - `taskletsPerDpu`: how the tasklets of one DPU are arranged over the
        dimensions;
      - `dpusPerDevice`: how the DPUs of the device are arranged over the
        dimensions.

    `order` lists dimensions fastest-changing first, as in `#ttg.blocked`.

    The product `sizePerTasklet[i] * taskletsPerDpu[i] * dpusPerDevice[i]`
    describes one pass over dimension `i`; when the tensor is larger the layout
    is replicated over it, again matching `#ttg.blocked` semantics.

    Note on `dpusPerDevice`: an operator kernel only ever sees the address space
    of its *own* DPU, so lowering from TTIR keeps this at all-ones. The field
    exists so that a graph-level compiler, which is what decides cross-DPU
    sharding, has somewhere to record its decision without a layout change.

    Example: a 64x32 tile over 16 tasklets of one DPU, row-major:

    ```mlir
    #pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [16, 1],
                        dpusPerDevice = [1, 1], order = [1, 0]}>
    ```
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerTasklet,
    ArrayRefParameter<"unsigned">:$taskletsPerDpu,
    ArrayRefParameter<"unsigned">:$dpusPerDevice,
    ArrayRefParameter<"unsigned">:$order // fastest-changing axis first
  );

  let builders = [
    // Distribute `numTasklets` tasklets over `shape`, starting from the
    // fastest-changing dimension, giving each tasklet `sizePerTasklet`
    // elements. Any tasklets left over after every dimension is covered are
    // folded into the slowest-changing dimension so the product stays exact.
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerTasklet,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numTasklets,
                     "unsigned":$numDpus), [{
      unsigned rank = sizePerTasklet.size();
      SmallVector<unsigned, 4> taskletsPerDpu(rank, 1);
      SmallVector<unsigned, 4> dpusPerDevice(rank, 1);

      unsigned remaining = numTasklets;
      for (unsigned d = 0; d < rank && remaining > 1; ++d) {
        unsigned i = order[d];
        unsigned avail = std::max<unsigned>(
            1, (unsigned)shape[i] / std::max<unsigned>(1, sizePerTasklet[i]));
        taskletsPerDpu[i] = std::clamp<unsigned>(remaining, 1, avail);
        remaining /= taskletsPerDpu[i];
      }
      // Whatever did not fit wraps at the slowest-changing dimension, so that
      // product(taskletsPerDpu) == numTasklets always holds.
      if (remaining > 1)
        taskletsPerDpu[order[rank - 1]] *= remaining;

      // Kernels are single-DPU by construction; see the note above.
      (void)numDpus;

      return $_get(context, sizePerTasklet, taskletsPerDpu, dpusPerDevice,
                   order);
    }]>
  ];

  let extraClassDeclaration = [{
    // Number of tasklets this layout spreads a tile over.
    unsigned getTotalTaskletsPerDpu() const;
    // Number of DPUs this layout spreads a tensor over.
    unsigned getTotalDpusPerDevice() const;
    // Elements owned by a single tasklet.
    unsigned getTotalSizePerTasklet() const;
  }];

  // Hand-written, like #ttg.blocked: the generic parser cannot read an
  // `ArrayRef<unsigned>` from bracket syntax.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Memory spaces
//===----------------------------------------------------------------------===//

def TTPIM_WRAMSpace : TritonPIM_Attr<"WRAMSpace", "wram"> {
  let mnemonic = "wram";
  let description = [{
    Indicates that a memory descriptor points into WRAM: the small, fast,
    software-managed scratchpad private to a DPU. Data must be moved here
    explicitly before it can be computed on -- there is no cache that would do
    it implicitly.
  }];
}

def TTPIM_MRAMSpace : TritonPIM_Attr<"MRAMSpace", "mram"> {
  let mnemonic = "mram";
  let description = [{
    Indicates that a memory descriptor points into MRAM: the large main memory
    of a DPU. Addressable only by its own DPU; neither the host nor any other
    DPU shares this address space.
  }];
}

def TTPIM_L1Space : TritonPIM_Attr<"L1Space", "l1"> {
  let mnemonic = "l1";
  let description = [{
    Indicates that a memory descriptor points into an L1 buffer: storage private
    to one functional unit, holding the operands that unit is working on.

    Distinct from `#pim.wram` because it is not shared. Two units on the same
    NPU have separate L1s, so a buffer one unit produced is not visible to
    another until it has been moved out to L2. Modelling both as WRAM would make
    that invisible transfer look free.
  }];
}

def TTPIM_L2Space : TritonPIM_Attr<"L2Space", "l2"> {
  let mnemonic = "l2";
  let description = [{
    Indicates that a memory descriptor points into L2: the buffer shared by the
    functional units of a single NPU, and the staging point between them and
    main memory.

    Shared *within* one NPU only. There is no cross-NPU address space, so an L2
    descriptor is never reachable from another NPU -- any such exchange has to go
    through the host.
  }];
}

def TTPIM_MemSpace : AnyAttrOf<[TTPIM_WRAMSpace, TTPIM_MRAMSpace,
                                TTPIM_L1Space, TTPIM_L2Space]>;

//===----------------------------------------------------------------------===//
// DMA direction
//===----------------------------------------------------------------------===//

def TTPIM_DmaMRAMToWRAM : I32EnumAttrCase<"MRAMToWRAM", 0, "mram_to_wram">;
def TTPIM_DmaWRAMToMRAM : I32EnumAttrCase<"WRAMToMRAM", 1, "wram_to_mram">;

def TTPIM_DmaDir : I32EnumAttr<"DmaDir",
  "Direction of an MRAM<->WRAM transfer", [
    TTPIM_DmaMRAMToWRAM,
    TTPIM_DmaWRAMToMRAM
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

//===----------------------------------------------------------------------===//
// Functional units
//===----------------------------------------------------------------------===//
//
// A target NPU is not a single ALU: it is a matrix unit, a vector unit, an
// activation/scaling datapath and a DMA engine, each with its own operand
// staging. Which unit runs an operator decides how it must be described
// downstream, so the assignment is recorded in the IR rather than re-derived.

// The first four ordinals are frozen: old assembly round-trips only if `nmu`
// stays 0, `vpu` 1, `cstl` 2 and `dma` 3.
def TTPIM_UnitNMU  : I32EnumAttrCase<"NMU",  0, "nmu">;
def TTPIM_UnitVPU  : I32EnumAttrCase<"VPU",  1, "vpu">;
// `cstl` was one name for six different blocks. It stays as "not yet broken
// down" -- old IR uses it, and a newly expanded phase should name the actual
// block instead.
def TTPIM_UnitCSTL : I32EnumAttrCase<"CSTL", 2, "cstl">;
def TTPIM_UnitDMA  : I32EnumAttrCase<"DMA",  3, "dma">;
// The blocks `cstl` used to cover. Naming them apart is what lets cost
// extraction tell a table lookup from a rescale from a grouped reduction --
// under one name they all looked like the same traversal.
def TTPIM_UnitFpsu       : I32EnumAttrCase<"FPSU",       4, "fpsu">;
def TTPIM_UnitKantor     : I32EnumAttrCase<"Kantor",     5, "kantor">;
def TTPIM_UnitPooling    : I32EnumAttrCase<"Pooling",    6, "pooling">;
def TTPIM_UnitActivation : I32EnumAttrCase<"Activation", 7, "activation">;
def TTPIM_UnitCombiner   : I32EnumAttrCase<"Combiner",   8, "combiner">;

def TTPIM_FunctionalUnit : I32EnumAttr<"FunctionalUnit",
  "Functional unit of an NPU that executes an operator", [
    TTPIM_UnitNMU, TTPIM_UnitVPU, TTPIM_UnitCSTL, TTPIM_UnitDMA,
    TTPIM_UnitFpsu, TTPIM_UnitKantor, TTPIM_UnitPooling,
    TTPIM_UnitActivation, TTPIM_UnitCombiner
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_FunctionalUnitAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_FunctionalUnit, "unit"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Phase specification
//===----------------------------------------------------------------------===//
//
// The hardware's smallest schedulable unit is not an operator but one streaming
// pass a fixed-function engine makes between two memory levels. An operator
// that takes several such passes is expanded into a chain of ops, and this
// attribute records, on each op of that chain, that it owns one of them.
//
// Only ops that really occupy a traversal carry it: a layout reshape and the
// softmax stabilization (an FPSU affine folded into the exp phase) are helpers,
// so a consumer counts phases by the distinct `index` values it sees, not by
// the number of ops.

def TTPIM_PhaseSpecAttr : TritonPIM_Attr<"PhaseSpec", "phase_spec"> {
  let mnemonic = "phase_spec";

  let description = [{
    One engine traversal of an expanded operator.

    `index` is the 0-based phase number within the operator. `bytes` is the
    *logical* buffer the phase streams -- element count times element size -- not
    a physical L2 segment: segments carry alignment and are folded across nodes,
    which is the orchestrator's business, not the operator's.

    `forceConsecutive` marks a phase that shares an intermediate buffer with its
    neighbours and therefore must not be reordered (the RoPE chain is the case
    that exists). `reads` records which earlier phases a phase consumes, which is
    how the fan-out is preserved: dynamic quantization's phases 1 and 2 both read
    phase 0, and serializing that into a chain would put the reciprocal off by
    256x. A phase may only read a strictly earlier one -- the topology is a DAG,
    so a self-reference or a forward reference would describe a cycle the
    hardware cannot schedule.

    Example:

    ```mlir
    phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu>]
    #pim.phase_spec<index = 1, bytes = 64, unit = activation, reads = [0]>
    ```
  }];

  let parameters = (ins
    "int64_t":$index,
    "int64_t":$bytes,
    EnumParameter<TTPIM_FunctionalUnit>:$unit,
    DefaultValuedParameter<"bool", "false">:$forceConsecutive,
    OptionalParameter<"::mlir::ArrayAttr">:$reads
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Numeric modes
//===----------------------------------------------------------------------===//

def TTPIM_NumFloat      : I32EnumAttrCase<"FloatingPoint", 0, "floating_point">;
def TTPIM_NumFixed      : I32EnumAttrCase<"FixedPoint",    1, "fixed_point">;
def TTPIM_NumFixed2Float: I32EnumAttrCase<"Fixed2Float",   2, "fixed2float">;

def TTPIM_NumericMode : I32EnumAttr<"NumericMode",
  "Numeric format a unit's datapath operates in", [
    TTPIM_NumFloat, TTPIM_NumFixed, TTPIM_NumFixed2Float
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_NumericModeAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_NumericMode, "numeric_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Quantization granularity
//===----------------------------------------------------------------------===//

// MLIR integer types are signless, but the downstream graph format records a
// per-buffer signed/unsigned/float tag. Without this the signedness of an `i8`
// buffer would be lost at the hand-off, so it is carried explicitly.
def TTPIM_ExtSigned   : I32EnumAttrCase<"Signed",   1, "signed_ext">;
def TTPIM_ExtUnsigned : I32EnumAttrCase<"Unsigned", 2, "unsigned_ext">;
def TTPIM_ExtFloat    : I32EnumAttrCase<"Float",    3, "float_ext">;

def TTPIM_DataExtension : I32EnumAttr<"DataExtension",
  "Signedness tag of a quantized buffer", [
    TTPIM_ExtSigned, TTPIM_ExtUnsigned, TTPIM_ExtFloat
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_QuantPerTensor  : I32EnumAttrCase<"PerTensor",  0, "per_tensor">;
def TTPIM_QuantPerChannel : I32EnumAttrCase<"PerChannel", 1, "per_channel">;
def TTPIM_QuantPerGroup   : I32EnumAttrCase<"PerGroup",   2, "per_group">;

def TTPIM_QuantGranularity : I32EnumAttr<"QuantGranularity",
  "How a scale/zero-point set is shared across a tensor", [
    TTPIM_QuantPerTensor, TTPIM_QuantPerChannel, TTPIM_QuantPerGroup
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

// What part a quantized tensor plays. The layout (`QuantGranularity`) says how
// many scales there are; this says whether they mean anything.
//
// Named for what each case *is*, not after the target's 0-6 field numbering:
// two of those numbers describe the same situation from different sides, and a
// reader who has the number still has to look up which. The seven collapse to
// five -- `transparent` (the placeholder that is not really quantized),
// `input`, `weight`, `intermediate` (which covers the fp16-domain middles), and
// `fp_clamp` (type unchanged, bound real).
def TTPIM_RoleTransparent  : I32EnumAttrCase<"Transparent",  0, "transparent">;
def TTPIM_RoleInput        : I32EnumAttrCase<"Input",        1, "input">;
def TTPIM_RoleWeight       : I32EnumAttrCase<"Weight",       2, "weight">;
def TTPIM_RoleIntermediate : I32EnumAttrCase<"Intermediate", 3, "intermediate">;
def TTPIM_RoleFpClamp      : I32EnumAttrCase<"FpClamp",      4, "fp_clamp">;

def TTPIM_QuantRole : I32EnumAttr<"QuantRole",
  "What part a quantized tensor plays in the operator", [
    TTPIM_RoleTransparent, TTPIM_RoleInput, TTPIM_RoleWeight,
    TTPIM_RoleIntermediate, TTPIM_RoleFpClamp
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_QuantSpecAttr : TritonPIM_Attr<"QuantSpec", "quant_spec"> {
  let mnemonic = "quant_spec";

  let description = [{
    Layout of a scale/zero-point set, i.e. how many quantization parameters a
    tensor carries and along which axis they are indexed.

    Only the *layout* lives here. The scale and zero-point values themselves are
    SSA operands of the op that uses them, so they stay visible to the ordinary
    dataflow machinery instead of hiding in an attribute.

      - `per_tensor`: one scale for the whole tensor; `axis` and `groupSize` are
        unused and must be 0.
      - `per_channel`: one scale per index along `axis`.
      - `per_group`: one scale per `groupSize` consecutive elements along `axis`;
        `groupSize` must be positive.

    `dataExt` tags how the quantized buffer's element type is to be read, which
    a signless MLIR integer type cannot say on its own.

    `role` says what part the quantized tensor plays, which the layout alone does
    not determine: a placeholder that is not really quantized and an fp16 tensor
    that is merely clamped have the same layout as a genuinely quantized one, and
    emitting quantization fields for either would describe a tensor that is not
    there. `range` is the value range when it is *not* implied by the element
    type -- the clamped fp16 case is exactly that: the type stays f16 and the
    bound is real, so nothing but an explicit range can express it.

    Example:

    ```mlir
    #pim.quant_spec<granularity = per_channel, axis = 1>
    #pim.quant_spec<granularity = per_tensor, role = fp_clamp, range = [-65504.0, 65504.0]>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_QuantGranularity>:$granularity,
    DefaultValuedParameter<"int64_t", "0">:$axis,
    DefaultValuedParameter<"int64_t", "0">:$groupSize,
    DefaultValuedParameter<
        "::mlir::triton::pim::DataExtension",
        "::mlir::triton::pim::DataExtension::Signed">:$dataExt,
    DefaultValuedParameter<
        "::mlir::triton::pim::QuantRole",
        "::mlir::triton::pim::QuantRole::Intermediate">:$role,
    OptionalParameter<"::mlir::TypeAttr">:$fpDtype,
    OptionalParameter<"::mlir::ArrayAttr">:$range,
    // 逐通道 / 逐组是**量化决策本身**的一部分，不是硬件块的私有配置：
    // 定点、池化、逐元素乘三块只是同一决策的三处投影。写在这里才有一个
    // 唯一真源；各块自己的字段由 deriveHardwareAxes 从这里派生。
    // 默认值按 per_channel 取（实测的主流形态）：逐通道开、逐组关。
    // 其它粒度必须显式给出与之相符的值，verifier 会核对。
    DefaultValuedParameter<"bool", "true">:$spc,
    DefaultValuedParameter<"int64_t", "1">:$spcAxis,
    DefaultValuedParameter<"bool", "false">:$spg,
    DefaultValuedParameter<"int64_t", "-1">:$spgAxis,
    DefaultValuedParameter<"int64_t", "0">:$spgGroupSize
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // The layout half of `verify`, split out so the role checks read as one
    // block rather than trailing a long sequence of axis arithmetic.
    static ::llvm::LogicalResult verifyLayout(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        ::mlir::triton::pim::QuantGranularity granularity, int64_t axis,
        int64_t groupSize);
  }];
}

//===----------------------------------------------------------------------===//
// Post-accumulation scaling datapath
//===----------------------------------------------------------------------===//

def TTPIM_KantorOff        : I32EnumAttrCase<"Off", 0, "off">;
def TTPIM_KantorEltMulFP16 : I32EnumAttrCase<"ElementwiseMulFp16", 1,
                                             "elementwise_mul_fp16">;
def TTPIM_KantorFloatEltAndScale
    : I32EnumAttrCase<"FloatEltwiseAndScale", 2, "float_elt_wise_and_scale">;
def TTPIM_KantorFp2Int     : I32EnumAttrCase<"Fp2IntConverter", 3,
                                             "fp2int_converter">;
def TTPIM_KantorEltMulFixed : I32EnumAttrCase<"ElementwiseMulFixedPoint", 4,
                                              "elementwise_mul_fixed_point">;
def TTPIM_KantorScalar     : I32EnumAttrCase<"Scalar", 5, "scalar">;

def TTPIM_KantorMode : I32EnumAttr<"KantorMode",
  "Mode of the elementwise/rescaling block that follows accumulation", [
    TTPIM_KantorOff, TTPIM_KantorEltMulFP16, TTPIM_KantorFloatEltAndScale,
    TTPIM_KantorFp2Int, TTPIM_KantorEltMulFixed, TTPIM_KantorScalar
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_KantorBlockAttr : TritonPIM_Attr<"KantorBlock", "kantor_block"> {
  let mnemonic = "kantor_block";

  let description = [{
    Configuration of one post-accumulation rescaling block.

    The hardware has several of these blocks, named `A`, `B` and so on, and an
    operator may engage more than one. `id` says which physical block this
    configures, so two entries cannot silently refer to the same one.

    `spec` is the layout of the scale set the block reads; the values themselves
    are data buffers the graph compiler emits.

    Example:

    ```mlir
    #pim.kantor_block<id = "A", mode = scalar,
                      spec = #pim.quant_spec<granularity = per_channel, axis = 1>>
    ```
  }];

  let parameters = (ins
    StringRefParameter<"physical block this configures, e.g. \"A\"">:$id,
    EnumParameter<TTPIM_KantorMode>:$mode,
    OptionalParameter<"::mlir::triton::pim::QuantSpecAttr">:$spec
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Weight binding
//===----------------------------------------------------------------------===//
//
// Which operand rides the matrix unit's weight path and how it is laid out
// there. A model weight and an activation used *as* a weight are both on that
// path, but they differ in element width -- int4 for the projections, int8 for
// the cached K/V slice attention multiplies against -- and once both are
// signless quantized integers that difference is gone from the type.

def TTPIM_WgtFmtWeight    : I32EnumAttrCase<"Weight",           0, "weight">;
def TTPIM_WgtFmtTranspose : I32EnumAttrCase<"WeightsTranspose", 1,
                                            "weights_transpose">;

def TTPIM_WeightFormat : I32EnumAttr<"WeightFormat",
  "Layout a weight operand arrives in on the weight path", [
    TTPIM_WgtFmtWeight, TTPIM_WgtFmtTranspose
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_WgtRoleModel : I32EnumAttrCase<"ModelWeight",        0, "model_weight">;
def TTPIM_WgtRoleAct   : I32EnumAttrCase<"ActivationAsWeight", 1,
                                         "activation_as_weight">;

def TTPIM_WeightRole : I32EnumAttr<"WeightRole",
  "Where a weight-path operand comes from", [
    TTPIM_WgtRoleModel, TTPIM_WgtRoleAct
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

//===----------------------------------------------------------------------===//
// Operand residency
//===----------------------------------------------------------------------===//
//
// Which operand stays resident on the weight path and is reused across passes.
// This is the prefill/decode difference in one attribute, and it is the largest
// cost difference inside a PIM matmul: a resident operand is loaded once, a
// streamed one is re-read every pass.
//
// It cannot be inferred and it cannot be a boolean. The shapes are the same
// either way, and there are three cases rather than two: a model weight, a
// cached K/V slice used *as* a weight, and prefill's activation-by-activation
// multiply where neither side is resident.

def TTPIM_StatWeight : I32EnumAttrCase<"Weight",     0, "weight">;
def TTPIM_StatKV     : I32EnumAttrCase<"KV",         1, "kv">;
def TTPIM_StatAct    : I32EnumAttrCase<"Activation", 2, "activation">;

def TTPIM_Stationarity : I32EnumAttr<"Stationarity",
  "Which operand stays resident on the weight path", [
    TTPIM_StatWeight, TTPIM_StatKV, TTPIM_StatAct
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_StationarityAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_Stationarity, "stationarity"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Mask geometry
//===----------------------------------------------------------------------===//
//
// Which shape of mask the scores are combined with. The two differ by more than
// extent: decode masks one query position against the whole cache, prefill masks
// a square where each position sees only what precedes it. The geometry cannot
// be recovered from the shapes alone once a batch axis is 1.

def TTPIM_MaskVector : I32EnumAttrCase<"Vector",     0, "vector">;
def TTPIM_MaskTril   : I32EnumAttrCase<"CausalTril", 1, "causal_tril">;

def TTPIM_MaskLayout : I32EnumAttr<"MaskLayout",
  "Geometry of an attention mask", [
    TTPIM_MaskVector, TTPIM_MaskTril
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_MaskLayoutAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_MaskLayout, "mask_layout"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Vector unit parameters
//===----------------------------------------------------------------------===//

def TTPIM_VpuParamsAttr : TritonPIM_Attr<"VpuParams", "vpu_params"> {
  let mnemonic = "vpu_params";

  let description = [{
    The vector unit's configuration for a normalization.

    This is the one operator whose target configuration is a nested block rather
    than a flat field set, and the block is small: which axis is normalized, and
    whether the unit applies an input scale on the way in.

    Both defaults match every occurrence measured in the reference artifact
    (`axis = -1`, no input scaling), so an absent attribute and an explicit
    default describe the same hardware.
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "-1">:$axis,
    DefaultValuedParameter<"bool", "false">:$useScaling
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Fusion shape
//===----------------------------------------------------------------------===//
//
// What a fusion looks like in the target's graph format. The two forms are not
// two spellings of one thing:
//
//   - `named` is a nested block carrying the folded operator's own name, type
//     and activation kind (the SiLU folded into the gate projection);
//   - `flat` is a single integer flag saying "this node's tail is a
//     quantization" (the RoPE chain's quantized tail).
//
// They are mutually exclusive, and a node carrying the wrong one describes a
// fusion that is not there.

def TTPIM_CtrNamed : I32EnumAttrCase<"Named", 0, "named">;
def TTPIM_CtrFlat  : I32EnumAttrCase<"Flat",  1, "flat">;

def TTPIM_ContractionForm : I32EnumAttr<"ContractionForm",
  "Shape a fusion takes in the target's graph format", [
    TTPIM_CtrNamed, TTPIM_CtrFlat
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_ContractionAttr : TritonPIM_Attr<"Contraction", "contraction"> {
  let mnemonic = "contraction";

  let description = [{
    The fusion this operator carries, in the shape the graph format writes it.

    The folded operator's table, window and activation mode stay on *this*
    operator rather than inside the nested block -- the block names the fusion,
    it does not re-describe the datapath.

    Example:

    ```mlir
    #pim.contraction<form = named, blockName = "fused_Silu_act", innerOp = "Lut", actKind = "Silu">
    #pim.contraction<form = flat, flagName = "dq_contraction">
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_ContractionForm>:$form,
    OptionalParameter<"::mlir::StringAttr">:$blockName,
    OptionalParameter<"::mlir::StringAttr">:$innerOp,
    OptionalParameter<"::mlir::StringAttr">:$actKind,
    OptionalParameter<"::mlir::StringAttr">:$flagName
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTPIM_WeightBindingAttr : TritonPIM_Attr<"WeightBinding", "weight_binding"> {
  let mnemonic = "weight_binding";

  let description = [{
    How one operand of `pim.matmul` is staged on the weight path.

    `elemBits` is 4 for a model weight and 8 for an activation used as one.
    **int4 is stored one byte per element**, sign-extended over the range
    `[-8, 7]` -- there is no nibble packing and no special alignment. That is
    the opposite of what the width suggests, and getting it wrong produces a
    weight buffer half the expected size.

    `groupSize` is the granularity the scale set was built at, which the
    accumulator has to dequantize in step with: dequantizing after summing the
    whole row cannot merge two groups that carry different scales.

    `sfMultiplier` is the subnormal guard: when a per-group scale would land in
    fp16's subnormal range it is stored multiplied by a power of two, and the
    matching reciprocal is applied on the scaling side, so the product is one.

    `contentHash` is the weight buffer's content fingerprint, used to share one
    weight between nodes. It is optional here -- whoever holds the bytes is the
    natural place to compute it, and that is not always the operator compiler.

    Example:

    ```mlir
    #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_WeightFormat>:$format,
    EnumParameter<TTPIM_WeightRole>:$role,
    "int64_t":$elemBits,
    DefaultValuedParameter<"int64_t", "128">:$groupSize,
    DefaultValuedParameter<"int64_t", "1">:$sfMultiplier,
    OptionalParameter<"::mlir::StringAttr">:$contentHash
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTPIM_KantorSpecAttr : TritonPIM_Attr<"KantorSpec", "kantor_spec"> {
  let mnemonic = "kantor_spec";

  let description = [{
    Configuration of the elementwise-multiply / format-conversion unit for one
    phase, plus the rescaling blocks it engages.

    `mode` is the dialect's reading of what the unit does. `cardValue` is the
    value the target's own field carries, and it is kept separately because the
    two do not agree: the RoPE multiplies are card value 5 while being
    elementwise multiplies, and the dialect calls 5 `scalar`. A consumer cannot
    recover the card value from `#pim.datapath`, which is why it is stored
    rather than derived. -1 means the mode's own value is the card value.

    `blocks` is asymmetric by design: the A and B blocks do not carry the same
    fields, and the RoPE chain is the case where one has a scale and the other
    does not. Nothing here may assume the two mirror each other.

    Example:

    ```mlir
    kantor = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_KantorMode>:$mode,
    DefaultValuedParameter<"int64_t", "-1">:$cardValue,
    OptionalParameter<"::mlir::ArrayAttr">:$blocks
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Fixed-point rescaling
//===----------------------------------------------------------------------===//
//
// The scaling datapath a phase runs its operands through: add a 32-bit bias,
// multiply by a 16-bit scale, round, right-shift, saturate. It does no
// reduction and no table lookup, which is what separates it from the pooling
// and activation units.
//
// `floating_point` and `floating_point_32` are two distinct settings of the
// same unit, not two ways of writing one: the 16-bit phases use the former and
// the 32-bit accumulator path the latter. The target's field carries the name,
// so the enum carries the name too -- the ordinal is not the value.

def TTPIM_FpsuFloating   : I32EnumAttrCase<"FloatingPoint",   0, "floating_point">;
def TTPIM_FpsuFloating32 : I32EnumAttrCase<"FloatingPoint32", 1, "floating_point_32">;
def TTPIM_FpsuFixed      : I32EnumAttrCase<"FixedPoint",      2, "fixed_point">;

def TTPIM_FpsuMode : I32EnumAttr<"FpsuMode",
  "Numeric format the fixed-point rescaling unit works in", [
    TTPIM_FpsuFloating, TTPIM_FpsuFloating32, TTPIM_FpsuFixed
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_FpsuSpecAttr : TritonPIM_Attr<"FpsuSpec", "fpsu_spec"> {
  let mnemonic = "fpsu_spec";

  let description = [{
    Configuration of one rescale a phase performs on its way out.

    `mode` names the numeric format the unit runs in. The per-channel and
    per-group flags describe the granularity of the scale set the unit applies;
    they are the *same* quantization decision the pooling and elementwise
    blocks see, projected onto this unit's field names, so the axes here are
    this unit's numbering (1 for the channel axis) and not the quantization
    spec's. `deriveHardwareAxes` is what projects the spec onto them.

    `spc` defaults to true because this unit always applies its per-channel
    scale: every one of the target's nodes writes `fpsu_spc 1` at the top level,
    in all four dynamic-quantization phases and in the softmax phases. Only a
    per_tensor spec -- a single scale for the whole tensor -- turns it off.
    `spg` is false for this unit in every configuration: grouping is the pooling
    and elementwise-multiply units' work, not this one's.

    The values themselves are not here: the bias, scale and shift tensors are
    operands of the op that owns the phase.

    Example:

    ```mlir
    fpsu = #pim.fpsu_spec<mode = floating_point>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_FpsuMode>:$mode,
    DefaultValuedParameter<"bool", "true">:$spc,
    DefaultValuedParameter<"int64_t", "1">:$spcAxis,
    DefaultValuedParameter<"bool", "false">:$spg,
    DefaultValuedParameter<"int64_t", "-1">:$spgAxis,
    DefaultValuedParameter<"int64_t", "-1">:$spgGroupSize
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTPIM_DatapathAttr : TritonPIM_Attr<"Datapath", "datapath"> {
  let mnemonic = "datapath";

  let description = [{
    Configuration of the fixed-function datapath an operator runs through: the
    numeric format of the accumulating unit, then the two post-accumulation
    blocks that rescale its output.

      - `nmuMode`: format the accumulating unit consumes and produces.
        `fixed2float` means fixed-point in, floating-point out.
      - `scaleMode` / `scaleSpec`: the per-output-channel scale-and-bias block
        applied to the accumulator.
      - `kantorBlocks`: the elementwise-multiply and float-to-fixed conversion
        blocks after it -- an array of `#pim.kantor_block`, one entry per engaged
        physical block. Absent means none is engaged, which is how the datapath
        says the accumulator's scaled output is already in its final form.

    This is deliberately a description of *configuration*, not of data: the
    scale, bias and shift tensors these blocks read are operands of the op.

    Example:

    ```mlir
    #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point,
                  scaleSpec = #pim.quant_spec<granularity = per_channel, axis = 1>,
                  kantorBlocks = [#pim.kantor_block<id = "A", mode = scalar>]>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_NumericMode>:$nmuMode,
    EnumParameter<TTPIM_NumericMode>:$scaleMode,
    OptionalParameter<"::mlir::triton::pim::QuantSpecAttr">:$scaleSpec,
    OptionalParameter<"::mlir::ArrayAttr">:$kantorBlocks,
    // The integer MAC accumulates a group at a time, dequantizing each group
    // before folding it into the running total. The alternative -- accumulate
    // the whole row in integers, then dequantize once -- gives different
    // numbers, because two groups with different scales cannot share one
    // integer accumulator.
    //
    // `pim.dequantize` stays for the accumulate-then-dequantize meaning; the
    // w4a8 projections must not take that path.
    DefaultValuedParameter<"bool", "false">:$groupDequantAccum,
    DefaultValuedParameter<"int64_t", "0">:$groupSize
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Activation lookup tables
//===----------------------------------------------------------------------===//
//
// The target implements activations as table lookups rather than as arithmetic,
// so `relu` and `silu` are the same instruction with different table contents.
// The kind is kept in the IR anyway: a table is opaque bytes, and neither the
// cost model nor a reader can recover the function from it.

def TTPIM_ActRelu      : I32EnumAttrCase<"Relu",      0, "relu">;
def TTPIM_ActReluX     : I32EnumAttrCase<"ReluX",     1, "relu_x">;
def TTPIM_ActLeakyRelu : I32EnumAttrCase<"LeakyRelu", 2, "leaky_relu">;
def TTPIM_ActSigmoid   : I32EnumAttrCase<"Sigmoid",   3, "sigmoid">;
def TTPIM_ActSilu      : I32EnumAttrCase<"Silu",      4, "silu">;
def TTPIM_ActTanh      : I32EnumAttrCase<"Tanh",      5, "tanh">;
def TTPIM_ActGelu      : I32EnumAttrCase<"Gelu",      6, "gelu">;
def TTPIM_ActExp       : I32EnumAttrCase<"Exp",       7, "exp">;
def TTPIM_ActReciprocal: I32EnumAttrCase<"Reciprocal",8, "reciprocal">;
def TTPIM_ActSqrt      : I32EnumAttrCase<"Sqrt",      9, "sqrt">;
def TTPIM_ActRsqrt     : I32EnumAttrCase<"Rsqrt",    10, "rsqrt">;
// Dynamic quantization's phase 1 evaluates the identity through the table so
// that the scaling lands in the FPSU rather than in arithmetic. It used to be
// spelled `relu`, which described the wrong function. Appending after the
// existing eleven keeps their ordinals -- and therefore old assembly -- intact.
def TTPIM_ActIdentity  : I32EnumAttrCase<"Identity", 11, "identity">;

def TTPIM_ActivationKind : I32EnumAttr<"ActivationKind",
  "Function a lookup table evaluates", [
    TTPIM_ActRelu, TTPIM_ActReluX, TTPIM_ActLeakyRelu, TTPIM_ActSigmoid,
    TTPIM_ActSilu, TTPIM_ActTanh, TTPIM_ActGelu, TTPIM_ActExp,
    TTPIM_ActReciprocal, TTPIM_ActSqrt, TTPIM_ActRsqrt, TTPIM_ActIdentity
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_ActivationKindAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_ActivationKind, "activation"> {
  let assemblyFormat = "`<` $value `>`";
}

// Which half of the table an entry addresses. Interleaved tables let two
// operators share one physical table.
def TTPIM_LutRegular : I32EnumAttrCase<"Regular", 0, "regular">;
def TTPIM_LutEven    : I32EnumAttrCase<"Even",    1, "even">;
def TTPIM_LutOdd     : I32EnumAttrCase<"Odd",     2, "odd">;

def TTPIM_LutMode : I32EnumAttr<"LutMode",
  "Addressing mode of an activation lookup table", [
    TTPIM_LutRegular, TTPIM_LutEven, TTPIM_LutOdd
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_LutModeAttr : EnumAttr<TritonPIM_Dialect, TTPIM_LutMode, "lut_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Elementwise and pooling kinds
//===----------------------------------------------------------------------===//

def TTPIM_EltAdd    : I32EnumAttrCase<"Add",    0, "add">;
def TTPIM_EltSub    : I32EnumAttrCase<"Sub",    1, "sub">;
def TTPIM_EltMul    : I32EnumAttrCase<"Mul",    2, "mul">;
def TTPIM_EltDiv    : I32EnumAttrCase<"Div",    3, "div">;
def TTPIM_EltMax    : I32EnumAttrCase<"Max",    4, "max">;
def TTPIM_EltMin    : I32EnumAttrCase<"Min",    5, "min">;
// Hardware-native group absmax used by dynamic-quantization phase 0.
// Distinct from Max: DQ pooling type 4 is max(|x|), not max(x).
def TTPIM_EltAbsMax : I32EnumAttrCase<"AbsMax", 6, "absmax">;

def TTPIM_EltwiseKind : I32EnumAttr<"EltwiseKind",
  "Elementwise / reduction kind", [
    TTPIM_EltAdd, TTPIM_EltSub, TTPIM_EltMul, TTPIM_EltDiv,
    TTPIM_EltMax, TTPIM_EltMin, TTPIM_EltAbsMax
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_EltwiseKindAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_EltwiseKind, "eltwise"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTPIM_PoolMax    : I32EnumAttrCase<"Max",           0, "max">;
def TTPIM_PoolAvg    : I32EnumAttrCase<"Average",       1, "average">;
def TTPIM_PoolGlobal : I32EnumAttrCase<"GlobalAverage", 2, "global_average">;

// No AbsMax here on purpose: the dynamic-quantization group absmax is a
// reduction along an axis, so it is `pim.reduce_axis` with
// `#pim.eltwise<absmax>`. This enum is for sliding-window pooling.
def TTPIM_PoolKind : I32EnumAttr<"PoolKind",
  "Pooling reduction", [TTPIM_PoolMax, TTPIM_PoolAvg, TTPIM_PoolGlobal]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_PoolKindAttr : EnumAttr<TritonPIM_Dialect, TTPIM_PoolKind, "pool"> {
  let assemblyFormat = "`<` $value `>`";
}

// A *grouped* reduction -- collapse each run of `groupSize` elements down to
// one, which is what dynamic quantization's phase 0 does. Kept apart from
// `PoolKind` rather than folded into it: a windowed pool has geometry, a
// grouped reduction has a group size and a quantization granularity, and the
// two carry different target fields. In particular the sliding-window path
// never performs an absmax, so adding one there would describe a reduction the
// hardware does not do.
def TTPIM_GPoolAbsMax : I32EnumAttrCase<"AbsMax", 0, "absmax">;
def TTPIM_GPoolMax    : I32EnumAttrCase<"Max",    1, "max">;
def TTPIM_GPoolSum    : I32EnumAttrCase<"Sum",    2, "sum">;

def TTPIM_GlobalPoolKind : I32EnumAttr<"GlobalPoolKind",
  "Grouped reduction kind on the pooling unit", [
    TTPIM_GPoolAbsMax, TTPIM_GPoolMax, TTPIM_GPoolSum
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_GlobalPoolKindAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_GlobalPoolKind, "pool_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTPIM_PoolDTypeFloat : I32EnumAttrCase<"FloatingPoint", 0, "floating_point">;
def TTPIM_PoolDTypeFixed : I32EnumAttrCase<"FixedPoint",    1, "fixed_point">;

def TTPIM_PoolDType : I32EnumAttr<"PoolDType",
  "Numeric domain a grouped reduction accumulates in", [
    TTPIM_PoolDTypeFloat, TTPIM_PoolDTypeFixed
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_PoolDTypeAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_PoolDType, "pool_dtype"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Window geometry
//===----------------------------------------------------------------------===//

def TTPIM_WindowAttr : TritonPIM_Attr<"Window", "window"> {
  let mnemonic = "window";

  let description = [{
    Sliding-window geometry shared by convolution and pooling: kernel extent,
    stride, padding and dilation, plus the group count for grouped convolution.

    `pads` is `[top, right, bottom, left]`, matching the order the downstream
    graph format uses, so the two do not disagree about which edge is which.
    `outputPadding` is used only by transposed convolution and is `[]`
    otherwise.

    Example:

    ```mlir
    #pim.window<kernel = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
                dilations = [1, 1]>
    ```
  }];

  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$kernel,
    "::mlir::DenseI64ArrayAttr":$strides,
    "::mlir::DenseI64ArrayAttr":$pads,
    "::mlir::DenseI64ArrayAttr":$dilations,
    DefaultValuedParameter<"int64_t", "1">:$group,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$outputPadding
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTPIM_BroadcastSpecAttr : TritonPIM_Attr<"BroadcastSpec", "broadcast_spec"> {
  let mnemonic = "broadcast_spec";

  let description = [{
    广播语义：sin/cos 表从 `[1,1,1,128]` 广播到 `[1,32,1,128]` 时，
    广播那一路带自己的量化参数，所以它是一个独立的语义步骤而不是形状推断。

    `axes` 与 `repeats` 一一对应：第 i 个轴重复 `repeats[i]` 次。

    Example:

    ```mlir
    #pim.broadcast_spec<axes = [1], repeats = [32]>
    ```
  }];

  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$axes,
    "::mlir::DenseI64ArrayAttr":$repeats
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTPIM_ActSpecAttr : TritonPIM_Attr<"ActSpec", "act_spec"> {
  let mnemonic = "act_spec";

  let description = [{
    An activation folded into the operator that produces its input.

    The target's graph format stores an activation inside its producer rather
    than as a separate node, so this is an attribute on the producing op instead
    of a standalone `pim.lut`. Only the activation's configuration lives here;
    the lookup table itself is a data buffer the graph compiler emits.

    `alpha` carries the slope of `leaky_relu`, and `clipMin`/`clipMax` the bounds
    of `relu_x`; each is meaningless for the other kinds.

    Example:

    ```mlir
    #pim.act_spec<kind = silu, mode = regular>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_ActivationKind>:$kind,
    DefaultValuedParameter<
        "::mlir::triton::pim::LutMode",
        "::mlir::triton::pim::LutMode::Regular">:$mode,
    OptionalParameter<"::mlir::FloatAttr">:$alpha,
    OptionalParameter<"::mlir::FloatAttr">:$clipMin,
    OptionalParameter<"::mlir::FloatAttr">:$clipMax
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TTPIM_PoolSpecAttr : TritonPIM_Attr<"PoolSpec", "pool_spec"> {
  let mnemonic = "pool_spec";

  let description = [{
    A pooling reduction folded into the operator that produces its input.

    The target's graph format allows a pool to sit in the same node as its
    producer, after any folded activation -- so a convolution, its activation and
    a following pool can all be one node. This attribute is that trailing pool.

    `window` is required for a windowed reduction and must be omitted for
    `global_average`, which always covers the whole spatial extent.

    Example:

    ```mlir
    #pim.pool_spec<kind = max,
                   window = #pim.window<kernel = [3, 3], strides = [2, 2],
                                        pads = [1, 1, 1, 1], dilations = [1, 1]>>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_PoolKind>:$kind,
    OptionalParameter<"::mlir::triton::pim::WindowAttr">:$window
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Transpose purpose
//===----------------------------------------------------------------------===//
//
// Why a tensor's dimensions are permuted, which is a different question from
// *how*. The axis order and the purpose go together in this network: swapping
// the head axis emits a node, reordering the halves of a rotary embedding is
// absorbed into the rotary op, and transposing K for a score matmul disappears
// into the weight's layout.
//
// `Absorbed` is what keeps the node count right. A permutation that survives
// into the graph format as a node is a layer the hardware does not have, and
// `Pure` -- an MLIR statement about side effects -- says nothing either way.

def TTPIM_TensorTranspose : I32EnumAttrCase<"TensorTranspose", 0, "tensor_transpose">;
def TTPIM_LayoutReorder  : I32EnumAttrCase<"LayoutReorder",   1, "layout_reorder">;
def TTPIM_TpAbsorbed     : I32EnumAttrCase<"Absorbed",        2, "absorbed">;

def TTPIM_TransposePurpose : I32EnumAttr<"TransposePurpose",
  "Why a tensor's dimensions are permuted", [
    TTPIM_TensorTranspose, TTPIM_LayoutReorder, TTPIM_TpAbsorbed
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_TransposePurposeAttr
    : TritonPIM_Attr<"TransposePurpose", "transpose_purpose"> {
  let mnemonic = "transpose_purpose";

  let description = [{
    How a phase addresses what it reads and writes.

    `cardValue` is the addressing card the phase walks with, carried alongside
    the purpose rather than as a purpose of its own: the reciprocal phase of
    dynamic quantization walks the per-group scalars, and whether that is a
    scalar walk or a strided one depends on how many groups there are. The
    graph format has no field for it -- an exhaustive search for a
    transpose-type key finds nothing -- so it stays here and is not emitted.

    Example:

    ```mlir
    #pim.transpose_purpose<purpose = layout_reorder, cardValue = 1>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_TransposePurpose>:$purpose,
    DefaultValuedParameter<"int64_t", "0">:$cardValue
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Combination mode
//===----------------------------------------------------------------------===//
//
// How a partial result rejoins the chain that consumes it. The graph format
// has no field for this -- an exhaustive search of the merge unit, partial-sum
// and skip-connection families finds nothing -- so it is configuration, and
// adding it must leave the emitted graph byte for byte unchanged.

def TTPIM_Straightforward  : I32EnumAttrCase<"Straightforward",  0, "straightforward">;
def TTPIM_SkipConnection   : I32EnumAttrCase<"SkipConnection",   1, "skip_connection">;
def TTPIM_OutputMapMerger  : I32EnumAttrCase<"OutputMapMerger",  2, "output_map_merger">;

def TTPIM_CombineMode : I32EnumAttr<"CombineMode",
  "How a partial result rejoins the chain that consumes it", [
    TTPIM_Straightforward, TTPIM_SkipConnection, TTPIM_OutputMapMerger
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_CombineModeAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_CombineMode, "combine_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Half-word permutation
//===----------------------------------------------------------------------===//
//
// Half the words come from the first half of the row and half from the second,
// with the halves swapped. Which words move is a function of the column index
// rather than of the values, so code that reorders data and code that negates
// must agree about it -- the sign has to travel with the sine, which reads the
// second half.

//===----------------------------------------------------------------------===//
// Cache write mode
//===----------------------------------------------------------------------===//
//
// The two write paths differ in addressing, not in extent: a scatter takes its
// destination from an index operand, a range write walks contiguous addresses.
// That is why the index is required in one mode and forbidden in the other --
// an index that nothing reads would be a silent second source of truth.

def TTPIM_KvScatter    : I32EnumAttrCase<"Scatter",    0, "scatter">;
def TTPIM_KvRangeWrite : I32EnumAttrCase<"RangeWrite", 1, "range_write">;
def TTPIM_KvRead       : I32EnumAttrCase<"Read",       2, "read">;

def TTPIM_KvMode : I32EnumAttr<"KvMode",
  "How a cache op addresses what it reads or writes", [
    TTPIM_KvScatter, TTPIM_KvRangeWrite, TTPIM_KvRead
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_KvModeAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_KvMode, "kv_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // TRITONPIM_ATTRDEFS
