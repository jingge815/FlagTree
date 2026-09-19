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

def TTPIM_DmaDirAttr : EnumAttr<TritonPIM_Dialect, TTPIM_DmaDir, "dma_dir">;

//===----------------------------------------------------------------------===//
// Functional units
//===----------------------------------------------------------------------===//
//
// A target NPU is not a single ALU: it is a matrix unit, a vector unit, an
// activation/scaling datapath and a DMA engine, each with its own operand
// staging. Which unit runs an operator decides how it must be described
// downstream, so the assignment is recorded in the IR rather than re-derived.

def TTPIM_UnitNMU  : I32EnumAttrCase<"NMU",  0, "nmu">;
def TTPIM_UnitVPU  : I32EnumAttrCase<"VPU",  1, "vpu">;
def TTPIM_UnitCSTL : I32EnumAttrCase<"CSTL", 2, "cstl">;
def TTPIM_UnitDMA  : I32EnumAttrCase<"DMA",  3, "dma">;

def TTPIM_FunctionalUnit : I32EnumAttr<"FunctionalUnit",
  "Functional unit of an NPU that executes an operator", [
    TTPIM_UnitNMU, TTPIM_UnitVPU, TTPIM_UnitCSTL, TTPIM_UnitDMA
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::pim";
}

def TTPIM_FunctionalUnitAttr
    : EnumAttr<TritonPIM_Dialect, TTPIM_FunctionalUnit, "unit"> {
  let assemblyFormat = "`<` $value `>`";
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

    Example:

    ```mlir
    #pim.quant_spec<granularity = per_channel, axis = 1>
    ```
  }];

  let parameters = (ins
    EnumParameter<TTPIM_QuantGranularity>:$granularity,
    DefaultValuedParameter<"int64_t", "0">:$axis,
    DefaultValuedParameter<"int64_t", "0">:$groupSize,
    DefaultValuedParameter<
        "::mlir::triton::pim::DataExtension",
        "::mlir::triton::pim::DataExtension::Signed">:$dataExt
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
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
    OptionalParameter<"::mlir::ArrayAttr">:$kantorBlocks
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

def TTPIM_ActivationKind : I32EnumAttr<"ActivationKind",
  "Function a lookup table evaluates", [
    TTPIM_ActRelu, TTPIM_ActReluX, TTPIM_ActLeakyRelu, TTPIM_ActSigmoid,
    TTPIM_ActSilu, TTPIM_ActTanh, TTPIM_ActGelu, TTPIM_ActExp,
    TTPIM_ActReciprocal, TTPIM_ActSqrt, TTPIM_ActRsqrt
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

#endif // TRITONPIM_ATTRDEFS
