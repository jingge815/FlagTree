#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Conversion/TritonToTritonPIM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/Transforms/TritonPIMConversion.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONPIM
#include "triton/Conversion/TritonToTritonPIM/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pim;

// Carry over named attrs (e.g. tt.divisibility) that the conversion does not
// otherwise touch.
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

// Rebuilds an op with converted result types and operands. This covers the bulk
// of the conversion: for anything that is not shape-changing, attaching a
// layout to the types is the whole job.
template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                     retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arith & Math
//===----------------------------------------------------------------------===//

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (isa<RankedTensorType>(retShapedType)) {
      assert(value && "expected a dense elements attribute");
      // Reshaping onto the encoded type is how the attribute picks up the
      // layout; the element data is unchanged.
      value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonPIMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>,
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>, GenericOpPattern<arith::NegFOp>,
      // MaxMin
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxNumFOp>,
      GenericOpPattern<arith::MaxSIOp>, GenericOpPattern<arith::MaxUIOp>,
      GenericOpPattern<arith::MinimumFOp>, GenericOpPattern<arith::MinNumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Comparison
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Cast
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>, GenericOpPattern<arith::BitcastOp>,
      // Select
      GenericOpPattern<arith::SelectOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonPIMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<
      GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
      GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
      GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
      GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
      GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
      GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
      GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//===----------------------------------------------------------------------===//
// Triton
//===----------------------------------------------------------------------===//

// Broadcasting keeps the source layout and only widens the shape, so that no
// data movement is implied.
struct TritonBroadcastPattern : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());
    Attribute srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = op.getType().cloneWithEncoding(srcEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

// `expand_dims`, `trans`, `reduce`, `scan`, `join`, `split` and `cat` all infer
// their result encoding through the dialect layout interface, which PIM
// implements. So they need patterns only to rebuild the op, not to compute
// layouts here.
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto argType = cast<RankedTensorType>(adaptor.getSrc().getType());
    if (!argType.getEncoding())
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, adaptor.getSrc(), adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    if (!cast<RankedTensorType>(src.getType()).getEncoding())
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::TransOp>(
                      op, src, adaptor.getOrder()),
                  adaptor.getAttributes());
    return success();
  }
};

// `tt.dot` survives lowering untouched: PIM has no tensor core, so there is no
// operand layout to negotiate. Only the accumulator's result type needs the
// layout attached, and it comes straight from the converted operand.
struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto cType = dyn_cast<RankedTensorType>(adaptor.getC().getType());
    if (!cType || !cType.getEncoding())
      return failure();

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::DotOp>(
                      op, cType, adaptor.getA(), adaptor.getB(), adaptor.getC(),
                      adaptor.getInputPrecision(),
                      adaptor.getMaxNumImpreciseAcc()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Concatenation doubles the extent along dimension 0 while keeping the
    // source layout, matching how TTGIR handles it.
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    auto srcType = cast<RankedTensorType>(adaptor.getLhs().getType());
    if (!srcType.getEncoding())
      return failure();
    auto retType = resultType.cloneWithEncoding(srcType.getEncoding());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::CatOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The result encoding is inferred via inferDefaultJoinOpEncoding.
    rewriter.replaceOpWithNewOp<triton::JoinOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // `split` requires its last dimension to sit whole within one tasklet;
    // relayout the operand so that holds, then split.
    auto srcTy = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto srcEnc = dyn_cast<TaskletTiledEncodingAttr>(srcTy.getEncoding());
    if (!srcEnc)
      return failure();

    unsigned rank = srcTy.getRank();
    SmallVector<unsigned> size(srcEnc.getSizePerTasklet());
    SmallVector<unsigned> tasklets(srcEnc.getTaskletsPerDpu());
    SmallVector<unsigned> dpus(srcEnc.getDpusPerDevice());

    Value src = adaptor.getSrc();
    if (size[rank - 1] != 2 || tasklets[rank - 1] != 1) {
      // Fold the tasklets that covered the split axis into the next dimension
      // out, so the tasklet count stays exact.
      unsigned freed = tasklets[rank - 1];
      size[rank - 1] = 2;
      tasklets[rank - 1] = 1;
      dpus[rank - 1] = 1;
      if (freed > 1 && rank >= 2)
        tasklets[rank - 2] *= freed;

      SmallVector<unsigned> order;
      order.reserve(rank);
      order.push_back(rank - 1);
      for (unsigned dim : srcEnc.getOrder())
        if (dim != rank - 1)
          order.push_back(dim);

      auto newEnc = TaskletTiledEncodingAttr::get(getContext(), size, tasklets,
                                                  dpus, order);
      auto newTy = srcTy.cloneWithEncoding(newEnc);
      src = rewriter.create<ConvertLayoutOp>(op.getLoc(), newTy, src);
    }

    rewriter.replaceOpWithNewOp<triton::SplitOp>(op, src);
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newScan = rewriter.create<triton::ScanOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis(),
        adaptor.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    TypeConverter::SignatureConversion result(op.getNumArguments());
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody().end());
    if (!newOp.getBody().empty())
      rewriter.applySignatureConversion(&newOp.getBody().front(), result,
                                        converter);
    return success();
  }
};

void populateTritonPatterns(TritonPIMTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<
      // clang-format off
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>,
      GenericOpPattern<triton::ReshapeOp>,
      GenericOpPattern<triton::BitcastOp>,
      GenericOpPattern<triton::FpToFpOp>,
      GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>,
      GenericOpPattern<triton::SplatOp>,
      GenericOpPattern<triton::UnsplatOp>,
      GenericOpPattern<triton::AddPtrOp>,
      GenericOpPattern<triton::ClampFOp>,
      GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>,
      GenericOpPattern<triton::MulhiUIOp>,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>,
      GenericOpPattern<triton::MakeRangeOp>,
      GenericOpPattern<triton::ReduceReturnOp>,
      GenericOpPattern<triton::ScanReturnOp>,
      // Memory accesses stay as tt.load/tt.store here; `pim-explicit-dma`
      // turns them into WRAM staging plus DMA.
      GenericOpPattern<triton::LoadOp>,
      GenericOpPattern<triton::StoreOp>,
      GenericOpPattern<triton::HistogramOp>,
      GenericOpPattern<triton::GatherOp>,
      GenericOpPattern<triton::ExternElementwiseOp>,
      GenericOpPattern<triton::PrintOp>,
      GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::AtomicCASOp>,
      GenericOpPattern<triton::AtomicRMWOp>,
      GenericOpPattern<triton::CallOp>,
      GenericOpPattern<ReturnOp>,
      TritonBroadcastPattern,
      TritonCatPattern,
      TritonJoinOpPattern,
      TritonSplitOpPattern,
      TritonReducePattern,
      TritonScanPattern,
      TritonExpandDimsPattern,
      TritonTransPattern,
      TritonDotPattern,
      TritonFuncOpPattern
      // clang-format on
      >(typeConverter, context);
}

//===----------------------------------------------------------------------===//
// SCF & CF
//===----------------------------------------------------------------------===//
// Structural type conversions, borrowed from
// SCF/Transforms/StructuralTypeConversions.cpp via the TTGIR version.

struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                          *getTypeConverter())))
      return rewriter.notifyMatchFailure(op, "could not convert body types");

    newOp->setOperands(adaptor.getOperands());
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), newResultTypes,
                                               adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonPIMTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                          *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                          *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                          *converter)))
      return failure();
    return success();
  }
};

void populateCFPatterns(TritonPIMTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

class ConvertTritonToTritonPIM
    : public triton::impl::ConvertTritonToTritonPIMBase<
          ConvertTritonToTritonPIM> {
public:
  using ConvertTritonToTritonPIMBase::ConvertTritonToTritonPIMBase;

  void runOnOperation() override {
    if (target.getValue().empty()) {
      mlir::emitError(getOperation().getLoc(),
                      "'convert-triton-to-pim' requires 'target' option to be "
                      "set");
      return signalPassFailure();
    }
    if (numTasklets <= 0) {
      mlir::emitError(getOperation().getLoc(),
                      "'num-tasklets' must be positive");
      return signalPassFailure();
    }
    if (numDpus <= 0) {
      mlir::emitError(getOperation().getLoc(), "'num-dpus' must be positive");
      return signalPassFailure();
    }
    if (wramBytes <= 0) {
      mlir::emitError(getOperation().getLoc(), "'wram-bytes' must be positive");
      return signalPassFailure();
    }
    if (mramBytes <= 0) {
      mlir::emitError(getOperation().getLoc(), "'mram-bytes' must be positive");
      return signalPassFailure();
    }
    if (dmaAlign <= 0) {
      mlir::emitError(getOperation().getLoc(), "'dma-align' must be positive");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Record the hardware description before conversion, so that anything
    // consulting the module during conversion sees it.
    Builder b(context);
    mod->setAttr(AttrNumDpusName, b.getI32IntegerAttr(numDpus));
    mod->setAttr(AttrNumTaskletsName, b.getI32IntegerAttr(numTasklets));
    mod->setAttr(AttrWramBytesName, b.getI32IntegerAttr(wramBytes));
    mod->setAttr(AttrMramBytesName, b.getI64IntegerAttr(mramBytes));
    mod->setAttr(AttrDmaAlignName, b.getI32IntegerAttr(dmaAlign));
    mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

    TritonPIMTypeConverter typeConverter(context, numTasklets, numDpus,
                                         enableSourceRemat);
    TritonPIMConversionTarget convTarget(*context, typeConverter);

    RewritePatternSet patterns(context);
    populateArithPatternsAndLegality(typeConverter, patterns);
    populateMathPatternsAndLegality(typeConverter, patterns);
    populateTritonPatterns(typeConverter, patterns);
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);
    patterns.insert<GenericOpPattern<ub::PoisonOp>>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
