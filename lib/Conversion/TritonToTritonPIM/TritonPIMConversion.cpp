#include "triton/Dialect/TritonPIM/Transforms/TritonPIMConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::pim;

//===----------------------------------------------------------------------===//
// TypeConverter
//===----------------------------------------------------------------------===//

TritonPIMTypeConverter::TritonPIMTypeConverter(MLIRContext *context,
                                               int numTasklets, int numDpus,
                                               bool enableSourceRemat)
    : context(context), numTasklets(numTasklets), numDpus(numDpus) {
  addConversion([](Type type) { return type; });

  // Attach a tasklet layout to every tensor that lacks one.
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    if (tensorType.getEncoding())
      return tensorType;
    TaskletTiledEncodingAttr encoding = getDefaultTaskletTiledEncoding(
        this->context, tensorType.getShape(), this->numTasklets, this->numDpus);
    return tensorType.cloneWithEncoding(encoding);
  });

  // A `tt.ptr<tensor<>>` needs the layout attached to its pointee.
  addConversion([this](triton::PointerType ptrType) -> triton::PointerType {
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (!pointeeTensorType)
      return ptrType;
    return triton::PointerType::get(convertType(pointeeTensorType),
                                    ptrType.getAddressSpace());
  });

  if (enableSourceRemat) {
    addSourceMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, tensorType, inputs)
          .getResult(0);
    });
  }

  // Called when a remapped operand's layout is not the one its consumer wants.
  // Partial conversion needs this hook to have somewhere to put the mismatch,
  // which is why the dialect carries a `convert_layout` op at all.
  addTargetMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                             ValueRange inputs, Location loc) -> Value {
    return builder.create<ConvertLayoutOp>(loc, tensorType, inputs.front())
        .getResult();
  });
}

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

TritonPIMConversionTarget::TritonPIMConversionTarget(
    MLIRContext &context, TritonPIMTypeConverter &typeConverter)
    : ConversionTarget(context) {
  addLegalDialect<TritonPIMDialect>();

  // These SCF ops have no PIM lowering: `parallel`/`reduce` presuppose a
  // hardware scheduler PIM does not have, and `execute_region` would hide
  // control flow the DMA placement needs to see.
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect, ub::UBDialect>(
      [&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });

  // Unlike TTGIR, `tt.dot` needs no special operand layout: there is no tensor
  // core to feed, so the operands stay in whatever tasklet layout they have.
  // The generic check above is therefore sufficient for it.

  addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) -> bool {
    for (auto arg : funcOp.getArguments()) {
      if (auto tensor = dyn_cast<RankedTensorType>(arg.getType()))
        if (!tensor.getEncoding())
          return false;
    }
    return true;
  });
}

bool TritonPIMConversionTarget::isDynamicallyLegal(
    Operation *op, const TypeConverter &typeConverter) {
  for (auto &region : op->getRegions())
    if (!typeConverter.isLegal(&region))
      return false;
  return typeConverter.isLegal(op);
}
