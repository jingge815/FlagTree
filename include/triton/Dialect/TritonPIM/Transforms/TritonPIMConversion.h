//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonPIM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONPIM_TRANSFORMS_TRITONPIMCONVERSION_H_
#define TRITON_DIALECT_TRITONPIM_TRANSFORMS_TRITONPIMCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

namespace mlir {

class TritonPIMTypeConverter : public TypeConverter {
public:
  // `placement` is the graph compiler's cross-DPU decision for the tensors this
  // kernel works on, or null when the module carries none (the single-DPU case).
  // It is what lets the attached layout encoding record a real `dpusPerDevice`;
  // `numDpus` cannot, being a hardware count rather than a split.
  TritonPIMTypeConverter(MLIRContext *context, int numTasklets, int numDpus,
                         bool enableSourceRemat,
                         triton::pim::PlacementSpecAttr placement = {});
  int getNumTasklets() const { return numTasklets; }
  int getNumDpus() const { return numDpus; }
  triton::pim::PlacementSpecAttr getPlacement() const { return placement; }

private:
  MLIRContext *context;
  int numTasklets;
  int numDpus;
  triton::pim::PlacementSpecAttr placement;
};

class TritonPIMConversionTarget : public ConversionTarget {
public:
  explicit TritonPIMConversionTarget(MLIRContext &ctx,
                                     TritonPIMTypeConverter &typeConverter);

  // An op is legal once its tensor operands and results carry PIM layouts.
  static bool isDynamicallyLegal(Operation *op,
                                 const TypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONPIM_TRANSFORMS_TRITONPIMCONVERSION_H_
