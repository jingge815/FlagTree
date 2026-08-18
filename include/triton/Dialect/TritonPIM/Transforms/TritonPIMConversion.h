//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonPIM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONPIM_TRANSFORMS_TRITONPIMCONVERSION_H_
#define TRITON_DIALECT_TRITONPIM_TRANSFORMS_TRITONPIMCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TritonPIMTypeConverter : public TypeConverter {
public:
  TritonPIMTypeConverter(MLIRContext *context, int numTasklets, int numDpus,
                         bool enableSourceRemat);
  int getNumTasklets() const { return numTasklets; }
  int getNumDpus() const { return numDpus; }

private:
  MLIRContext *context;
  int numTasklets;
  int numDpus;
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
