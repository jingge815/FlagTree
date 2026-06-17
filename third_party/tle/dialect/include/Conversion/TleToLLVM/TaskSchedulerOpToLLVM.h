#pragma once

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace mlir::triton::tle {

void populateTaskSchedulerOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    unsigned benefit = 1);

} // namespace mlir::triton::tle
