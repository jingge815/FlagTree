#ifndef TRITON_CONVERSION_TRITONTOTRITONPIM_PASSES_H
#define TRITON_CONVERSION_TRITONTOTRITONPIM_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DECL
#include "triton/Conversion/TritonToTritonPIM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonPIM/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_CONVERSION_TRITONTOTRITONPIM_PASSES_H
