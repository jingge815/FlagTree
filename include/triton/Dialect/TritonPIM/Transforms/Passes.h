#ifndef TRITON_DIALECT_TRITONPIM_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONPIM_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

namespace mlir::triton::pim {

#define GEN_PASS_DECL
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonPIM/Transforms/Passes.h.inc"

} // namespace mlir::triton::pim

#endif // TRITON_DIALECT_TRITONPIM_TRANSFORMS_PASSES_H_
