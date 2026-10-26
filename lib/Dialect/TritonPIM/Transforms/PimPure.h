// 三个通用优化共用的纯性判定。
//
// 上游 CSE 与 DCE 只认 `Pure` 标记，而 pim.matmul、pim.eltwise、pim.lut
// 都没标，所以那两个 pass 对它们不生效。这里改按性质表判定：没有内存
// 副作用的 pim.* 算子视为纯算子。

#ifndef TRITON_DIALECT_TRITONPIM_TRANSFORMS_PIMPURE_H_
#define TRITON_DIALECT_TRITONPIM_TRANSFORMS_PIMPURE_H_

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonPIM/IR/Dialect.h"

namespace mlir::triton::pim {

// 带内存副作用的算子（dma、kv_cache、wram_alloc）返回 false。
inline bool isPurePim(Operation *op) {
  // pim 方言没声明内存副作用接口的算子，isMemoryEffectFree 一律返回 false，
  // 那会把 pim.eltwise 这类纯算子也判成有副作用，所以没声明的视为纯。
  // 声明了的按接口报的效果判定：标 Pure 的 tasklet_id、reshape 也走这个接口。
  if (llvm::isa_and_nonnull<TritonPIMDialect>(op->getDialect()))
    return !llvm::isa<MemoryEffectOpInterface>(op) || isMemoryEffectFree(op);
  // 其它方言按自己声明的 Pure 判定，这样 A 路 TTIR 里的 arith 也纳入。
  return isPure(op);
}

// 整算子函数：第一个参数是整块张量或片上缓冲，由图编译器发射，不是从 TTIR 转来的。
// A 路内核的第一个参数是标量指针，或元素类型为指针的张量（指针张量也不是整块张量）。
// 这类函数里没人用的张量结果是内核输出，`pim-lower-to-emitc` 按同一判据分流。
inline bool isOperatorLevelFunc(triton::FuncOp func) {
  if (!func || func.getNumArguments() == 0)
    return false;
  Type first = func.getArgument(0).getType();
  if (isa<MemDescType>(first))
    return true;
  auto tensor = dyn_cast<RankedTensorType>(first);
  return tensor && !isa<triton::PointerType>(tensor.getElementType());
}

} // namespace mlir::triton::pim

#endif // TRITON_DIALECT_TRITONPIM_TRANSFORMS_PIMPURE_H_
