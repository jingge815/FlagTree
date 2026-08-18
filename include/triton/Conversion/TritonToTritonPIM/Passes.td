#ifndef TRITON_TO_TRITONPIM_CONVERSION_PASSES
#define TRITON_TO_TRITONPIM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonPIM : Pass<"convert-triton-to-pim", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonPIM";

    let description = [{
      Converts the Triton dialect into the TritonPIM dialect: the near-memory
      counterpart of `convert-triton-to-tritongpu`, and like it a partial
      conversion that also touches `Arith`, `Math`, `SCF` and `CF`.

      The work is to enhance every tensor type, and every `tt.ptr<tensor<>>`,
      with a `#pim.tasklet_tiled` encoding describing how the tensor is spread
      over a DPU's tasklets -- derived from the `num-tasklets` and `num-dpus`
      options. The arithmetic itself is left alone: `tt.dot`, `arith` and `scf`
      ops come through unchanged, since the mathematics of an operator does not
      depend on the memory architecture underneath it.

      Memory accesses are also left alone here. `tt.load` and `tt.store` survive
      this pass with encoded operands; turning them into explicit MRAM<->WRAM
      transfers is the job of `pim-explicit-dma`, mirroring how TTGIR keeps
      layout assignment separate from memory lowering.

      The hardware parameters are recorded on the module as `pim.num-dpus`,
      `pim.num-tasklets`, `pim.wram-bytes` and `pim.target`.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::pim::TritonPIMDialect"];

   let options = [
      Option<"target", "target",
             "std::string", /*default*/"\"\"",
             "the PIM target, e.g., pim:v1">,
      Option<"numDpus", "num-dpus",
             "int32_t", /*default*/"1",
             "number of DPUs on the device">,
      Option<"numTasklets", "num-tasklets",
             "int32_t", /*default*/"16",
             "number of tasklets per DPU">,
      Option<"wramBytes", "wram-bytes",
             "int32_t", /*default*/"65536",
             "WRAM budget per DPU, in bytes">,
      Option<"enableSourceRemat", "enable-source-remat",
             "bool", /*default*/"false",
             "enable trivial source rematerialization">,
   ];
}

#endif // TRITON_TO_TRITONPIM_CONVERSION_PASSES
