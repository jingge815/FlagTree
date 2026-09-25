#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Conversion/TritonToTritonPIM/Passes.h"
#include "triton/Dialect/TritonPIM/Transforms/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_analysis(py::module &&m) {
  py::class_<mlir::ModuleAllocation>(m, "allocation", py::module_local())
      .def(py::init<mlir::ModuleOp>());
  py::class_<mlir::ModuleMembarAnalysis>(m, "membar", py::module_local())
      .def(py::init<mlir::ModuleAllocation *>())
      .def("run", &mlir::ModuleMembarAnalysis::run);
}

void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
  ADD_PASS_WRAPPER_0("print_ir", createPrintIRPass);
}

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createTritonCombineOps);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createTritonReorderBroadcast);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createTritonRewriteTensorPointer);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_descriptor_to_pointer",
                     createTritonRewriteTensorDescriptorToPointer);
  ADD_PASS_WRAPPER_0("add_loop_unroll", createTritonLoopUnroll);
  ADD_PASS_WRAPPER_0("add_triton_licm", createTritonLoopInvariantCodeMotion);
  ADD_PASS_WRAPPER_0("add_loop_aware_cse", createTritonLoopAwareCSE);
  ADD_PASS_OPTION_WRAPPER_4("add_convert_to_ttgpuir",
                            createConvertTritonToTritonGPU, const std::string &,
                            int, int, int);
}

void init_triton_passes_ttgpuir(py::module &&m) {
  using namespace mlir;
  using namespace mlir::triton::gpu;
  using namespace mlir::triton::instrument;
  ADD_PASS_WRAPPER_0("add_process_shared_memory_hint",
                     createTritonGPUProcessSharedMemoryHint); // flagtree hints
  ADD_PASS_WRAPPER_0("add_coalesce", createTritonGPUCoalesce);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createTritonGPUOptimizeThreadLocality);
  ADD_PASS_OPTION_WRAPPER_1("add_hoist_tmem_alloc",
                            createTritonGPUHoistTMEMAlloc, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_assign_latencies",
                            createTritonGPUAssignLatencies, int);
  ADD_PASS_WRAPPER_0("add_schedule_loops", createTritonGPUScheduleLoops);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline", createTritonGPUPipeline, int, bool);
  ADD_PASS_OPTION_WRAPPER_1("add_warp_specialize",
                            createTritonGPUAutomaticWarpSpecialization, int);
  ADD_PASS_WRAPPER_0("add_prefetch", createTritonGPUPrefetch);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul", createTritonGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_reorder_instructions",
                     createTritonGPUReorderInstructions);
  ADD_PASS_WRAPPER_0("add_f32_dot_tc", createTritonGPUF32DotTC);
  ADD_PASS_OPTION_WRAPPER_1("add_optimize_dot_operands",
                            createTritonGPUOptimizeDotOperands, bool);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     createTritonGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_reduce_data_duplication",
                     createTritonGPUReduceDataDuplication);
  ADD_PASS_WRAPPER_0("add_allocate_warp_groups",
                     createTritonGPUAllocateWarpGroups);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory", createAllocateSharedMemory);
  ADD_PASS_WRAPPER_0("add_allocate_global_scratch_memory",
                     createTritonGPUGlobalScratchAllocationPass);
  ADD_PASS_WRAPPER_0("add_combine_tensor_select_and_if",
                     createTritonGPUCombineTensorSelectAndIf);
  ADD_PASS_WRAPPER_0("add_optimize_accumulator_init",
                     createTritonGPUOptimizeAccumulatorInit);
  ADD_PASS_WRAPPER_0("add_fuse_nested_loops", createTritonGPUFuseNestedLoops);
  ADD_PASS_WRAPPER_0("add_coalesce_async_copy",
                     createTritonGPUCoalesceAsyncCopy);
  ADD_PASS_WRAPPER_0("add_concurrency_sanitizer",
                     createTritonInstrumentConcurrencySanitizer);
}

void init_triton_passes_convert(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_scf_to_cf", createSCFToControlFlowPass);
  ADD_PASS_WRAPPER_0("add_cf_to_llvmir", createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_index_to_llvmir", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_arith_to_llvmir", createArithToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_nvvm_to_llvm", createConvertNVVMToLLVMPass);
}

void init_triton_passes_llvmir(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_di_scope", mlir::createLLVMDIScope);
}

void init_gluon_passes(py::module &&m) {
  using namespace mlir;
  namespace gluon = mlir::triton::gluon;
  ADD_PASS_WRAPPER_0("add_resolve_auto_encodings",
                     gluon::createGluonResolveAutoEncodingsPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", gluon::createGluonCanonicalize);
  ADD_PASS_WRAPPER_0("add_inliner", gluon::createGluonInline);
}

void init_triton_passes_pim(py::module &&m) {
  using namespace mlir::triton;
  // Seven options, more than ADD_PASS_OPTION_WRAPPER_4 handles, so this one
  // is spelled out.
  m.def("add_convert_to_pim",
        [](mlir::PassManager &pm, const std::string &target, int numDpus,
           int numTasklets, int wramBytes, int64_t mramBytes, int dmaAlign,
           bool enableSourceRemat) {
          pm.addPass(createConvertTritonToTritonPIM(
              {target, numDpus, numTasklets, wramBytes, mramBytes, dmaAlign,
               enableSourceRemat}));
        },
        py::arg("pm"), py::arg("target"), py::arg("num_dpus") = 1,
        py::arg("num_tasklets") = 16, py::arg("wram_bytes") = 65536,
        py::arg("mram_bytes") = 4294967296LL, py::arg("dma_align") = 8,
        py::arg("enable_source_remat") = false);
  ADD_PASS_WRAPPER_0("add_explicit_dma", pim::createTritonPIMExplicitDMA);
  // The operator-level pipeline: fuse, then expand into phases, then check the
  // invariants that span more than one attribute. The graph compiler drives
  // these through `triton-opt` rather than in-process, so the wrappers are for
  // debugging and for callers that already hold a ModuleOp.
  ADD_PASS_WRAPPER_0("add_fuse_activation", pim::createTritonPIMFuseActivation);
  ADD_PASS_WRAPPER_0("add_expand_phases", pim::createTritonPIMExpandPhases);
  ADD_PASS_WRAPPER_0("add_verify_gml_contract",
                     pim::createTritonPIMVerifyGmlContract);
  // B 路走到 C 的唯一出口，缺了它调用方只能起子进程。
  ADD_PASS_WRAPPER_0("add_lower_to_emitc", pim::createTritonPIMLowerToEmitC);
  // full_m/full_n/full_k default to -1 ("not supplied", fall back to
  // structural IR inference) -- see TileToBudget.cpp's inferFullShape for
  // why a caller with the real launch-site M/N/K (e.g. genesim_bridge,
  // which captures them from the FlagGems kernel launch) needs to be able
  // to override the fallback per-dimension.
  m.def("add_tile_to_budget",
        [](mlir::PassManager &pm, int64_t fullM, int64_t fullN,
           int64_t fullK) {
          pm.addPass(pim::createTritonPIMTileToBudget({fullM, fullN, fullK}));
        },
        py::arg("pm"), py::arg("full_m") = -1, py::arg("full_n") = -1,
        py::arg("full_k") = -1);
}

void init_triton_passes(py::module &&m) {
  init_triton_analysis(m.def_submodule("analysis"));
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_convert(m.def_submodule("convert"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_triton_passes_pim(m.def_submodule("pim"));
  init_triton_passes_llvmir(m.def_submodule("llvmir"));
  init_gluon_passes(m.def_submodule("gluon"));
}
