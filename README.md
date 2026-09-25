# FlagTree 算子编译器（PIM 支路）

> 本文件是 flagOS 存算一体（PIM）编译体系中 **flagtree 算子编译器** 的代码索引。
> **维护约定：每次代码提交必须同步更新本文件与根目录 `claude.md`（目录索引、核心模块说明、提交记录）。**
> 最近更新：2026-09-21，对应 HEAD `c4dc3eb95`。

## 1. 仓库概述

flagtree 是 flagOS 存算一体编译器体系中的**算子编译器核心模块**：基于 FlagTree（Triton 3.5.1 多后端发行版）扩展出 **TritonPIM 方言与算子级编译流水线**，向上承接图编译器（`flagos-pim-compiler`，以 GML 图格式交接算子/相位/属性），向下输出算子级 PIM IR（`.pimir`）、GML 节点所需属性，以及可转 C/NumPy 执行的 EmitC 代码（对接 `opcompiler_bridge`、`genesim_bridge`）。GPU 主路径不受影响：PIM 从 TTIR 旁路（sidecar）生成，作为 NVIDIA 等后端编译的附加产物。

**核心适用人群**：算子编译器研发、PIM 硬件适配、GML 图格式与后端对接人员。

**提交者邮箱说明**：需求中的 `fengjg@ios.cn`，在仓库 git 历史中的实际提交邮箱为 **`fengjg@ios.ac.cn`**（作者 `fengjg`，共 **8** 个提交，全部集中在 PIM 支路，见 §5）。

## 2. 核心功能

1. **TritonPIM 方言（`pim`，命名空间 `mlir::triton::pim`）**
   - `#pim.tasklet_tiled`：张量在 DPU 内 tasklet 上的分布（`sizePerTasklet`/`taskletsPerDpu`/`dpusPerDevice`/`order`）。
   - `!pim.memdesc<shape x elem, #pim.wram|#pim.mram|#pim.l1|#pim.l2>`：DPU 本地/片上缓冲区描述。
   - 模块硬件契约属性：`pim.num-dpus`、`pim.num-tasklets`、`pim.wram-bytes`、`pim.mram-bytes`、`pim.dma-align`、`pim.l1-bytes`、`pim.l2-bytes`、`pim.target`、`pim.wram-bytes-used`、`pim.tile-m/n/k`、`pim.tile-wram-bytes`。
2. **TTIR → TritonPIM 转换（`convert-triton-to-pim`）**：给张量与指针张量补 PIM 布局编码，记录硬件参数；算术与内存访问保持原样（与 `convert-triton-to-tritongpu` 对称）。
3. **显式 DMA lowering（`pim-explicit-dma`）**：把 `tt.load/tt.store` 改写为 `pim.wram_alloc + pim.dma_load/dma_store + pim.barrier + pim.wram_load/store`；WRAM 分配外提出循环；窄指针分析标注 `contiguous_dim/elem_stride/base_arg`；统计 `pim.wram-bytes-used` 并校验预算。
4. **WRAM 预算感知切分（`pim-tile-to-budget`）**：为线性算子推断/校正 full M/N/K（可从真实 launch 参数经 `full-m/n/k` 覆盖），搜索能装进 WRAM 的 tile，写出 `pim.tile-*` 供下游 lowering 与成本模型使用。
5. **GML 图生成支撑（算子级算子集）**：`pim.matmul/conv/eltwise/lut/pool/quantize/dequantize/reduce_axis/rope/normalize/softmax` 等计算算子，`pim.buffer_alloc/buffer_copy/decompress_weight/param/transpose/reshape/split/concat/mask/kv_cache/split_heads` 等结构与数据算子；`#pim.datapath`、`#pim.quant_spec`、`#pim.kantor_block`、`#pim.act_spec`、`#pim.pool_spec` 等描述硬件配置。
6. **图交接融合（`pim-fuse-activation`）**：把 `pim.lut`（及尾随 `pim.pool`）折进 `pim.matmul/conv/eltwise` 的 `activation`/`fusedPool` 属性——目标图格式只允许激活内嵌在生产者节点中，不融合即不可表达。
7. **相位展开（`pim-expand-phases`）**：将不透明算子展开为硬件相位链：`pim.quantize` 动态量化 → 4 相（分组 absmax / 恒等表 / 倒数表 / 浮点转定点），`pim.softmax` → 5 相，`pim.rope` → 3 相。每相的结构化配置写成 ODS 属性：相位号、逻辑缓冲字节、引擎与相位间依赖在 `#pim.phase_spec`，定标通路在 `#pim.fpsu_spec`，逐元素乘与格式转换在 `#pim.kantor_spec`，查表窗口在 `pim.lut` 自身。每相还标出它跑在哪个硬件子块（`fpsu` / `kantor` / `pooling` / `activation` / `combiner`），不再统称 `cstl`——统称会让成本抽取分不出遍历类型。K 路 RoPE 的量化尾另盖 `#pim.contraction<form = flat>`。
8. **跨属性不变量校验（`pim-verify-gml-contract`）**：只读 pass，查单个 verifier 看不到的一致性——相位数在同一函数内唯一且从 0 连续、动态量化的相 1 与相 2 都读相 0（写成串行链会让倒数错 256 倍）、`pim.normalize` 不得携带定点/池化数据通路字段。
9. **EmitC/NumPy 可执行产物（`pim-lower-to-emitc`）**：两条入口。**tile 级**：单 DPU、任意 `pim.num-tasklets >= 1`，按 tasklet 数将 `tt.dot` 的 M 维拆成静态展开的行块。**算子级**：图编译器发的整算子级 IR（展开后的相位链）没有 DMA，按整张量降级——张量实参变裸指针、末尾追加一个输出指针（无人读的实参不占形参），逐算子发循环与查表。两条都经 `-convert-func-to-emitc` / `mlir-translate --mlir-to-cpp` 生成纯 C（供 `opcompiler_bridge/driver.py` 执行）。未识别的**算子级** `pim.*` 直接报错：静默跳过会让生成的 C 算成另一个函数，而 numpy 镜像『对上了』只是因为两边错得一样。
10. **Python / 后端接入**：`python/src/passes.cc` 暴露 `passes.pim.add_convert_to_pim/add_tile_to_budget/add_explicit_dma/add_fuse_activation/add_expand_phases/add_verify_gml_contract`；`python/triton/backends/pim_sidecar.py` 在 NVIDIA 后端 `make_ttir()` 末尾按需旁路生成 PIM IR；`bin/RegisterTritonDialects.h` 向 `triton-opt` 注册方言与 pass。
11. **PIM 目标校验放宽**：`lib/Dialect/Triton/IR/Traits.cpp::verifyTensorSize` 对带 `pim.target` 的模块取消“元素数必须为 2 的幂”限制（模型维度如 llama2 MLP 11008），其他后端行为不变。

## 3. 目录结构索引

粒度统一到**二级目录**；未列出的三级目录请按此表定位后再 `glob/grep`。
本表以 `include/triton`、`lib/Dialect`、`python/triton` 等**源码根**为一级目录，以其直接子目录（如 `Dialect/TritonPIM`、`Conversion/TritonToTritonPIM`）为二级目录展开，保证索引落到实际模块；仓库顶层无二级结构的目录（`bin`、`cmake`、`scripts` 等）单列。
优先级图例：**【核心模块】**【重要模块】【工具脚本】【测试用例】【备查】；临时/生成物标 **【可忽略】**。

### 3.1 PIM 算子编译器核心（高频检索）

| 优先级 | 一级目录 | 二级目录 | 核心功能 | 核心文件 | 适用场景 |
|--------|----------|----------|----------|----------|----------|
| 【核心模块】 | include/triton | Dialect/TritonPIM | PIM 方言/类型/属性/算子与 pass 的 TableGen 声明 | `IR/PIMDialect.td`、`IR/PIMOps.td`、`IR/PIMAttrDefs.td`、`IR/PIMTypes.td`、`IR/Dialect.h`、`Transforms/Passes.td` | 查 PIM IR 长什么样、属性/算子名、pass 选项 |
| 【核心模块】 | include/triton | Conversion/TritonToTritonPIM | `convert-triton-to-pim` pass 声明与选项 | `Passes.h`、`Passes.td` | 查转换 pass 选项（target/num-dpus/num-tasklets/wram-bytes 等） |
| 【核心模块】 | lib/Dialect | TritonPIM/IR | 方言实现：verifier、布局推导、类型/属性解析打印 | `Dialect.cpp`、`Ops.cpp`、`Types.cpp` | 算子 verifier 报错、layout 推导、memdesc 尺寸计算 |
| 【核心模块】 | lib/Dialect | TritonPIM/Transforms | 6 个 PIM pass 的完整实现 | `ExplicitDMA.cpp`（隐式访存→DMA）、`TileToBudget.cpp`（切 tile）、`FuseActivation.cpp`、`ExpandPhases.cpp`、`VerifyGmlContract.cpp`、`LowerPIMToEmitC.cpp`（tile 级 `tt.dot` 与算子级整张量两条降级） | 查算子 lowering、DMA 改写、tiling、相位、EmitC 生成 |
| 【核心模块】 | lib/Conversion | TritonToTritonPIM | TTIR→PIM 类型转换器、合法性与 pattern | `TritonToTritonPIMPass.cpp`、`TritonPIMConversion.cpp` | 查类型/布局转换规则、pattern 注册 |
| 【核心模块】 | bin | —（无二级目录） | `triton-opt` 等工具的方言/pass 注册 | `RegisterTritonDialects.h`、`CMakeLists.txt`、`triton-opt.cpp` | 报 “no registered dialect/pass”、新增 pass 注册 |
| 【核心模块】 | python/src | —（无二级目录） | Python 侧 IR/pass 绑定 | `passes.cc`（`passes.pim.*`）、`ir.cc`（方言加载、`ModuleOp.clone`） | 查 Python 调用 PIM pass 的入口/参数 |
| 【核心模块】 | python/triton | backends | PIM sidecar 旁路输出 | `pim_sidecar.py`、`compiler.py`、`driver.py` | 查 `.pimir` 生成流程、`FLAGTREE_*` 环境变量 |
| 【核心模块】 | third_party/nvidia | backend | NVIDIA 编译入口挂载 sidecar | `backend/compiler.py`（`make_ttir` 末尾调用 `emit_pim_ir`） | 查 GPU 编译与 PIM 旁路的衔接点 |
| 【核心模块】 | lib/Dialect | Triton/IR | Triton 公共 verifier（PIM 放宽 2 的幂限制） | `Traits.cpp`（`verifyTensorSize`/`isPIMModule`） | 查非 2 的幂 shape 报错与 PIM 例外 |
| 【测试用例】 | test/Dialect | TritonPIM | PIM 方言/pass 的 lit 测试（输入输出对照） | `ops.mlir`、`explicit_dma.mlir`、`tile_to_budget_*.mlir`、`lower_to_emitc*.mlir`、`operator_ops*.mlir`、`operator_shape_ops.mlir`、`fuse_activation.mlir`、`expand_phases*.mlir`、`phase_spec*.mlir`、`verify_gml_contract.mlir`、`tensor_size_pim.mlir` | 验证行为边界、抄 lit 命令 |
| 【测试用例】 | test/Conversion | —（无二级目录） | TTIR→PIM 转换测试 | `triton_to_pim.mlir` | 查布局转换期望结果 |
| 【备查】 | docs | —（无二级目录，PIM 部分） | PIM 实现技术文档（架构/文件/问题清单） | `triton-pim-support-20260818.md` | 快速理解 PIM 设计与现状 |

### 3.2 编译器公共核心（include/lib）

| 优先级 | 一级目录 | 二级目录 | 核心功能 | 核心文件 | 适用场景 |
|--------|----------|----------|----------|----------|----------|
| 【重要模块】 | include/triton | Analysis | 轴信息、别名、分配、membar 分析接口 | `AxisInfo.h`、`Alias.h`、`Allocation.h`、`Membar.h`、`Utility.h` | 查分析接口、别名/内存分析 |
| 【重要模块】 | include/triton | Conversion/TritonToTritonGPU、Conversion/TritonGPUToLLVM | GPU 路径转换 pass 声明 | `TritonToTritonGPU/Passes.h`、`TritonGPUToLLVM/Passes.h` | 对照 PIM 与 GPU 路径的差异 |
| 【重要模块】 | include/triton | Dialect/Triton、Dialect/TritonGPU、Dialect/TritonNvidiaGPU、Dialect/TritonInstrument、Dialect/Gluon | Triton 核心方言与 GPU/仪表/GLUON 方言声明 | `IR/*.td`、`IR/*.h`、`Transforms/Passes.td` | 查 TTIR/TTGIR 算子、属性、pass |
| 【重要模块】 | include/triton | Target/LLVMIR、Tools | LLVMIR 目标与工具头文件 | `Target/LLVMIR/Passes.h`、`Tools/Sys/GetEnv.hpp` | 查 LLVM 降级、环境变量工具 |
| 【重要模块】 | lib | Analysis、Dialect、Conversion、Instrumentation、Target、Tools | 上述头文件对应的 C++ 实现；含 TritonGPUToLLVM 主降级链路 | `lib/Analysis/*.cpp`、`lib/Conversion/TritonGPUToLLVM`、`lib/Target/LLVMIR` | 查 GPU 降级、分析实现 |
| 【重要模块】 | lib/flagtree、include/flagtree | Common | FlagTree 统一硬件抽象 | `Common/UnifiedHardware.cc`、`Common/UnifiedHardware.h` | 查统一硬件/后端抽象 |

### 3.3 Python 前端与运行时

| 优先级 | 一级目录 | 二级目录 | 核心功能 | 核心文件 | 适用场景 |
|--------|----------|----------|----------|----------|----------|
| 【重要模块】 | python/triton | compiler | 编译流水线、stage 调度、launcher 生成 | `compiler.py`、`code_generator.py`、`make_launcher.py` | 查 stage 流程、编译 cache、错误处理 |
| 【重要模块】 | python/triton | language | Triton 语言前端（语义、core、math、random） | `core.py`、`semantic.py`、`math.py`、`standard.py`、`target_info.py` | 查前端语义、内建函数 |
| 【重要模块】 | python/triton | runtime | JIT/launch/autotune/interpreter | `jit.py`、`driver.py`、`autotuner.py`、`interpreter.py` | 查 kernel 启动、autotune、解释执行 |
| 【重要模块】 | python/triton | tools、spec、extension、experimental | 工具、Ascend spec、扩展、Gluon/TLE 实验特性 | `tools/*`、`spec/ascend/*`、`experimental/gluon/*`、`experimental/tle/*` | 查布局工具、实验后端特性 |
| 【重要模块】 | python/triton_kernels | triton_kernels、tests、bench | 常用融合算子实现与基准（可用 PIM 目标复用） | `triton_kernels/*` | 查实际算子实现样例 |
| 【测试用例】 | python/test | unit、regression、backend、gluon、tle、kernel_comparison、microbenchmark | Python 侧单测/回归/后端测试 | `unit/*`、`regression/*`、`backend/*`、`tle/*` | 功能回归、后端对比 |
| 【工具脚本】 | python | scripts、setup_tools、examples、tutorials | 打包/离线构建工具、示例与教程 | `setup_tools/utils/*`、`scripts/*` | 离线构建、下载子模块（flir 等）、示例 |

### 3.4 测试与基准

| 优先级 | 一级目录 | 二级目录 | 核心功能 | 核心文件 | 适用场景 |
|--------|----------|----------|----------|----------|----------|
| 【测试用例】 | test | Conversion、Dialect、Triton、TritonGPU、TritonNvidiaGPU、Gluon、Hopper、NVWS、Proton、Analysis、LLVMIR、Tools、lib、include | MLIR lit 测试主体（PIM 测试见 §3.1） | 各目录 `*.mlir`；`test/lib/*` 测试用 pass | 复现低层 IR 行为、抄 RUN 命令 |
| 【测试用例】 | unittest | Analysis、Dialect、Tools | C++ 单测（GoogleTest，`check-triton-unit-tests`） | `Dialect/TritonGPU/*`、`Analysis/*` | C++ 层单测 |
| 【工具脚本】 | utils | —（无二级目录） | 测试校验生成等辅助脚本 | `generate-test-checks.py` | 更新 golden 测试 |

### 3.5 硬件后端与第三方

| 优先级 | 一级目录 | 二级目录 | 核心功能 | 核心文件 | 适用场景 |
|--------|----------|----------|----------|----------|----------|
| 【重要模块】 | third_party/nvidia | backend、include、lib、language、hopper、tools、unittest | NVIDIA 后端（PIM sidecar 挂载点在此） | `backend/compiler.py`、`backend/driver.py` | 查 CUDA 工具链、sidecar 挂载 |
| 【重要模块】 | third_party/amd | backend、include、lib、language、python、test、tools、unittest | AMD 后端 | `backend/compiler.py`、`lib/*` | AMD 适配 |
| 【重要模块】 | third_party/ascend | backend、include、lib、language、runtime、python、tutorials、unittest、bin | 昇腾后端 + DynamicCV 流水 | `lib/DynamicCVPipeline/*` | 昇腾适配/流水线 |
| 【重要模块】 | third_party/enflame | backend、language、python、triton_gcu、cmake | 燧原后端 | `backend/compiler.py` | 燧原适配 |
| 【重要模块】 | third_party/mthreads | backend、include、lib、language、python、plugin、bin | 摩尔线程后端 | `backend/compiler.py` | 摩尔线程适配 |
| 【重要模块】 | third_party/tsingmicro | backend、include、lib、language、runtime、python、scripts、examples、crt、bin | 清微后端 | `backend/compiler.py` | 清微适配 |
| 【重要模块】 | third_party/iluvatar、third_party/xpu | include/lib/python/backend 等 | 天数、XPU 后端 | `backend/compiler.py` | 对应硬件适配 |
| 【重要模块】 | third_party/tle、third_party/proton | dialect/dsa/test/utils；csrc/Dialect/proton/test | TLE 语言扩展方言；Proton 性能分析 | `tle/dialect/*`、`proton/csrc/*` | TLE 扩展、profiling |
| 【备查】 | third_party/flir、third_party/f2reduce | backend/include/lib/python/test | 外部子模块（离线构建时下载） | `setup_tools/utils/__init__.py` | 子模块版本/下载问题 |

### 3.6 构建、脚本与文档

| 优先级 | 一级目录 | 二级目录 | 核心功能 | 核心文件 | 适用场景 |
|--------|----------|----------|----------|----------|----------|
| 【工具脚本】 | cmake | —（无二级目录） | LLVM hash、工具链版本、CMake 辅助函数 | `llvm-hash.txt`、`AddTritonUnitTest.cmake`、`FindLLVM.cmake` | 查 LLVM 版本、构建配置 |
| 【工具脚本】 | scripts、dockerfiles | —（无二级目录） | 源码 LLVM 构建脚本、CI 镜像 | `build-llvm-project.sh` | 仅本地源码构建 LLVM 时；**官方安装走 §4 的 `0-install-flagtree.sh`（自带 LLVM）** |
| 【备查】 | docs | getting-started、programming-guide、python-api、backend、meetups、_templates | 官方文档（含 PIM 技术文档在 docs 根） | `getting-started/installation.rst`、`triton-pim-support-20260818.md` | 安装、教程 |
| 【备查】 | documents | tle | TLE 文档（含 DSA 用法） | `tle/*.md` | TLE/DSA 对接 |
| 【可忽略】 | build、.superpowers | — | CMake 构建产物 / 内部流程临时目录 | — | 不参与代码检索 |

## 4. 构建与运行（以 `flagOS-installers/0-install-flagtree.sh` 为准）

**唯一官方构建入口是同级目录 `flagOS-installers/0-install-flagtree.sh`**，负责准备自带工具链、构建 wheel 并安装到独立 prefix；**不要自造构建流程**。环境要求：**Ubuntu 22.04 + x86_64**，无需 root；GPU 可选（纯 CPU 机器同样可编译，算子编译走 `opcompiler_bridge/cpu_host.py` 前端路径）。

### 4.1 一键安装 / 重装（C++ 或 Python 改动后重跑）

```bash
cd /path/to/flagOS-installers

# 默认：源码 ./FlagTree，安装到 ../flagOS-installed/flagTree
bash 0-install-flagtree.sh

# 常用参数
bash 0-install-flagtree.sh --source-dir /path/to/FlagTree \
                           --prefix /path/to/flagOS-installed/flagTree --max-jobs 8
bash 0-install-flagtree.sh --skip-test        # 跳过安装后验证

# 源码目录带未提交改动（本地开发）时必须显式放行，否则脚本拒绝执行：
ALLOW_DIRTY_FLAGTREE_SOURCE=1 bash 0-install-flagtree.sh --source-dir /path/to/FlagTree
```

脚本关键行为：平台与命令检查 → 下载/解压自带 **Python 3.10.20、LLVM `llvm-7d5de303`、Triton 构建依赖**（`triton-home/` + `nvidia-toolchain-12.8/`：ptxas/cuobjdump/nvdisasm/cudart/libdevice）与 sysroot 开发头（zlib/libxml2）→ 安装构建依赖（cmake 3.31.10、ninja 1.13.0、pybind11 3.0.4、lit 18.1.8、numpy 1.26.4、pytest 8.3.5、torch 2.7.1+cu128）→ 校验 FlagTree 源码与 `third_party/flir`（pin `165f387b28e3`）→ `pip wheel . --no-build-isolation --no-deps` 构建（`TRITON_BUILD_PROTON=OFF`、`TRITON_BUILD_UT=OFF`、`TRITON_BUILD_WITH_CCACHE=OFF`、`TRITON_PARALLEL_LINK_JOBS=1`、`MAX_JOBS`、`TRITON_APPEND_CMAKE_ARGS` 注入 sysroot zlib/libxml2）→ `pip install --force-reinstall --no-deps` → 写环境脚本 → `import torch, triton` + `examples/matmul_sm80.py` 验证。

### 4.2 安装后目录（`$PREFIX`）

| 路径 | 作用 |
|------|------|
| `env-flagtree.sh` | **使用/增量编译前必须 `source`**：导出 PATH、LLVM_SYSPATH、`TRITON_BUILD_DIR`、`TRITON_DUMP_DIR`、ptxas/cudart 等 |
| `python` → `python-3.10.20` | 自带 Python（含 cmake/ninja/lit/pytest） |
| `llvm-7d5de303/`、`sysroot/`、`nvidia-toolchain-12.8/`、`triton-home/` | LLVM、开发头、CUDA 工具链、构建依赖 |
| `build/flagtree-cmake/`（`$TRITON_BUILD_DIR`） | CMake 构建目录，`bin/triton-opt`、`bin/triton-lsp` 等在此 |
| `wheels/`、`cache/`、`mlir-dumps/`、`triton-stage-dumps/` | wheel、编译 cache、MLIR/阶段 dump |
| `examples/matmul_sm80.py` | 安装验证示例 |

### 4.3 使用与测试

```bash
source /path/to/flagOS-installed/flagTree/env-flagtree.sh    # 每个 shell 必须先执行

# PIM 常用命令（triton-opt 来自前缀构建目录）
TRITON_OPT=$TRITON_BUILD_DIR/bin/triton-opt
$TRITON_OPT test/Dialect/TritonPIM/ops.mlir -pim-explicit-dma
$TRITON_OPT in.mlir -convert-triton-to-pim='target=pim:v1 num-tasklets=16 wram-bytes=65536' \
  -pim-tile-to-budget -pim-explicit-dma          # 顺序：tile-to-budget 必须在 explicit-dma 之前
$TRITON_OPT in.mlir -pim-fuse-activation -pim-expand-phases -pim-verify-gml-contract
$TRITON_OPT in.mlir -pim-lower-to-emitc -convert-func-to-emitc   # 再接 mlir-translate --mlir-to-cpp

# lit 测试（含 test/Dialect/TritonPIM/*），在源码目录执行
make test-lit

# 随 GPU 编译旁路生成 PIM IR（NVIDIA 后端）
FLAGTREE_EMIT_PIM=1 python kernel.py    # 产物：$TRITON_DUMP_DIR/pim-<hash>/<kernel>.pimir
```

**增量编译**：source 环境脚本后在源码目录 `make triton-opt`（或 `ninja -C $TRITON_BUILD_DIR triton-opt`），仅适用于 C++/lit 快速迭代；**Python 侧改动（如 `pim_sidecar.py`）需重跑 `0-install-flagtree.sh`，或重新 `pip install --force-reinstall --no-deps <prefix>/wheels/flagtree-*.whl`** 才会进入已安装包。

**Sidecar 环境变量**（`python/triton/backends/pim_sidecar.py`）：`FLAGTREE_EMIT_PIM`（默认 0，置 1 开启）、`FLAGTREE_PIM_TARGET`（`pim:v1`）、`FLAGTREE_PIM_NUM_DPUS`（1）、`FLAGTREE_PIM_NUM_TASKLETS`（16）、`FLAGTREE_PIM_WRAM_BYTES`（65536）、`FLAGTREE_PIM_MRAM_BYTES`（8 GiB）、`FLAGTREE_PIM_DMA_ALIGN`（8）。

## 5. 核心提交变更记录（提交者 `fengjg@ios.ac.cn`，时间倒序）

| commit | 日期 | 提交标题 | 核心功能说明 | 影响模块/目录 |
|--------|------|----------|--------------|----------------|
| `c4dc3eb95` | 2026-09-19 | fix invalid.mlir bug | 修正 `atomic_cas` 类型不匹配的 expected-error 文本顺序（`cmp type matches ptr type` / `value type matches ptr type`），恢复 lit 测试通过 | `test/Triton/invalid.mlir` |
| `62e28b5f0` | 2026-09-19 | add flagtree to glm generate | 新增 `pim-expand-phases`：将 `pim.quantize`/`pim.softmax`/`pim.rope` 展开为 4/5/3 相硬件相位链并标注 `pim.phase-bytes`；PIM 模块放宽 tensor 元素数 2 的幂限制（按 `pim.target` 判定，不影响其他后端）；补充通过扩张后的算子/属性定义与测试 | `lib/Dialect/TritonPIM/Transforms/ExpandPhases.cpp`、`include/triton/Dialect/TritonPIM/IR/PIMAttrDefs.td`、`Transforms/Passes.td`、`lib/Dialect/Triton/IR/Traits.cpp`、`test/Dialect/TritonPIM/{expand_phases,expand_phases_negative,tensor_size_pim}.mlir` |
| `0ed52a321` | 2026-09-15 | support glm generate | 新增 GML 图所需的算子级算子集与属性：计算算子（matmul/conv/eltwise/lut/pool/quantize/dequantize/reduce_axis/rope/normalize/softmax）、结构算子（buffer_alloc/buffer_copy/decompress_weight/param/transpose/reshape/split/concat/mask/kv_cache/split_heads）、硬件配置属性（datapath/kantor_block/quant_spec/act_spec/pool_spec）与 L1/L2 内存；新增 `pim-fuse-activation`（激活内嵌进生产者节点，图格式硬性要求）；补齐 verifier 与大量正反测试 | `include/triton/Dialect/TritonPIM/IR/{PIMOps.td,PIMAttrDefs.td,PIMTypes.td,Dialect.h}`、`lib/Dialect/TritonPIM/IR/{Dialect.cpp,Ops.cpp,Types.cpp}`、`lib/Dialect/TritonPIM/Transforms/FuseActivation.cpp`、`test/Dialect/TritonPIM/{operator_ops,operator_ops_negative,operator_shape_ops,fuse_activation}.mlir` |
| `dc35b24df` | 2026-08-30 | genesim opt for pim m,n,k, gemv placement, add tiling pass | `pim-tile-to-budget` 增加 `full-m/n/k` 选项，允许从真实 launch 参数覆盖结构性 M/N/K 推断（适配 FlagGems `linear_kernel` 这类网格切分 + 运行期标量参数的 kernel）；优化 GEMV 场景 tile/放置；sidecar 在 IR 含 `tt.dot` 时按 `tile-to-budget → explicit-dma` 正确顺序执行 | `lib/Dialect/TritonPIM/Transforms/TileToBudget.cpp`、`include/triton/Dialect/TritonPIM/Transforms/Passes.td`、`python/src/passes.cc`、`python/triton/backends/pim_sidecar.py` |
| `80757547f` | 2026-08-28 | suport loop tiling for pim | 新增 WRAM 预算感知 tiling pass `pim-tile-to-budget`（校验线性算子 tile、搜索可装进 WRAM 的切分、写 `pim.tile-m/n/k/wram-bytes`）；补齐模块属性 `pim.mram-bytes`/`pim.dma-align` 及默认值；扩张 convert pass 选项与 DMA/emitc 路径；新增 4 个 tiling 测试 | `lib/Dialect/TritonPIM/Transforms/TileToBudget.cpp`、`include/triton/Dialect/TritonPIM/IR/Dialect.h`、`Transforms/Passes.td`、`lib/Conversion/TritonToTritonPIM/TritonToTritonPIMPass.cpp`、`ExplicitDMA.cpp`、`LowerPIMToEmitC.cpp`、`python/src/passes.cc`、`test/Dialect/TritonPIM/tile_to_budget_*.mlir` |
| `cc37b94ef` | 2026-08-27 | suport multi tasklets for pim | 将 `LowerPIMSingleTasklet.cpp` 重构为 `LowerPIMToEmitC.cpp`，支持任意 `pim.num-tasklets >= 1`：把 `tt.dot` 的 M 维按 tasklet 拆成静态展开的顺序行块（UPMEM 惯例），`tid` 在编译期展开、无运行期 tasklet 值；扩大 pass 描述与测试 | `lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp`、`Transforms/Passes.td`、`bin/CMakeLists.txt`、`test/Dialect/TritonPIM/{lower_to_emitc,lower_to_emitc_negative}.mlir` |
| `207baa875` | 2026-08-26 | Connection from Graph Compiler to Operator Compiler | 打通图编译器到算子编译器的执行链路：新增 `pim-lower-to-emitc`（单 tasklet 初版），把 PIM IR 降为 `emitc.*` 供图编译器 bridge 生成 numpy 可执行 C；`triton-opt` 链接 EmitC 相关库并注册 | `lib/Dialect/TritonPIM/Transforms/LowerPIMSingleTasklet.cpp`、`Transforms/Passes.td`、`bin/CMakeLists.txt`、`bin/RegisterTritonDialects.h` |
| `eebe1b60b` | 2026-08-18 | support pim mlir | PIM 支路首个提交：新增 TritonPIM 方言（`#pim.tasklet_tiled` 布局、`!pim.memdesc` 类型、WRAM/DMA/barrier 算子）、`convert-triton-to-pim` 转换 pass、`pim-explicit-dma` 显式 DMA pass；接入 Python 绑定、NVIDIA sidecar 与 `triton-opt` 注册；`.gitignore` 放行 `pim_sidecar.py`；附 977 行实现技术文档 | `include/triton/{Dialect/TritonPIM,Conversion/TritonToTritonPIM}`、`lib/{Dialect/TritonPIM,Conversion/TritonToTritonPIM}`、`test/{Conversion/triton_to_pim,Dialect/TritonPIM/ops,explicit_dma}.mlir`、`python/src/{ir,passes}.cc`、`python/triton/backends/pim_sidecar.py`、`third_party/nvidia/backend/compiler.py`、`docs/triton-pim-support-20260818.md` |

## 6. 文档维护规则

1. **任何代码提交必须同步更新本文件与 `claude.md`**：
   - 新增/删除/移动目录或核心文件 → 更新 §3 目录索引与 `claude.md` 目录速查表；
   - 新增/修改 PIM pass、算子、属性、环境变量 → 更新 §2 核心功能与对应目录行；
   - 功能提交 → 在 §5 追加一行（时间倒序），并同步 `claude.md` 的“关键提交变更速览”；
   - 修改构建/测试命令 → 更新 §4。
2. **索引粒度固定为二级目录**，不深入三级；核心文件必须写实际路径。
3. **不得虚构**：目录说明、pass 行为、提交记录均须与代码和 `git log` 一致；提交者以 git 实际邮箱 `fengjg@ios.ac.cn` 为准。
4. 纯格式化、注释类提交可只在 `claude.md` 中省略，但 §5 仍须保留一行。
