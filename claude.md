# claude.md — flagtree 算子编译器 AI 索引

> 极简检索索引，**修改代码后必须同步更新本文件与 `README.md`**。维护提交行见 §4。

## 1. 仓库核心定位

flagtree = flagOS 存算一体（PIM）体系的**算子编译器**：基于 FlagTree/Triton 3.5.1，扩展 `pim` 方言与算子级编译流水线；上游对接图编译器（`flagos-pim-compiler`，GML 图交接），下游输出 `.pimir` / GML 节点属性 / EmitC→C（`opcompiler_bridge`）。GPU 主路径不变，PIM 从 TTIR 旁路。提交者实际邮箱：**fengjg@ios.ac.cn**（8 commits）。

## 2. 目录速查表

| 目录 | 核心功能 | 检索关键词 | 场景 |
|------|----------|------------|------|
| `include/triton/Dialect/TritonPIM/IR/` | PIM 方言/属性/类型/算子声明（TableGen） | `tasklet_tiled` `memdesc` `datapath` `quant_spec` `phase_spec` `fpsu_spec` `kantor_spec` `weight_binding` `contraction` `global_pool` `gather` | 查 PIM IR 定义、属性/算子名 |
| `include/triton/Dialect/TritonPIM/Transforms/` | PIM pass 声明与选项 | `pim-explicit-dma` `pim-tile-to-budget` `pim-fuse-activation` `pim-expand-phases` `pim-verify-gml-contract` `pim-lower-to-emitc` | 查 pass 选项/描述 |
| `include/triton/Conversion/TritonToTritonPIM/` | `convert-triton-to-pim` 声明 | `Passes.td` `target` `num-dpus` `num-tasklets` `wram-bytes` | 查转换 pass 选项 |
| `lib/Dialect/TritonPIM/IR/` | verifier、layout 推导、类型实现 | `Dialect.cpp` `Ops.cpp` `Types.cpp` `InferLayout` `WRAMAllocOp::verify` | 报错定位、布局推导、memdesc |
| `lib/Dialect/TritonPIM/Transforms/` | 6 个 PIM pass 实现 | `ExplicitDMA` `TileToBudget` `FuseActivation` `ExpandPhases` `VerifyGmlContract` `LowerPIMToEmitC` | **PIM 核心改动高频区** |
| `lib/Conversion/TritonToTritonPIM/` | TTIR→PIM 类型转换/pattern | `TritonPIMTypeConverter` `GenericOpPattern` `applyPartialConversion` | 布局/类型转换问题 |
| `bin/RegisterTritonDialects.h` | triton-opt 方言/pass 注册 | `registerTritonPIMPasses` `TritonPIMDialect` `EmitC` | pass 未注册、新增 pass |
| `python/src/passes.cc` | Python pass 绑定 | `init_triton_passes_pim` `add_convert_to_pim` `add_tile_to_budget` `add_explicit_dma` | Python 调 PIM pass |
| `python/src/ir.cc` | 方言加载、`ModuleOp.clone` | `load_dialects` `clone` | Python 侧 IR/克隆 |
| `python/triton/backends/pim_sidecar.py` | GPU 编译旁路生成 PIM IR | `FLAGTREE_EMIT_PIM` `make_pimir` `emit_pim_ir` `.pimir` | `.pimir` 生成/环境变量 |
| `third_party/nvidia/backend/compiler.py` | sidecar 挂载点（`make_ttir` 末尾） | `is_enabled` `emit_pim_ir` | GPU 编译耦合问题 |
| `lib/Dialect/Triton/IR/Traits.cpp` | 放宽 PIM 模块 2 的幂 tensor 限制 | `verifyTensorSize` `isPIMModule` `pim.target` | 非 2 的幂 shape 报错 |
| `test/Dialect/TritonPIM/` | PIM lit 测试（行为边界真源） | `ops` `explicit_dma` `tile_to_budget` `lower_to_emitc` `operator_ops` `fuse_activation` `expand_phases` `phase_spec` `verify_gml_contract` | 抄 lit 命令/看期望 IR |
| `test/Conversion/triton_to_pim.mlir` | 转换测试 | `convert-triton-to-pim` | 转换结果对照 |
| `docs/triton-pim-support-20260818.md` | PIM 设计与现状（含问题清单） | `技术文档` `mermaid` | 快速理解整体设计 |
| `include/triton/{Analysis,Dialect/Triton*,Conversion/TritonGPUToLLVM}`、`lib/{Analysis,Dialect/Triton*,Conversion/TritonGPUToLLVM,Target}` | Triton 公共核心/GPU 路径 | `TTIR` `TTGIR` `AxisInfo` | 对照 GPU 路径、公共基础设施 |
| `python/triton/{language,compiler,runtime}` | Triton 前端/流水线/运行时 | `semantic.py` `compiler.py` `jit.py` | 前端/启动问题 |
| `third_party/{nvidia,amd,ascend,enflame,mthreads,tsingmicro,iluvatar,xpu,tle,proton}` | 各硬件后端/TLE/Proton | `backend/compiler.py` | 后端适配问题 |
| `test/`、`unittest/`、`python/test/` | lit/C++/Python 测试 | `check-triton-lit-tests` | 回归验证 |
| `build/`、`.superpowers/`、`python/__pycache__/` | **【可忽略】** 生成物/临时目录 | — | 不检索 |

## 3. 核心模块与入口

**IR 定义（TableGen，改前先看）**
- `include/triton/Dialect/TritonPIM/IR/PIMOps.td`：存储类算子 `pim.tasklet_id/dpu_id/wram_alloc/dma_load/dma_store/wram_load/wram_store/barrier/convert_layout`；GML 算子 `pim.quantize/dequantize/matmul/conv/lut/eltwise/pool/reduce_axis/rope/normalize/softmax`；结构算子 `pim.buffer_alloc/buffer_copy/decompress_weight/param/transpose/reshape/split/concat/mask/kv_cache/split_heads`。
- `PIMAttrDefs.td`：`tasklet_tiled`、`quant_spec`（含 role/fpDtype/range）、`kantor_block`、`kantor_spec`、`datapath`、`act_spec`、`pool_spec`、`phase_spec`、`fpsu_spec`、`weight_binding`、`contraction`、FunctionalUnit(**9 值**：nmu/vpu/cstl/dma 序数冻结 + fpsu/kantor/pooling/activation/combiner)、FpsuMode、GlobalPoolKind、PoolDType、WeightFormat/Role、QuantRole、ContractionForm。
- `PIMTypes.td` / `PIMDialect.td`：`!pim.memdesc`（wram/mram/l1/l2）、方言定义。
- 模块属性名常量集中在 `lib/Dialect/TritonPIM/IR/Dialect.h`（`AttrNumTaskletsName` 等 + `lookupNumTasklets`/`maybeLookupWramBytes`）。

**核心实现**
| 位置 | 关键函数/类 | 作用 |
|------|-------------|------|
| `lib/Conversion/.../TritonToTritonPIMPass.cpp` | `TritonPIMTypeConverter`、`TritonPIMConversionTarget`、`TritonDotPattern` 等 | TTIR→PIM 布局转换 |
| `lib/Dialect/TritonPIM/Transforms/ExplicitDMA.cpp` | `rewriteLoad`、`rewriteStore`、`createHoistedAlloc`、`analyzePointers` | tt.load/store→显式 DMA；标注 `contiguous_dim/elem_stride/base_arg`；统计 `pim.wram-bytes-used` |
| `TileToBudget.cpp` | `inferFullShape`、`inferVisibleTile`、`searchTile`、`fitsBudget` | 搜索装进 WRAM 的 tile；写 `pim.tile-m/n/k` |
| `FuseActivation.cpp` | `fuse`、`activationSpecOf`、`poolSpecOf` | `pim.lut`(+`pim.pool`) 折进 matmul/conv/eltwise 的 `activation`/`fusedPool` |
| `ExpandPhases.cpp` | `expandQuantize`(4相)、`expandSoftmax`(5相)、`expandRope`(3相)、`setPhaseSpec` | 展开硬件相位链，结果写成 `#pim.phase_spec` / `#pim.fpsu_spec` / `#pim.kantor_spec` |
| `VerifyGmlContract.cpp` | 相位数连续性与唯一性、DQ 相 1/2 扇出读相 0、normalize 不得带 datapath | 跨属性不变量校验（只读，不改 IR） |
| `LowerPIMToEmitC.cpp` | `makeView`、`resolveBaseArg`、`kF16Helpers` | 单 DPU、num-tasklets≥1 行块展开，降 emitc |
| `lib/Dialect/TritonPIM/IR/Dialect.cpp` | `TritonPIMInferLayoutInterface` | trans/reduce/expand_dims/reshape 等布局推导 |
| `python/triton/backends/pim_sidecar.py` | `make_pimir`、`emit_pim_ir`、`_default_output_path` | 旁路管线：clone→convert→(含 tt.dot 才)tile→explicit-dma→写 `.pimir` |

**编译流程关键节点**
1. 图编译器 bridge（外部仓库）发算子级 PIM IR（`pim.matmul` 等）→ `-pim-fuse-activation` → `-pim-expand-phases`
2. TTIR 路径：`-convert-triton-to-pim` → `-pim-tile-to-budget`（**必须在前**）→ `-pim-explicit-dma`
3. 执行产物：`-pim-lower-to-emitc -convert-func-to-emitc` → `mlir-translate --mlir-to-cpp`
4. GPU sidecar：`FLAGTREE_EMIT_PIM=1` 时 NVIDIA `make_ttir` 末尾 clone TTIR 走 2，产物 `$TRITON_DUMP_DIR/pim-<hash>/<kernel>.pimir`
5. Python pass 入口：`passes.pim.add_convert_to_pim(pm, target, num_dpus=1, num_tasklets=16, wram_bytes=65536, mram_bytes=..., dma_align=8)` / `add_tile_to_budget(pm, full_m=-1, full_n=-1, full_k=-1)` / `add_explicit_dma(pm)`

**构建/运行（唯一入口：`flagOS-installers/0-install-flagtree.sh`，勿自造流程）**
- 安装/重装：`bash 0-install-flagtree.sh [--source-dir <repo>] [--prefix <dir>] [--max-jobs N] [--skip-test]`；源码有未提交改动需 `ALLOW_DIRTY_FLAGTREE_SOURCE=1`
- 使用前每个 shell 必须 `source <prefix>/env-flagtree.sh`（PATH/LLVM_SYSPATH/TRITON_BUILD_DIR/TRITON_DUMP_DIR 等）
- triton-opt：`$TRITON_BUILD_DIR/bin/triton-opt`；lit：源码目录 `make test-lit`；C++ 增量：`ninja -C $TRITON_BUILD_DIR triton-opt`；**Python 改动需重跑安装脚本或重装 `<prefix>/wheels/flagtree-*.whl`**
- prefix 默认 `flagOS-installers/../flagOS-installed/flagTree`；脚本自带 Python 3.10.20 + LLVM `llvm-7d5de303`，构建 `TRITON_BUILD_PROTON=OFF`/`TRITON_BUILD_UT=OFF`

## 4. 关键提交变更速览（fengjg@ios.ac.cn，按模块）

| 模块 | commit | 日期 | 核心作用 |
|------|--------|------|----------|
| PIM 方言基础 | `eebe1b60b` | 2026-08-18 | 建 TritonPIM 方言/布局/类型，convert + explicit-dma，Python/sidecar/NVIDIA 接入 |
| 图编译器连接 | `207baa875` | 2026-08-26 | 新增 `pim-lower-to-emitc`（单 tasklet），打通 GML bridge 生成 C |
| 图编译器连接 | `cc37b94ef` | 2026-08-27 | 改多 tasklet：`LowerPIMSingleTasklet`→`LowerPIMToEmitC`，M 维行块展开 |
| 切 tile | `80757547f` | 2026-08-28 | 新增 `pim-tile-to-budget`，WRAM 预算切分，mram/dma-align/tile-* 属性 |
| 切 tile | `dc35b24df` | 2026-08-30 | tile pass 加 `full-m/n/k` 覆盖 + GEMV 放置；sidecar 固定 pass 顺序 |
| GML 生成 | `0ed52a321` | 2026-09-15 | 算子级算子集/属性（datapath/kantor/quant_spec）+ `pim-fuse-activation` + L1/L2 |
| GML 生成 | `62e28b5f0` | 2026-09-19 | 新增 `pim-expand-phases`（DQ4/Softmax5/RoPE3）；PIM 模块放宽 2 的幂限制 |
| 测试修复 | `c4dc3eb95` | 2026-09-19 | 修正 `test/Triton/invalid.mlir` atomic_cas expected-error 文本 |

## 5. AI 检索指引（场景 → 目录）

- 算子 lowering / DMA 改写 → `lib/Dialect/TritonPIM/Transforms/ExplicitDMA.cpp` + `test/Dialect/TritonPIM/explicit_dma.mlir`
- GML/算子级算子定义、属性含义 → `include/triton/Dialect/TritonPIM/IR/PIMOps.td`、`PIMAttrDefs.td` + `operator_*.mlir`
- 相位展开（量化/softmax/rope）→ `ExpandPhases.cpp` + `expand_phases*.mlir`
- 相位/定标属性含义、跨属性不变量 → `phase_spec.mlir`、`verify_gml_contract.mlir` + `VerifyGmlContract.cpp`
- 权值通路（int4/int8 绑权、内容指纹）→ `operator_ops.mlir` 的 `weight_binding` 段 + `PIMAttrDefs.td::WeightBindingAttr`
- 激活融合/图格式约束 → `FuseActivation.cpp` + `fuse_activation.mlir`
- WRAM 超预算 / tiling / M/N/K 推断 → `TileToBudget.cpp` + `tile_to_budget_*.mlir`
- EmitC/C 产物、numpy 执行 → `LowerPIMToEmitC.cpp` + `lower_to_emitc*.mlir`（外部 `opcompiler_bridge/driver.py`）
- 算子级（整张量）内核降到 C、相位链怎么变成循环 → `lower_to_emitc_ops*.mlir` + `LowerPIMToEmitC.cpp` 的 operator-level 段
- 分组归约 / 定点定标 / 逐元素乘 / 查表算子 → `PIMOps.td` 的 `global_pool` / `fpsu_scale` / `kantor` / `gather` + `operator_ops.mlir`
- 组反量化累加（w4a8 投影的组边界）→ `PIMAttrDefs.td::DatapathAttr` 的 `groupDequantAccum` / `groupSize` + `operator_ops_negative.mlir` + `lower_to_emitc_ops.mlir` 的 `w4a8` 段
- 缓存读写模式（散写/区间写/读、i16 行号、更新侧定点定标）→ `PIMOps.td` 的 `kv_cache` + `PIMAttrDefs.td::KvMode` + `operator_ops_negative.mlir` 的 kv 段
- 相位地址卡与重排用途（`transpose_purpose` / `onthefly`）、合并模式 → `PIMAttrDefs.td::TransposePurposeAttr` / `CombineModeAttr` / `OnTheFlyAttr` + `operator_ops.mlir`
- 扩展位与元素类型的对应（`pim.convert`）、模块版本属性与 `isPhased` → `PIMOps.td::ConvertOp` + `Dialect.h::AttrRtlVersionName` + `VerifyGmlContract.cpp::checkPhasedMarker`
- 融合在图格式里的两种写法（嵌套块 vs 扁平标志）→ `PIMAttrDefs.td::ContractionAttr` + `FuseActivation.cpp::namedContraction` + `expand_phases.mlir` 的 RoPE 段
- 相位跑在哪个硬件子块 → `FunctionalUnit` 九值 + `ExpandPhases.cpp` 各相的 unit 实参
- `.pimir` 输出、`FLAGTREE_*` 环境变量 → `python/triton/backends/pim_sidecar.py`
- Python 侧 pass 调用/参数 → `python/src/passes.cc::init_triton_passes_pim`
- 编译报错 “no registered dialect/pass” → `bin/RegisterTritonDialects.h`
- 非 2 的幂 shape 被拒 → `lib/Dialect/Triton/IR/Traits.cpp::verifyTensorSize`
- verifier/layout 推导 → `lib/Dialect/TritonPIM/IR/{Ops,Dialect,Types}.cpp`
- pass 顺序问题（如 tiling 晚于 DMA）→ `pim_sidecar.py::make_pimir` 注释 + lit 的 RUN 行
- 构建/lit/安装失败 → `flagOS-installers/0-install-flagtree.sh`（唯一构建入口）、`<prefix>/env-flagtree.sh`、`Makefile`（`test-lit`）、`bin/CMakeLists.txt`
- 设计背景/已知问题 → `docs/triton-pim-support-20260818.md`

## 6. 代码修改约定

1. 改代码必须同步更新 **本文件 + `README.md`**：目录/文件增删 → §2、§3；pass/算子/属性/环境变量变化 → §3；功能提交 → §4 追加一行。
2. 新增核心功能必须补 `test/Dialect/TritonPIM/*.mlir` lit 测试，并更新 §5 检索指引。
3. 索引粒度到二级目录，禁止虚构路径；提交者邮箱以 git 实际值 `fengjg@ios.ac.cn` 为准。
4. 生成/临时目录（`build/`、`.superpowers/`）不得写入索引正文。
