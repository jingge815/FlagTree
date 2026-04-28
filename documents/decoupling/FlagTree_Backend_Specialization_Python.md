# FlagTree Backend Specialization 统一设计（Python）

FlagTree 设计的后端统一特化，目的是整合后端接入范式，对后端的特化实现清晰化管理，为后端适配 Triton 版本升级迁移提供工程基础。具体实施方案是将各后端对 Triton 的特化，从以往的 fork 仓库直接修改并单独维护，标准化为定义接口并在后端目录下给出差异化实现。

## 1. 原则与规范

主干代码在保证缺省逻辑不变的基础上，允许调用接口，然后在后端目录中（third_party/backendxxx/）实现特化。主干代码原则上不允许直接出现某后端的特化实现，也不允许对后端做选择判断后特化实现。<br>
得益于 Python 的语法能力，通过统一的接口 spec、spec_func 接入特化函数字符串，特化函数由后端按需添加。当多后端对同一段主干代码有特化需求时，应协调保障多方特化功能。<br>

## 2. 接口

FlagTree 为 Python 代码的后端特化提供两种接口：spec 接口特化函数实现，spec_func 接口特化函数定义。由于调用了当前活动驱动类中的成员，只能在活动后端发现并激活后使用，因此一般来说只能用于一个局部作用域内。如果用在 py 文件的全局作用域且该文件在启动初期被 import，则会报错。

- python/triton/runtime/driver.py
```python
# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func(*args, **kwargs)
    return None
```

```python
# flagtree backend func specialization
def spec_func(function_name: str):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func
    return None
```

## 3. 后端入口注册

后端驱动类下需添加 spec 成员，注册该后端目录下的特化实现入口（以 iluvatar 后端为例）。注意原有的 utils 成员需改成 property，否则会循环注册。

- third_party/iluvatar/backend/driver.py
```python
class BackendDriver(GPUDriver):
    def __init__(self):
        # self.utils = CudaUtils()  # 对于 Triton 3.1 需改为 property
        self.launcher_cls = CudaLauncher
        # flagtree backend specialization
        from triton.backends.iluvatar import spec
        self.spec = spec
        super().__init__()
    @property
    def utils(self):
        return CudaUtils()
```

## 4. 使用实例

### 4.1 情形一：特化实现函数的一部分（spec）

#### 4.1.1 第一步：调用统一特化

本例中，缺省实现是 return tl.tensor(...)，特化函数起名为 atomic_add_int64。

- python/triton/language/semantic.py
```python
def atomic_add(ptr: tl.tensor, val: tl.tensor, mask: tl.tensor, sem: str, scope: str, builder: ir.builder) -> tl.tensor:
    ...
    rett = tl.tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle, sem, scope), val.type)
    # flagtree backend specialization
    from triton.runtime.driver import spec
    return spec("atomic_add_int64", sca_ty, builder, val, ptr, mask, sem, scope) or rett
```

#### 4.1.2 第二步：注册特化方法

- <strong>third_party/iluvatar/backend/spec/</strong>\_\_init\_\_.py
```python
from .triton.language.semantic import *
__all__ = [
    ..., "atomic_add_int64", ...
]
```

#### 4.1.3 第三步：实现特化函数

- <strong>third_party/iluvatar/backend/spec/</strong>triton/language/semantic.py
```python
def atomic_add_int64(sca_ty, builder, val, ptr, mask, sem, scope):
    from triton.language.semantic import full, and_, cast, lshr, bitcast, add, _bool_like, where, shl, or_
    ...
```

需要注意的是，如果需要特化一个判断条件（即特化函数返回布尔类型），那么应设计为后端特化时返回 True（缺省返回 False）。这是为了与 spec 方法当后端未做相应函数的特化时缺省返回 None 保持判断结果一致，保证缺省实现不变。

### 4.2 情形二：定义特化函数（spec_func）

#### 4.2.1 第一步：调用统一特化

- python/triton/ops/matmul.py
```python
@jit
def _kernel(A, B, C, M, N, K, ...):
    ...

class _matmul(torch.autograd.Function):
    # flagtree backend specialization
    from triton.runtime.driver import spec_func
    kernel = spec_func("matmul_kernel") or _kernel
    ...
```

#### 4.2.2 第二步：注册特化方法

- <strong>third_party/iluvatar/backend/spec/</strong>\_\_init\_\_.py
```python
from .triton.ops.matmul import *
__all__ = [
    ..., "matmul_kernel", ...
]
```

#### 4.2.3 第三步：实现特化函数

- <strong>third_party/iluvatar/backend/spec/</strong>triton/ops/matmul.py
```python
def matmul_kernel(grid, a, b, c, M, N, K, ...):
    from triton.ops.matmul import get_configs_io_bound
    ...

    @jit
    def _kernel(A, B, C, M, N, K, ...):
        ...
    return _kernel[grid](a, b, c, M, N, K, ...)
```

### 4.3 情形三：添加新的原语接口（例如 spec_semantic_func）

在 python/triton/language/ 目录下常有后端需要添加新的 tl 原语。上文介绍过，spec_func 在例如 semantic.py 的全局 scope 下是不能调用的，因此添加方法需要使用本节介绍的方案。

#### 4.3.1 第一步：调用统一特化

自动遍历后端定义在 core_ext_spec_api_list 列表中的方法，加入到本模块（tl.core）。当然，也可以按需加入到其他模块（例如 tl）。注意对于 semantic.py 方法名需加上 ext_semantic_ 前缀，与 core.py 的重名函数区分开。

- python/triton/language/core.py
```python
def spec_core_func(spec):
    import sys
    current_module_name = __name__
    parent_module_name = '.'.join(current_module_name.split('.')[:-1])

    for spec_api_name in spec.core_ext_spec_api_list:
        if hasattr(spec, spec_api_name):
            spec_api = getattr(spec, spec_api_name)
            # triton.language
            setattr(sys.modules[parent_module_name], spec_api.__name__, spec_api)
            # triton.language.core
            setattr(sys.modules[__name__], spec_api.__name__, spec_api)
```

#### 4.3.2 第二步：注册后端入口

- third_party/ascend/backend/driver.py
```python
class NPUDriver(DriverBase):
    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        # flagtree backend specialization
        from triton.backends.ascend import spec
        self.spec = spec  # 4.1 情形一
        from triton.language.core import spec_core_func
        spec_core_func(spec)  # 4.3 情形三
        from triton.language.semantic import spec_semantic_func
        spec_semantic_func(spec)  # 4.3 情形三
        from triton.language.standard import spec_standard_func
        spec_standard_func(spec)  # 4.3 情形三
        from triton.language.math import spec_math_func
        spec_math_func(spec)  # 4.4 情形四
        super().__init__()
```

- <strong>third_party/ascend/backend/spec/</strong>\_\_init\_\_.py
```python
from .triton.language.semantic import *
__all__  = [
    "core_ext_spec_api_list",
]
```

#### 4.3.3 第三步：实现特化函数

- <strong>third_party/ascend/backend/spec/</strong>triton/language/core.py
```python
@_tensor_member_fn
@builtin
def gather(src, index, axis, _builder=None):
    ...

core_ext_spec_api_list = [
    "gather", ...
]
```

### 4.4 情形四：修改或新增 tl.math 原语

第一、第二步与 4.3 大体一致，第三步的区别在于应按 Triton 规范实现于 libdevice.py。

#### 4.4.1 第一步：调用统一特化

```python
def spec_math_func(spec):
    import sys
    current_module_name = __name__
    parent_module_name = '.'.join(current_module_name.split('.')[:-1])

    for spec_api_name in spec.math_ext_base_api_list:
        if hasattr(spec, spec_api_name):
            spec_api = getattr(spec, spec_api_name)
            # triton.language
            setattr(sys.modules[parent_module_name], spec_api.__name__, spec_api)
            # triton.language.math
            setattr(sys.modules[__name__], spec_api.__name__, spec_api)

    for spec_api_name in spec.math_ext_spec_api_list:
        if hasattr(spec, spec_api_name):
            spec_api = getattr(spec, spec_api_name)
            # triton.language.math
            setattr(sys.modules[__name__], spec_api.__name__, spec_api)
```

#### 4.4.2 第二步：注册后端入口

- third_party/ascend/backend/driver.py
```python
class NPUDriver(DriverBase):
    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        # flagtree backend specialization
        from triton.backends.ascend import spec
        self.spec = spec  # 4.1 情形一
        from triton.language.math import spec_math_func
        spec_math_func(spec)  # 4.4 情形四
        super().__init__()
```

#### 4.4.3 第三步：实现特化函数

- <strong>third_party/ascend/backend/spec/</strong>triton/language/math.py
```python
import triton.language as language
exp = language.extra.ascend.libdevice.exp
math_ext_base_api_list = [
    "exp", ...  # tl.math 原有方法，但实现有特化，例如支持的 dtype 不同
]
math_ext_spec_api_list = [
    "isnan", ...  # 后端向 tl.math 新增的方法
]
```

- third_party/ascend/language/ascend/libdevice.py
```python
from triton.language import core, semantic

@core.extern
@_check_dtype(dtypes=["bf16", "fp16", "fp32"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x, _builder=None):
    x = semantic.to_tensor(x, _builder)
    return core.tensor(_builder.create_exp(x.handle), x.type)

@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("fp16"),): ("__hmf_isnan", core.dtype("int1")),
            (core.dtype("bf16"),): ("__hmf_isnan", core.dtype("int1")),
        }, is_pure=True, _builder=_builder)
```

## 5.整文件特化（以Ascend为例）
## 5.1 compiler模块
## 5.1.1 compiler.py
```text
third_party/Ascend/backend/spec/triton/compiler/compiler.py
```
第一步：
主干 `python/triton/compiler/__init__.py` 中有：
```python
from triton.runtime.driver import spec_path

spec_path(__path__)
```

第二步：
在 `python/triton/runtime/driver.py` 中增加`spec_path(path_list)`：
```python
def spec_path(path_list: list):
    import os
    if not path_list:
        return
    current_path = path_list[0].replace(os.sep, "/")
    marker = "/triton/"
    idx = current_path.find(marker)
    if idx == -1:
        return
    triton_root = current_path[:idx + len("/triton")]
    rel_path = current_path[idx + len(marker):]
    backend_path = os.path.join(triton_root, "backends", "ascend", "spec", "triton", rel_path)
    if os.path.isdir(backend_path):
        path_list.insert(0, backend_path)
```

第三步：
将 `python/triton/compiler/compiler.py` 中的：
```python
from .._C.libtriton import get_cache_invalidating_env_vars, ir
```

改成：
```python
from .._C.libtriton import get_cache_invalidating_env_vars, ir, buffer_ir
from .._C.libtriton.ascend import ir as ascend_ir
```

第四步：
在 `compile()` 主流程中补充 dialect 加载：
```python
context = ir.context()
ir.load_dialects(context)
buffer_ir.load_dialects(context)
ascend_ir.load_dialects(context)
backend.load_dialects(context)
```

第五步：
在 `compile()` 主流程中生成异常信息
```python
    for ext, compile_ir in list(stages.items())[first_stage:]:
        try:
            next_module = compile_ir(module, metadata)
        except Exception as e:
            if (ext == "ttadapter"):
                stage_name = "ConvertTritonIRToLinalgIR"
            elif (ext == "npubin"):
                stage_name = "ConvertLinalgRToBinary"
            else:
                stage_name = "MLIRCompile"
            error_detail = e.stderr.decode('utf-8') if hasattr(e, 'stderr') and e.stderr else str(e)
            error_detail += f"\n\n[INFO]: The compiled kernel cache is in {fn_cache_manager.cache_dir}\n\n"
            raise MLIRCompilationError(stage_name, error_detail) from e
        ir_filename = f"{file_name}.{ext}"
        if (fn_override_manager is not None and (full_name := fn_override_manager.get_file(ir_filename)) is not None):
```

注意事项：
主干中有：
```python
from .code_generator import ast_to_ttir
from .errors import MLIRCompilationError
```
继续特化
- `code_generator.py`
- `errors.py`

## 5.1.2 code_generator.py
```text
third_party/ascend/backend/spec/triton/compiler/code_generator.py
```
第一步：
增加 Ascend / buffer 相关 import：
```python
import triton.language.extra.cann.extension as extension
from triton.extension.buffer.language import core as bl
from triton.extension.buffer.language.builder import setup_unified_builder_with_buffer_builder
from .._C.libtriton import ir, buffer_ir
from .._C.libtriton.ascend import ir as ascend_ir
from triton.language.extra.cann.extension.dispatch import ASCEND_WITH_DISPATCH
from triton.language.extra.cann.extension.builder import setup_unified_builder
```

第二步：
增加 `WITH_DISPATCH` 注册逻辑：
```python
WITH_DISPATCH = {}
WITH_DISPATCH.update(ASCEND_WITH_DISPATCH)
```

第三步：
将：
```python
def mangle_ty(ty):
    ...
```

改成支持后端覆盖：
```python
mangle_ty = WITH_DISPATCH.get("mangle_ty", mangle_ty)
```

第四步：
在 `CodeGenerator.__init__(...)` 中，除了主干的 `ir.builder(...)` 之外，补充：
```python
self.ascend_builder = ascend_ir.ascendnpu_ir_builder(context, getattr(options, "arch", ""))
self.ascend_builder.set_loc(file_name, begin_line, 0)
setup_unified_builder(self.builder, self.ascend_builder)

self.buffer_builder = buffer_ir.buffer_builder(context)
self.buffer_builder.set_loc(file_name, begin_line, 0)
setup_unified_builder_with_buffer_builder(self.builder, self.buffer_builder)
```

第五步：
在CodeGenerator中增加：visit_With()
```python
    def visit_With(self, node):
        """Handle 'with' statements using dispatch pattern."""
        assert len(node.items) == 1
        context = node.items[0].context_expr

        # Check if context is a Call and dispatch to registered handler
        if isinstance(context, ast.Call):
            withitemClass = self.visit(context.func)
            handler = WITH_DISPATCH.get(withitemClass)
            if handler:
                return handler(self, node)

        # Fall back to visiting body for unhandled cases
        return self.visit_compound_statement(node.body)
```

第六步：
在 `visit_For()` 中支持 Ascend 扩展 iterator，例如：
```python
if IteratorClass in [language.range, extension.parallel, tle.dsa.pipeline, tle.dsa.parallel]:
    ...
```

第七步：
在参数封装时支持 buffer 类型：
```python
if isinstance(param_type, bl.buffer_type):
    arg_values.append(bl.buffer(self.fn.args(idx), param_type))
else:
    arg_values.append(tensor(self.fn.args(idx), param_type))
```

注意事项：
这个文件一旦特化，通常意味着 AST -> TTIR 前端语义已经变化，因此它往往与：
- `compiler.py`
- `language/core.py`
- `language/semantic.py`

这些文件存在联动关系。

## 5.1.3 errors.py
```text
third_party/ascend/backend/spec/triton/compiler/errors.py
```

第一步：
将主干文件：
```text
python/triton/compiler/errors.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/compiler/errors.py
```

第二步：
保留主干中原有错误类型：
- `CompilationError`
- `CompileTimeAssertionFailure`
- `UnsupportedLanguageConstruct`

第三步：
补充 Ascend 编译链专用错误类型：
```python
class MLIRCompilationError(TritonError):
    ...
```

第四步：
在该错误类型中增加对 stage 名称的包装，例如：
```python
self.stage_name = stage_name
```

第五步：
增加对错误信息的过滤，例如：
```python
def filter_message(self, message):
    return message.split("Stack dump without symbol names")[0]
```

第六步：
增加格式化输出，用于把 backend 编译错误按统一格式打印：
```python
def format_line_delim(self, keyword):
    return f"///------------------{keyword}------------------\\n"
```

注意事项：
只要 `compiler.py` 中出现：
```python
from .errors import MLIRCompilationError
```

那么 `errors.py` 就不再能直接复用主干文件，而必须一起特化。

## 5.2 language模块
## 5.2.1 core.py
```text
third_party/ascend/backend/spec/triton/language/core.py
```

第一步：
将主干文件：
```text
python/triton/language/core.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/language/core.py
```
第二步：
检查 `code_generator.py` 是否已经对 `core.py` 提出了新能力要求。

对于 Ascend，`code_generator.py` 已经引入了：
- buffer 类型支持；
- Ascend 扩展 builder；
- 扩展 `with` 语法；
- 新 iterator / 新原语；

因此 `core.py` 需要继续提供与这些前端行为匹配的基础原语与类型支持。

第三步：
扩展 `_utils` 依赖。

主干中有：
```python
from ._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape
```

Ascend 版改成：
```python
from ._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape, get_primitive_bitwidth
```

这说明：

> `core.py` 已经开始依赖新的基础工具函数，因此 `_utils.py` 也属于潜在联动特化文件。

第四步：
在 `core.py` 中补充 Ascend 需要暴露给上层 `triton.language` 的基础原语。

从 Ascend 当前实现和 `language/__init__.py` 的导出关系看，后续通常会由 `core.py` 承载：
- `make_tensor_descriptor`
- `load_tensor_descriptor`
- `store_tensor_descriptor`
- `gather`
-
第五步：
确认这些新增原语是否需要：
- `@builtin`
- `@_tensor_member_fn`
- `_builder` 参数

因为一旦 `code_generator.py` 通过 `visit_Call()` / `visit_With()` 调到这些接口，它们就必须符合 Triton builtin 的调用契约。

注意事项：
如果 `core.py` 里新增了主干没有的基础原语，那么通常还要继续联动：
- `language/__init__.py`（负责导出）
- `semantic.py`（负责这些原语底层语义实现）

## 5.2.2 semantic.py
```text
third_party/ascend/backend/spec/triton/language/semantic.py
```

第一步：
将主干文件：
```text
python/triton/language/semantic.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/language/semantic.py
```

第二步：
补充 Ascend 环境相关依赖。

Ascend 版增加了：
```python
from . import is_compile_on_910_95
```

这说明 `semantic.py` 已经开始依赖 Ascend 环境状态，而不再只是主干纯通用语义。

第三步：
检查 `code_generator.py` / `core.py` 新增的语言原语，是否需要在 `semantic.py` 中提供底层语义支撑。

典型包括：
- descriptor 类操作；
- gather 类操作；
- buffer / address space / tensor kind 相关语义；
- 针对 Ascend 芯片型号的条件逻辑。

第四步：
如果后端需要根据芯片型号切换语义分支，则在 `semantic.py` 中加入类似：
```python
from . import is_compile_on_910_95
```
并在具体语义实现里基于该变量分流。

## 5.2.3 standard.py
```text
third_party/ascend/backend/spec/triton/language/standard.py
```

第一步：
将主干文件：
```text
python/triton/language/standard.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/language/standard.py
```

第二步：
检查是否需要调整标准库函数的默认行为。

例如 Ascend 版 `ravel`：
```python
return core.reshape(x, [x.numel], can_reorder=False)
```

而主干中是：
```python
return core.reshape(x, [x.numel], can_reorder=True)
```

第四步：
如果后端需要新增标准库层公开接口，例如 `topk`，则应在 `standard.py` 中实现，并准备由：
- `language/__init__.py`
统一导出到 `triton.language` 顶层。

## 5.2.4 math.py
```text
third_party/ascend/backend/spec/triton/language/math.py
```

第一步：
将主干文件：
```text
python/triton/language/math.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/language/math.py
```

第二步：
检查 dtype 支持范围是否需要扩大。

例如主干 `exp` 的 dtype 检查是：
```python
@_check_dtype(dtypes=["fp32", "fp64"])
```

Ascend 版放宽成：
```python
@_check_dtype(dtypes=["bf16", "fp16", "fp32", "fp8e4nv", "fp8e5", "fp64"])
```

第三步：
如果后端对 bool/int1 有特殊表示，则要修改 `_check_dtype(...)`。

Ascend 版增加了：
```python
if hasattr(arg, 'was_bool_to_int8') and arg.was_bool_to_int8:
    arg_type = 'int1'
```

第四步：
检查主干数学原语是否需要新增后端实现，或者替换到底层 libdevice / builder。

典型包括：
- `exp`
- `exp2`
- `log`
- `log2`
- `cos`

这些函数名不变，但后端可能会改变：
- 接受的 dtype；
- 具体 builder 行为；
- libdevice 映射。

## 5.2.5 language/__init__.py
```text
third_party/ascend/backend/spec/triton/language/__init__.py
```

第一步：
根据后端需要，增加语言层全局变量注入函数，例如：
```python
def language_extend_globals(globals_dict):
    ...
```

第二步：
将后端新增原语导出到 `triton.language` 顶层，例如：
```python
def language_extend_exports(globals_dict, all_list):
    from triton.language.core import make_tensor_descriptor, load_tensor_descriptor, store_tensor_descriptor, gather
    from triton.language.standard import topk
    ...
```

# 5.2.6 _utils.py
```text
third_party/ascend/backend/spec/triton/language/_utils.py
```

第一步：
将主干文件：
```text
python/triton/language/_utils.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/language/_utils.py
```

第二步：
检查 `core.py` 是否已经引入了新的 `_utils` 依赖。

对于 Ascend，主干中是：
```python
from ._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape
```

而 Ascend 版 `core.py` 已经变成：
```python
from ._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape, get_primitive_bitwidth
```

第三步：
按后端需求调整 `validate_block_shape(...)` 之类的基础检查规则。

例如主干中有：
```python
if not is_power_of_two(d):
    raise ValueError(...)
```

第四步：
在 `_utils.py` 中补充后端需要的 dtype / bitwidth / canonicalize 工具，例如：
- `type_canonicalisation_dict`
- `canonicalize_dtype(dtype)`
- `BITWIDTH_DICT`
- `get_primitive_bitwidth(dtype)`

## 5.3 runtime模块
- 如果后端已经对：
  - kernel 启动路径；
  - autotune 配置；
  - 本地缓存路径；
  - library entry；
  - 解释执行；

  这些行为提出了新要求，就要继续联动到 runtime。

## 5.3.1 jit.py
```text
third_party/ascend/backend/spec/triton/runtime/jit.py
```

第一步：
将主干文件：
```text
python/triton/runtime/jit.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/runtime/jit.py
```

第二步：
检查 `compiler.py` 的整文件特化是否已经改变了 JIT 编译入口依赖。

如果 `jit.py` 中会直接调用：
- `compile`
- `ASTSource`
- `make_backend`

第三步：
检查 `run()` 流程中的：
- signature 推导；
- constants 组织；
- options 生成；
- cache key；
- kernel 启动参数；
是否还与主干兼容。

第四步：
如果后端对 `repr`、launch options、autotune metadata、grid 组织有差异，则在 `jit.py` 中补充对应逻辑。

## 5.3.2 autotuner.py
```text
third_party/ascend/backend/spec/triton/runtime/autotuner.py
```

第一步：
将主干文件：
```text
python/triton/runtime/autotuner.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/runtime/autotuner.py
```

第二步：
检查后端对 autotune config 的默认值是否有差异。

第三步：
检查后端是否需要不同的 benchmark / pre_hook / post_hook / prune 行为。

第四步：
如果 `jit.py` 已经改变了 kernel 启动和 config 消费方式，则 `autotuner.py` 也应联动调整。

## 5.3.3 interpreter.py
```text
third_party/ascend/backend/spec/triton/runtime/interpreter.py
```

第一步：
将主干文件：
```text
python/triton/runtime/interpreter.py
```

复制到：
```text
third_party/ascend/backend/spec/triton/runtime/interpreter.py
```

第二步：
检查后端是否对解释执行的数据类型、pointer/materialize 行为有不同要求。

第三步：
如果 language 层已经引入：
- 新 dtype；
- 新 pointer 语义；
- 新 builtin；

则 interpreter 往往要一起补这些行为。

## 5.3.4 code_cache.py
```text
third_party/ascend/backend/spec/triton/runtime/code_cache.py
```

第一步：
如果后端不想复用主干 `runtime/cache.py` 的缓存目录组织方式，就新增自己的缓存模块。

Ascend 当前提供的是：
```text
third_party/ascend/backend/spec/triton/runtime/code_cache.py
```

第二步：
定义后端自己的缓存根目录，例如：
- `FLAGGEMS_CACHE_DIR`
- `.flaggems`

第三步：
根据后端需求拆分：
- `code_cache_dir()`
- `config_cache_dir()`
- `clear_cache()`

## 5.3.5 libentry.py
```text
third_party/ascend/backend/spec/triton/runtime/libentry.py
```

第一步：
如果后端需要额外的 library entry / autotune 数据库存储 / torch 设备适配逻辑，可以在 runtime 下新增：
```text
libentry.py
```

第二步：
接入后端设备接口，例如 Ascend 版中：
```python
import torch_npu
torch_device_fn = torch.npu
```

第三步：
如果后端需要独立维护 tuned config 数据库、kernel cache 入口、library kernel 启动接口，可在该文件中实现。

注意事项：
`libentry.py` 不一定在每个后端都需要；它更像是 runtime 子体系里的后端新增模块，而不是所有后端都必须照抄的固定文件。
