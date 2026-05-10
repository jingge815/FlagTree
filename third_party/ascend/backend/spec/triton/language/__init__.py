<<<<<<< flagtree/third_party/ascend/backend/spec/triton/language/__init__.py
def language_extend_globals(globals_dict):
    try:
        import acl
        is_compile_on_910_95 = acl.get_soc_name().startswith("Ascend910_95")
    except Exception as e:
        is_compile_on_910_95 = False
    globals_dict["is_compile_on_910_95"] = is_compile_on_910_95
=======
"""isort:skip_file"""
# Import order is significant here.
from triton.tools.get_ascend_devices import is_compile_on_910_95
from . import math
from . import extra
from .standard import (
    argmax,
    argmin,
    # cdiv,
    cumprod,
    cumsum,
    flip,
    interleave,
    max,
    min,
    ravel,
    sigmoid,
    softmax,
    sort,
    sum,
    swizzle2d,
    topk,
    xor_sum,
    zeros,
    zeros_like,
)
from .core import (
    PropagateNan,
    TRITON_MAX_TENSOR_NUMEL,
    _experimental_descriptor_load,
    _experimental_descriptor_store,
    make_tensor_descriptor,
    load_tensor_descriptor,
    store_tensor_descriptor,
    add,
    advance,
    arange,
    associative_scan,
    assume,
    atomic_add,
    atomic_and,
    atomic_cas,
    atomic_max,
    atomic_min,
    atomic_or,
    atomic_xchg,
    atomic_xor,
    bfloat16,
    block_type,
    broadcast,
    broadcast_to,
    cat,
    cast,
    clamp,
    const,
    constexpr,
    debug_barrier,
    device_assert,
    device_print,
    dot,
    dot_scaled,
    dtype,
    expand_dims,
    float16,
    float32,
    float64,
    float8e4b15,
    float8e4nv,
    float8e4b8,
    float8e5,
    float8e5b16,
    full,
    function_type,
    gather,
    histogram,
    inline_asm_elementwise,
    int1,
    int16,
    int32,
    int64,
    int8,
    join,
    load,
    make_block_ptr,
    max_constancy,
    max_contiguous,
    maximum,
    minimum,
    multiple_of,
    num_programs,
    permute,
    pi32_t,
    pointer_type,
    nv_tma_desc_type,
    program_id,
    range,
    reduce,
    reshape,
    split,
    static_assert,
    static_print,
    static_range,
    store,
    tensor,
    trans,
    uint16,
    uint32,
    uint64,
    uint8,
    view,
    void,
    where,
)
from .math import (umulhi, exp, exp2, fma, log, log2, cos, rsqrt, sin, sqrt, sqrt_rn, abs, fdiv, div_rn, erf, floor,
                   ceil, cdiv)
from .random import (
    pair_uniform_to_normal,
    philox,
    philox_impl,
    rand,
    rand4x,
    randint,
    randint4x,
    randn,
    randn4x,
    uint_to_uniform_float,
)
>>>>>>> triton-ascend@HEAD/python/triton/language/__init__.py


def language_extend_exports(globals_dict, all_list):
    from triton.language.core import make_tensor_descriptor, load_tensor_descriptor, store_tensor_descriptor, gather
    from triton.language.standard import topk
    globals_dict["make_tensor_descriptor"] = make_tensor_descriptor
    globals_dict["load_tensor_descriptor"] = load_tensor_descriptor
    globals_dict["store_tensor_descriptor"] = store_tensor_descriptor
    globals_dict["gather"] = gather
    globals_dict["topk"] = topk
    all_list.append("make_tensor_descriptor")
    all_list.append("load_tensor_descriptor")
    all_list.append("store_tensor_descriptor")
    all_list.append("gather")
    all_list.append("topk")
