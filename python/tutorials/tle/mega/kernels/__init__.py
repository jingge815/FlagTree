"""TLE-backed kernels used by the Qwen3 megakernel tutorial."""

from .attention import attention_decode, attention_ws
from .embedding import embedding
from .linear import (
    linear,
    linear_backend_name,
    lm_head,
    qkv_linear,
    silu_and_mul,
    silu_and_mul_packed,
    silu_and_mul_packed_out,
)
from .linear_rmsnorm import (
    linear_rmsnorm_reference,
    linear_rmsnorm_triton_baseline,
)
from .linear_fused_rmsnorm import (
    linear_fused_add_rms_norm_decode_baseline,
    linear_fused_add_rms_norm_decode_mega,
    linear_fused_add_rms_norm_decode_reference,
    validate_linear_fused_add_rms_norm_decode_mega,
)
from .norm import fused_add_rms_norm, rms_norm
from .rotary_cache import apply_rotary_pos_emb, head_rmsnorm_rope, store_cache

__all__ = [
    "apply_rotary_pos_emb",
    "attention_decode",
    "attention_ws",
    "embedding",
    "fused_add_rms_norm",
    "head_rmsnorm_rope",
    "linear",
    "linear_backend_name",
    "linear_fused_add_rms_norm_decode_baseline",
    "linear_fused_add_rms_norm_decode_mega",
    "linear_fused_add_rms_norm_decode_reference",
    "lm_head",
    "linear_rmsnorm_reference",
    "linear_rmsnorm_triton_baseline",
    "qkv_linear",
    "rms_norm",
    "silu_and_mul",
    "silu_and_mul_packed",
    "silu_and_mul_packed_out",
    "store_cache",
    "validate_linear_fused_add_rms_norm_decode_mega",
]
