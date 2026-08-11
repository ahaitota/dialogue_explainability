"""Custom LXT AttnLRP patch map for Qwen3.5 (transformers.models.qwen3_5).

LXT's built-in registry (`lxt.efficient.models.DEFAULT_MAP`) only covers plain
`transformers.models.qwen3.modeling_qwen3` — Qwen3.5 is from a a DIFFERENT module
(`transformers.models.qwen3_5.modeling_qwen3_5`) that LXT doesn't recognize, so
`monkey_patch()` raises `ValueError: ... not yet supported`.

This builds an equivalent custom `patch_map` from LXT's own generic patch
functions.
"""
from __future__ import annotations

from functools import partial

from torch.nn import Dropout

from lxt.efficient.models.gemma3 import gemma3_norm
from lxt.efficient.patches import check_already_patched, dropout_forward, gated_mlp_forward, patch_method
from lxt.efficient.patches import wrap_attention_forward

MODULE_NAME = "transformers.models.qwen3_5.modeling_qwen3_5"


def _patch_attention_compat(module) -> bool:
    """Same as LXT's `patch_attention`, but updates `ALL_ATTENTION_FUNCTIONS`
    entries in place instead of replacing the whole object with a plain dict.
    This transformers version exposes it as an `AttentionInterface` instance
    with a `.get_interface()` method that every model's attention forward
    calls -- LXT's wholesale-replacement approach silently breaks that
    (`AttributeError: 'dict' object has no attribute 'get_interface'`).
    """
    new_forward = wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    module.eager_attention_forward = new_forward

    for key, value in list(module.ALL_ATTENTION_FUNCTIONS.items()):
        new_forward = wrap_attention_forward(value)
        if check_already_patched(value, new_forward):
            return False
        module.ALL_ATTENTION_FUNCTIONS[key] = new_forward
    return True


def build_patch_map() -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

    return {
        Qwen3_5MLP: partial(patch_method, gated_mlp_forward),
        Qwen3_5RMSNorm: partial(patch_method, gemma3_norm, method_name="_norm"),
        Dropout: partial(patch_method, dropout_forward),
        modeling_qwen3_5: _patch_attention_compat,
    }

