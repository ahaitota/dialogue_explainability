"""Compatibility patch so LXT can import on transformers 5.x.

LXT (tested on transformers 4.52) imports `find_pruneable_heads_and_indices` from
`transformers.pytorch_utils`, which was removed in transformers 5.x. Only LXT's
BERT module (unused here) needs it, but the symbol must exist for the LXT package
to import at all. We reinstate the original implementation, then re-export
`monkey_patch`.

Verified on transformers 5.14.1: after this patch, `lxt.efficient` imports and
`monkey_patch(transformers.models.qwen3.modeling_qwen3)` applies cleanly.

Always import LXT via this module: `from dialexp._lxt_compat import monkey_patch`.
"""
from __future__ import annotations

import transformers.pytorch_utils as _pytorch_utils

if not hasattr(_pytorch_utils, "find_pruneable_heads_and_indices"):
    import torch

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        """Original transformers implementation (removed in 5.x)."""
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    _pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

from lxt.efficient import monkey_patch  # noqa: E402

__all__ = ["monkey_patch"]
