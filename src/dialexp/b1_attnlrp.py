"""B1 — Attention saliency via AttnLRP (Achtibat et al., 2024; LXT library).

Teacher-forced replay of saved Step A sequences: sum every explained answer
token's target logit into one backward pass, and measure how much input
relevance flows onto them from the prompt / reasoning / answer-so-far.

Caveats:
- Needs the transformers-5.x compatibility patch (`dialexp._lxt_compat`).
- Qwen3 attribution is experimental (skewed toward the first token).
- Sequences are reconstructed from `cot`/`response`; the reasoning/answer
  boundary uses char offsets, or a re-tokenized approximation as fallback.
- One combined backward pass (not one per token) for speed under gradient
  checkpointing; causal masking makes the region-fraction split exact anyway.
- `target: "value"` restricts attribution to the located parsed_answer span
  instead of the whole answer.

Reads results/step_a/<task>-<model>-<setup>.jsonl; writes
results/attnlrp/<task>-<model>-<setup>.jsonl.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path

from dialexp.config import Config

logger = logging.getLogger(__name__)

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase/strip-punctuation/collapse-whitespace text (& -> and), keeping
    an index back to the original string, for fuzzy answer matching."""
    out_chars: list[str] = []
    index_map: list[int] = []
    prev_was_space = False
    for i, ch in enumerate(text):
        if ch == "&":
            for c in "and":
                out_chars.append(c)
                index_map.append(i)
            prev_was_space = False
        elif ch.isalnum():
            out_chars.append(ch.lower())
            index_map.append(i)
            prev_was_space = False
        elif ch.isspace() and not prev_was_space:
            out_chars.append(" ")
            index_map.append(i)
            prev_was_space = True
    return "".join(out_chars), index_map


def _fuzzy_find(haystack: str, needle: str) -> tuple[int, int] | None:
    norm_hay, hay_map = _normalize_with_map(haystack)
    norm_needle, _ = _normalize_with_map(needle)
    norm_needle = norm_needle.strip()
    if not norm_needle or not hay_map:
        return None
    pos = norm_hay.rfind(norm_needle) 
    if pos == -1:
        return None
    return hay_map[pos], hay_map[pos + len(norm_needle) - 1] + 1


def _locate_answer_span(response: str, parsed_answer) -> tuple[int, int] | None:
    """Char span in `response` covering every parsed_answer value, or None."""
    if not parsed_answer:
        return None
    values = parsed_answer if isinstance(parsed_answer, list) else [parsed_answer]
    spans = [s for s in (_fuzzy_find(response, str(v)) for v in values) if s]
    if not spans:
        return None
    return min(s[0] for s in spans), max(s[1] for s in spans)


def _char_span_to_token_span(offsets, char_start: int, char_end: int) -> tuple[int, int] | None:
    positions = [i for i, (s, e) in enumerate(offsets) if e > char_start and s < char_end]
    return (min(positions), max(positions) + 1) if positions else None


def _build_model(config: Config):
    """Load the model in LXT replay mode (eager attention, frozen params, patched)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dialexp._lxt_compat import monkey_patch
    from dialexp._lxt_qwen3_5_patch import MODULE_NAME as _QWEN3_5_MODULE
    from dialexp._lxt_qwen3_5_patch import build_patch_map as _build_qwen3_5_patch_map

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    dtype = getattr(torch, _DTYPES.get(config.dtype, "bfloat16"))
    # eager attention exposes the ops monkey_patch needs to intercept
    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype=dtype, device_map=config.device, attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # patches backward() to compute AttnLRP relevance instead of gradients
    module = importlib.import_module(type(model).__module__)
    # LXT has no built-in map for Qwen3.5 - have to use custom patch map (see _lxt_qwen3_5_patch.py)
    patch_map = _build_qwen3_5_patch_map() if module.__name__ == _QWEN3_5_MODULE else None
    monkey_patch(module, patch_map=patch_map, verbose=False)
    # gradient checkpointing trades compute for memory (recomputes on backward)
    model.gradient_checkpointing_enable()
    model.train()
    return model, tokenizer


def _attribute_example(
    model, tokenizer, row: dict, max_answer_tokens: int | None, target: str = "whole",
) -> dict | None:
    """Returns None to skip a row (target="value" only, when parsed_answer can't be located)."""
    import torch

    messages = row["messages"]
    cot = row.get("cot") or ""
    response = row.get("response") or ""

    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(model.device)
    # join reasoning + </think> + answer (Step A stored them separately)
    reasoning_prefix = f"{cot}\n</think>\n\n" if cot else ""
    continuation = reasoning_prefix + response
    answer_char_start = len(reasoning_prefix)

    # tokenize once with char offsets for an exact reasoning/answer boundary
    offsets = None
    try:
        encoded = tokenizer(
            continuation, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        offsets = offsets.tolist() if hasattr(offsets, "tolist") else offsets
        offsets = offsets[0]
        cont_ids = encoded["input_ids"].to(model.device)
        reasoning_len = sum(1 for start, _ in offsets if start < answer_char_start)
    except NotImplementedError:
        # slow tokenizer, no offsets: fall back to the approximate boundary
        cont_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        reasoning_len = 0
        if reasoning_prefix:
            reasoning_len = tokenizer(
                reasoning_prefix, add_special_tokens=False, return_tensors="pt",
            ).input_ids.shape[1]

    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    n_prompt = prompt_ids.shape[1]
    answer_start = min(n_prompt + reasoning_len, full_ids.shape[1])
    answer_positions = list(range(answer_start, full_ids.shape[1]))

    value_start = value_end = None
    if target == "value":
        if row.get("finish_reason") == "length":
            logger.warning("SKIP id=%s: response truncated, no clean final answer to locate", row.get("id"))
            return None
        if offsets is None:
            logger.warning("SKIP id=%s: no char offsets (slow tokenizer) — can't locate value span", row.get("id"))
            return None
        span = _locate_answer_span(response, row.get("parsed_answer"))
        if span is None:
            logger.warning("SKIP id=%s: parsed_answer missing or not found in response text", row.get("id"))
            return None
        token_span = _char_span_to_token_span(offsets, answer_char_start + span[0], answer_char_start + span[1])
        if token_span is None:
            logger.warning("SKIP id=%s: located value span didn't map to any token", row.get("id"))
            return None
        value_start = max(n_prompt + token_span[0], answer_start)
        value_end = min(n_prompt + token_span[1], full_ids.shape[1])
        explained = list(range(value_start, value_end))
        if not explained:
            logger.warning("SKIP id=%s: value span is empty after clamping", row.get("id"))
            return None
    else:
        explained = answer_positions if max_answer_tokens is None else answer_positions[:max_answer_tokens]

    # one teacher-forced forward pass over the whole known sequence
    embeds = model.get_input_embeddings()(full_ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeds, use_cache=False).logits

    valid = [q for q in explained if q != 0]
    n = len(valid)
    if n:
        # one combined backward pass; causal masking makes the slices below exact
        combined_target = sum(logits[0, q - 1, full_ids[0, q]] for q in valid)
        combined_target.backward()
        # Input×Gradient: embedding * its relevance, summed → one score per token
        relevance = (embeds * embeds.grad).sum(-1)[0].float().abs()
        total = float(relevance.sum().item()) or 1.0
        mean_prompt = float(relevance[:n_prompt].sum().item()) / total
        mean_reasoning = float(relevance[n_prompt:answer_start].sum().item()) / total
        mean_answer = float(relevance[answer_start:].sum().item()) / total
        token_relevance = [round(v, 5) for v in (relevance / total).tolist()]
    else:
        mean_prompt = mean_reasoning = mean_answer = None
        token_relevance = []

    # tokens for rendering a whole-text heatmap; see plots.plot_b1_token_heatmap
    from lxt.utils import clean_tokens

    tokens = clean_tokens(tokenizer.convert_ids_to_tokens(full_ids[0])) if n else []

    return {
        "id": row["id"],
        "setup_id": row.get("setup_id"),
        "task_name": row.get("task_name"),
        "model": row.get("model"),
        "n_prompt_tokens": n_prompt,
        "n_reasoning_tokens": reasoning_len,
        "n_answer_tokens": len(answer_positions),
        "n_explained": n,
        "value_start": value_start,
        "value_end": value_end,
        "mean_reasoning_relevance": mean_reasoning,
        "mean_prompt_relevance": mean_prompt,
        "mean_answer_relevance": mean_answer,
        "tokens": tokens,
        "token_relevance": token_relevance,
    }


def run_b1(config: Config) -> None:
    """Standalone: builds its own model since LXT replay mode differs from the generation client."""
    model = tokenizer = None
    for task_name in config.tasks:
        for setup_id in config.setups:
            src = config.result_path(task_name, setup_id)
            if not src.exists():
                logger.warning("MISSING Step A input: %s — run Step A first", src)
                continue
            out_path = config.b1_path(task_name, setup_id)
            if out_path.exists():
                logger.warning("SKIP (exists): %s — delete the file to regenerate", out_path)
                continue

            if model is None:
                logger.info("loading model for AttnLRP replay (eager attention)...")
                model, tokenizer = _build_model(config)

            rows = _load_rows(src)[: config.b1["max_examples"]]
            target = config.b1.get("target", "whole")
            summaries = []
            for row in rows:
                summary = _attribute_example(model, tokenizer, row, config.b1["max_answer_tokens"], target=target)
                if summary is not None:
                    summaries.append(summary)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for summary in summaries:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            logger.info(
                "wrote %d/%d AttnLRP summaries -> %s (target=%s)",
                len(summaries), len(rows), out_path, target,
            )
