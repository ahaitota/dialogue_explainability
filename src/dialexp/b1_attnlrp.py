"""B1 — Attention saliency via AttnLRP (Achtibat et al., 2024; LXT library).

Teacher-forced **replay** of saved Step A sequences: reconstruct prompt + generated
tokens, run one LRP backward pass per explained answer token, and measure how much
input relevance flows from the **reasoning** tokens onto the **answer** tokens —
the proposal's B1 question ("did the reasoning actually influence the output?").

Uses LXT's efficient (Input×Gradient) AttnLRP: `monkey_patch` the model's modeling
module, forward on `inputs_embeds`, backward from a target-token logit, and read
`relevance = (embeds * embeds.grad).sum(-1)` per input token.

Caveats (see docs/project_structure.md):
- LXT needs a compatibility patch for transformers 5.x (`dialexp._lxt_compat`).
- Qwen3 attribution is experimental ("skewed toward first token") — accepted for
  now; swap to a fully-supported model (Llama-3/Gemma-3) later for cleaner maps.
- The generated sequence is *reconstructed* from `cot`/`response` (Step A did not
  save token ids). The reasoning/answer boundary is computed from character
  offsets (exact, for fast tokenizers), falling back to the old approximate
  re-tokenization method if the tokenizer doesn't support offset mapping.
- By default the WHOLE answer is explained (`config.b1["max_answer_tokens"] =
  None`); set an int to cap it for speed (pilot-scale). `config.b1["max_examples"]`
  still caps how many examples per (task, setup) are attributed — each backward
  pass is expensive.

Reads results/step_a/<task>-<model>-<setup>.jsonl; writes
results/attnlrp/<task>-<model>-<setup>.jsonl (per-example relevance summary).
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


def _build_model(config: Config):
    """Load the model in LXT replay mode (eager attention, frozen params, patched)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dialexp._lxt_compat import monkey_patch

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
    monkey_patch(importlib.import_module(type(model).__module__), verbose=False)
    return model, tokenizer


def _attribute_example(model, tokenizer, row: dict, max_answer_tokens: int | None) -> dict:
    import torch

    messages = row["messages"]
    cot = row.get("cot") or ""
    response = row.get("response") or ""

    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    # join reasoning + </think> + answer (Step A stored them separately)
    reasoning_prefix = f"{cot}\n</think>\n\n" if cot else ""
    continuation = reasoning_prefix + response
    answer_char_start = len(reasoning_prefix)

    # tokenize once with char offsets for an exact reasoning/answer boundary
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
    explained = answer_positions if max_answer_tokens is None else answer_positions[:max_answer_tokens]

    # one teacher-forced forward pass over the whole known sequence
    embeds = model.get_input_embeddings()(full_ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeds, use_cache=False).logits

    relevance_fracs = []
    for q in explained:
        if q == 0:
            continue
        target = full_ids[0, q]
        if embeds.grad is not None:
            embeds.grad.zero_()
        logits[0, q - 1, target].backward(retain_graph=True)
        # Input×Gradient: embedding * its relevance, summed → one score per token
        relevance = (embeds * embeds.grad).sum(-1)[0].float().abs()
        total = float(relevance.sum().item()) or 1.0
        # split relevance by region: prompt / reasoning / answer-so-far
        prompt_mass = float(relevance[:n_prompt].sum().item())
        reasoning_mass = float(relevance[n_prompt:answer_start].sum().item())
        answer_mass = float(relevance[answer_start:q].sum().item())
        relevance_fracs.append({
            "prompt": prompt_mass / total,
            "reasoning": reasoning_mass / total,
            "answer": answer_mass / total,
        })

    n = len(relevance_fracs)
    mean_reasoning = sum(f["reasoning"] for f in relevance_fracs) / n if n else None
    mean_prompt = sum(f["prompt"] for f in relevance_fracs) / n if n else None
    mean_answer = sum(f["answer"] for f in relevance_fracs) / n if n else None
    return {
        "id": row["id"],
        "setup_id": row.get("setup_id"),
        "task_name": row.get("task_name"),
        "model": row.get("model"),
        "n_prompt_tokens": n_prompt,
        "n_reasoning_tokens": reasoning_len,
        "n_answer_tokens": len(answer_positions),
        "n_explained": n,
        "mean_reasoning_relevance": mean_reasoning,
        "mean_prompt_relevance": mean_prompt,
        "mean_answer_relevance": mean_answer,
        "reasoning_relevance_per_token": [round(f["reasoning"], 4) for f in relevance_fracs],
        "answer_relevance_per_token": [round(f["answer"], 4) for f in relevance_fracs],
    }


def run_b1(config: Config) -> None:
    """AttnLRP runs over saved Step A results. Builds its own model
    (LXT replay mode differs from the generation client), so it is standalone."""
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
            summaries = [
                _attribute_example(model, tokenizer, row, config.b1["max_answer_tokens"])
                for row in rows
            ]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for summary in summaries:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            logger.info("wrote %d AttnLRP summaries -> %s", len(summaries), out_path)
