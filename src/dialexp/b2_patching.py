"""B2 — Causal faithfulness via activation patching.

Design follows Zhang & Nanda (2024, arXiv:2309.16042) and Heimersheim & Nanda
(2024, arXiv:2404.15255):

- **Corruption = Symmetric Token Replacement**, not Gaussian noise. Two modes, both
  same-length so positions stay aligned and the context stays in-distribution:
  `scale` swaps a domain number ("17.10 pounds" -> "20.52 pounds"), `hours` leaves
  the venue the model listed first open for only 30 minutes on the queried day
  ("10:30" -> "21:00") so it drops out of the answer.
- **Metric = logit difference**, not probability: `logit(clean answer token) -
  logit(token the corrupted run prefers)`, read at the first token of the located
  `parsed_answer` span. Needs no recomputation of the corrupted ground truth.
- **Both directions**: denoising (patch clean -> corrupted run) tests sufficiency,
  noising (patch corrupted -> clean run) tests necessity. They are not symmetric.
- **Single layer at a time**, never a sliding window over layers.

Patched sites are whole decoder-layer outputs (residual stream) at four position
groups — `fact` (the swapped tokens), `prompt`, `reasoning`, `answer` — mirroring
B1's region split, so the correlational (B1) and causal (B2) maps are comparable.

Reads results/step_a/<task>-<model>-<setup>.jsonl; writes
results/patching/<task>-<model>-<setup>.jsonl.
"""
from __future__ import annotations

import json
import logging
import re

from dialexp.b1_attnlrp import (
    _DTYPES,
    _char_span_to_token_span,
    _load_rows,
    _locate_answer_span,
    _normalize_with_map,
)
from dialexp.config import Config

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# tried in order; first one that keeps the string length (=> token count) wins
_FACTORS = (1.2, 0.8, 1.5, 0.6, 1.1, 0.9, 1.3, 0.7)


def _scale_number(text: str, factor: float) -> str | None:
    """Scale the first number in `text`, keeping the string length identical."""
    match = _NUMBER.search(text)
    if not match:
        return None
    raw = match.group()
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    new = f"{float(raw) * factor:.{decimals}f}"
    if len(new) != len(raw) or new == raw:
        return None
    return text[: match.start()] + new + text[match.end() :]


def _swap_number(data, dotted: str) -> tuple[str, str, str] | None:
    node = data[0] if isinstance(data, list) else data
    *parents, key = dotted.split(".")
    for parent in parents:
        if not isinstance(node, dict) or parent not in node:
            return None
        node = node[parent]
    if not isinstance(node, dict) or key not in node:
        return None
    old = str(node[key])
    for factor in _FACTORS:
        new = _scale_number(old, factor)
        if new is not None:
            node[key] = new
            return dotted, old, new
    return None


def _shrink_hours(data, parsed_answer, weekday: str | None) -> tuple[str, str, str] | None:
    """Leave the venue the model listed first open 30 minutes on the queried day.

    Too short for both question forms in this task ("at least an hour" and "the
    entire time between X and Y"), so that venue drops out of the answer.
    """
    if not (isinstance(data, list) and isinstance(parsed_answer, list) and parsed_answer and weekday):
        return None
    wanted = _normalize_with_map(str(parsed_answer[0]))[0].strip()
    venue = next(
        (v for v in data if _normalize_with_map(str(v.get("name", "")))[0].strip() == wanted), None,
    )
    if venue is None:
        return None
    hours = venue.get("openhours", {}).get(weekday)
    if not isinstance(hours, dict) or "open" not in hours or "close" not in hours:
        return None
    try:
        hh, mm = (int(part) for part in str(hours["close"]).split(":"))
    except ValueError:
        return None
    total = (hh * 60 + mm - 30) % (24 * 60)
    old, new = str(hours["open"]), f"{total // 60:02d}:{total % 60:02d}"
    if new == old or len(new) != len(old):
        return None
    hours["open"] = new
    return f"{venue['name']}.{weekday}.open", old, new


def _corrupt_messages(
    messages: list[dict], spec: dict, parsed_answer, weekday: str | None,
) -> tuple[list[dict], str, str, str] | None:
    """Symmetric token replacement of one domain fact in the last tool message."""
    idx = next((i for i in reversed(range(len(messages))) if messages[i].get("role") == "tool"), None)
    if idx is None:
        return None
    content = messages[idx].get("content") or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    # re-dumping must reproduce the original byte for byte, so the swap is the only change
    if json.dumps(data) != content:
        return None
    change = (
        _shrink_hours(data, parsed_answer, weekday)
        if spec.get("mode") == "hours"
        else _swap_number(data, spec["path"])
    )
    if change is None:
        return None
    patched = list(messages)
    patched[idx] = {**messages[idx], "content": json.dumps(data)}
    return patched, *change


def _build_model(config: Config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    dtype = getattr(torch, _DTYPES.get(config.dtype, "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(config.model, dtype=dtype, device_map=config.device)
    model.eval()
    return model, tokenizer


def _layers(model):
    """Decoder layers; Qwen3.5's multimodal class nests them under `language_model`."""
    inner = model.model
    return inner.layers if hasattr(inner, "layers") else inner.language_model.layers


def _cache_hook(cache: dict, layer_idx: int):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        cache[layer_idx] = hidden.detach().clone()

    return hook


def _patch_hook(cached, positions):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, positions] = cached[:, positions].to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    return hook


def _run(model, ids, read_pos: int, cache: dict | None = None, patch: tuple | None = None):
    """One teacher-forced forward pass; returns the logit row at `read_pos`.

    `cache` collects every layer's residual stream; `patch` is (layer, positions,
    source_cache) to overwrite before reading.
    """
    import torch

    handles = []
    if cache is not None:
        handles += [
            layer.register_forward_hook(_cache_hook(cache, i))
            for i, layer in enumerate(_layers(model))
        ]
    if patch is not None:
        layer_idx, positions, source = patch
        handles.append(_layers(model)[layer_idx].register_forward_hook(
            _patch_hook(source[layer_idx], positions),
        ))
    try:
        with torch.no_grad():
            return model(input_ids=ids, use_cache=False).logits[0, read_pos].float()
    finally:
        for handle in handles:
            handle.remove()


def _prepare(model, tokenizer, row: dict, spec: dict, weekday: str | None) -> dict | None:
    """Build the aligned clean/corrupted pair and the region positions, or None to skip."""
    import torch

    row_id = row.get("id")
    if row.get("finish_reason") == "length":
        logger.warning("SKIP id=%s: response truncated, no clean final answer to locate", row_id)
        return None

    corrupted = _corrupt_messages(row["messages"], spec, row.get("parsed_answer"), weekday)
    if corrupted is None:
        logger.warning("SKIP id=%s: could not build a same-length swap (%s)", row_id, spec)
        return None
    corrupt_messages, field, old_value, new_value = corrupted

    template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt", "return_dict": False}
    clean_prompt = tokenizer.apply_chat_template(row["messages"], **template_kwargs).to(model.device)
    corrupt_prompt = tokenizer.apply_chat_template(corrupt_messages, **template_kwargs).to(model.device)
    if clean_prompt.shape[1] != corrupt_prompt.shape[1]:
        logger.warning("SKIP id=%s: swap changed the prompt token count — positions can't align", row_id)
        return None

    cot = row.get("cot") or ""
    response = row.get("response") or ""
    reasoning_prefix = f"{cot}\n</think>\n\n" if cot else ""
    continuation = reasoning_prefix + response
    encoded = tokenizer(
        continuation, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"][0].tolist()
    cont_ids = encoded["input_ids"].to(model.device)

    span = _locate_answer_span(response, row.get("parsed_answer"))
    if span is None:
        logger.warning("SKIP id=%s: parsed_answer missing or not found in response text", row_id)
        return None
    answer_char_start = len(reasoning_prefix)
    token_span = _char_span_to_token_span(
        offsets, answer_char_start + span[0], answer_char_start + span[1],
    )
    if token_span is None:
        logger.warning("SKIP id=%s: located value span didn't map to any token", row_id)
        return None

    n_prompt = clean_prompt.shape[1]
    reasoning_len = sum(1 for start, _ in offsets if start < answer_char_start)
    answer_start = min(n_prompt + reasoning_len, n_prompt + cont_ids.shape[1])
    value_start = max(n_prompt + token_span[0], answer_start)
    if value_start <= 0:
        logger.warning("SKIP id=%s: value span starts at position 0", row_id)
        return None

    clean_ids = torch.cat([clean_prompt, cont_ids], dim=1)
    corrupt_ids = torch.cat([corrupt_prompt, cont_ids], dim=1)
    fact_positions = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
    if not fact_positions:
        logger.warning("SKIP id=%s: swap produced identical tokens", row_id)
        return None

    # patching at or after the read position is causally inert, so cap every region
    limit = value_start
    regions = {
        "fact": [p for p in fact_positions if p < limit],
        "prompt": list(range(0, min(n_prompt, limit))),
        "reasoning": list(range(n_prompt, min(answer_start, limit))),
        "answer": list(range(answer_start, limit)),
    }
    return {
        "clean_ids": clean_ids,
        "corrupt_ids": corrupt_ids,
        "regions": {name: pos for name, pos in regions.items() if pos},
        "value_start": value_start,
        "n_prompt": n_prompt,
        "corrupted_field": field,
        "corrupted_from": old_value,
        "corrupted_to": new_value,
    }


def _patch_example(model, tokenizer, row: dict, spec: dict, weekday: str | None) -> dict | None:
    prepared = _prepare(model, tokenizer, row, spec, weekday)
    if prepared is None:
        return None

    row_id = row.get("id")
    clean_ids, corrupt_ids = prepared["clean_ids"], prepared["corrupt_ids"]
    read_pos = prepared["value_start"] - 1

    clean_cache, corrupt_cache = {}, {}
    clean_logits = _run(model, clean_ids, read_pos, cache=clean_cache)
    corrupt_logits = _run(model, corrupt_ids, read_pos, cache=corrupt_cache)

    clean_token = int(clean_ids[0, prepared["value_start"]])
    corrupt_token = int(corrupt_logits.argmax())
    if corrupt_token == clean_token:
        logger.warning("SKIP id=%s: corruption didn't move the answer token — nothing to trace", row_id)
        return None

    def logit_diff(logits) -> float:
        return float(logits[clean_token] - logits[corrupt_token])

    ld_clean, ld_corrupt = logit_diff(clean_logits), logit_diff(corrupt_logits)
    denom = ld_clean - ld_corrupt
    if denom <= 0:
        logger.warning("SKIP id=%s: clean run doesn't prefer its own answer token", row_id)
        return None

    effects = []
    for direction in ("denoising", "noising"):
        base_ids = corrupt_ids if direction == "denoising" else clean_ids
        source = clean_cache if direction == "denoising" else corrupt_cache
        for region, positions in prepared["regions"].items():
            for layer in range(len(_layers(model))):
                patched = logit_diff(_run(model, base_ids, read_pos, patch=(layer, positions, source)))
                score = (patched - ld_corrupt) if direction == "denoising" else (ld_clean - patched)
                effects.append({
                    "direction": direction,
                    "region": region,
                    "layer": layer,
                    "score": round(score / denom, 5),
                })

    return {
        "id": row_id,
        "setup_id": row.get("setup_id"),
        "task_name": row.get("task_name"),
        "model": row.get("model"),
        "corrupted_field": prepared["corrupted_field"],
        "corrupted_from": prepared["corrupted_from"],
        "corrupted_to": prepared["corrupted_to"],
        "n_prompt_tokens": prepared["n_prompt"],
        "value_start": prepared["value_start"],
        "clean_token": tokenizer.decode([clean_token]),
        "corrupt_token": tokenizer.decode([corrupt_token]),
        "logit_diff_clean": round(ld_clean, 4),
        "logit_diff_corrupt": round(ld_corrupt, 4),
        "region_sizes": {name: len(pos) for name, pos in prepared["regions"].items()},
        "n_layers": len(_layers(model)),
        "effects": effects,
    }


def run_b2(config: Config) -> None:
    """Standalone: builds its own model (plain forward passes, no LXT/gradients)."""
    fields = config.b2["fields"]
    model = tokenizer = None
    for task_name in config.tasks:
        spec = fields.get(task_name)
        if not spec:
            logger.warning("SKIP task %s: no b2.fields entry naming the fact to corrupt", task_name)
            continue
        weekdays = {}
        if spec.get("mode") == "hours":
            with open(config.benchmark_path(task_name)) as f:
                weekdays = {ex["id"]: ex.get("weekday") for ex in json.load(f)["examples"]}
        for setup_id in config.setups:
            src = config.result_path(task_name, setup_id)
            if not src.exists():
                logger.warning("MISSING Step A input: %s — run Step A first", src)
                continue
            out_path = config.b2_path(task_name, setup_id)
            if out_path.exists():
                logger.warning("SKIP (exists): %s — delete the file to regenerate", out_path)
                continue

            if model is None:
                logger.info("loading model for activation patching...")
                model, tokenizer = _build_model(config)

            rows = _load_rows(src)[: config.b2["max_examples"]]
            traces = [
                t for row in rows
                if (t := _patch_example(model, tokenizer, row, spec, weekdays.get(row.get("id"))))
            ]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for trace in traces:
                    f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            logger.info("wrote %d/%d patching traces -> %s", len(traces), len(rows), out_path)
