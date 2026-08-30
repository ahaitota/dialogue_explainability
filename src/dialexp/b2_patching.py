"""B2 — Causal faithfulness via activation patching.

Design follows Zhang & Nanda (2024, arXiv:2309.16042) and Heimersheim & Nanda
(2024, arXiv:2404.15255):

- **Corruption = Symmetric Token Replacement**, not Gaussian noise. Two modes, both
  same-length so positions stay aligned and the context stays in-distribution:
  `scale` swaps a domain number ("17.10 pounds" -> "20.52 pounds"), `hours` shifts
  a venue's opening time ("10:30" -> "13:30").
- **Metric = logit difference**, not probability: `logit(token the model actually
  wrote) - logit(token the corrupted run prefers)`, read where the model's own
  reasoning first restates the corrupted fact.
- **Both directions**: denoising (patch clean -> corrupted run) tests sufficiency,
  noising (patch corrupted -> clean run) tests necessity. They are not symmetric.
- **Single layer at a time**, never a sliding window over layers.

**Why the measurement sits at the fact's restatement, not at the final answer.**
The first version read logits at the answer value and skipped 32 of 37 examples with
"corruption didn't move the answer token". The continuation is teacher-forced, so by
the time the answer appears the model has already written the number in its own
reasoning and is copying it — verified: in 27 of 33 usable rows the answer is stated
verbatim earlier in the forced text. Corrupting the prompt cannot move a token that
is being copied. The reasoning's *first* restatement of the fact has no such upstream
copy, so the prompt is its only possible source. It also drops the `parsed_answer`
dependency that cost a further 12 + 11 rows.

Patched sites are whole decoder-layer outputs (residual stream) at position groups —
`fact` (the swapped tokens), `prompt`, `reasoning`, `answer` — mirroring B1's region
split, so the correlational (B1) and causal (B2) maps are comparable.

Reads results/step_a/<task>-<model>-<setup>.jsonl; writes
results/patching/<task>-<model>-<setup>.jsonl.
"""
from __future__ import annotations

import json
import logging
import re

from dialexp.b1_attnlrp import _DTYPES, _char_span_to_token_span, _load_rows
from dialexp.config import Config

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# tried in order; first one that keeps the string length (=> token count) wins
_FACTORS = (1.2, 0.8, 1.5, 0.6, 1.1, 0.9, 1.3, 0.7)
_HOURS_SHIFT = 3  # opening times move by whole hours, staying a plausible opening time


def _prose_form(value: str) -> str:
    """The part of a fact value that appears in prose: "17.10 pounds" is written "£17.10"."""
    if ":" in value:
        return value
    match = _NUMBER.search(value)
    return match.group() if match else value


def _fact_pattern(value: str) -> re.Pattern | None:
    prose = _prose_form(value)
    if ":" in prose:
        return re.compile(re.escape(prose))
    if not _NUMBER.fullmatch(prose):
        return None
    # reject digits embedded in a longer number, so "50" does not match "150"
    return re.compile(rf"(?<![\d.]){re.escape(prose)}(?![\d])")


def _first_changed_offset(old: str, new: str) -> int | None:
    """Offset into the prose form of the first character the swap changes.

    Qwen tokenises numbers digit by digit, so "10:30" -> "13:30" shares its first
    token and reading there would compare two identical predictions.
    """
    old_prose, new_prose = _prose_form(old), _prose_form(new)
    if len(old_prose) != len(new_prose):
        return None
    return next((i for i, (a, b) in enumerate(zip(old_prose, new_prose)) if a != b), None)


def _restated_at(continuation: str, value: str) -> tuple[int, int] | None:
    """Character span where the model's own text first writes this fact back."""
    pattern = _fact_pattern(value)
    match = pattern.search(continuation) if pattern else None
    return match.span() if match else None


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


def _replace_everywhere(node, old: str, new: str) -> int:
    """Swap every occurrence of a value, returning how many were changed.

    The same fact is usually repeated across the tool result (one venue's opening
    time recurs for every weekday, and often across venues). Corrupting a single
    occurrence leaves the model clean copies to read instead, so the prediction
    never moves and the example yields nothing.
    """
    if isinstance(node, dict):
        changed = 0
        for key, value in node.items():
            if isinstance(value, str) and value == old:
                node[key] = new
                changed += 1
            else:
                changed += _replace_everywhere(value, old, new)
        return changed
    if isinstance(node, list):
        return sum(_replace_everywhere(item, old, new) for item in node)
    return 0


def _swap_number(data, paths, continuation: str) -> tuple[str, str, str, int] | None:
    """Corrupt the first candidate path whose value the reasoning actually restates."""
    for dotted in paths:
        node = data[0] if isinstance(data, list) else data
        *parents, key = dotted.split(".")
        for parent in parents:
            if not isinstance(node, dict) or parent not in node:
                node = None
                break
            node = node[parent]
        if not isinstance(node, dict) or key not in node:
            continue
        old = str(node[key])
        if _restated_at(continuation, old) is None:
            continue
        for factor in _FACTORS:
            new = _scale_number(old, factor)
            if new is not None:
                return dotted, old, new, _replace_everywhere(data, old, new)
    return None


def _shift_hours(data, continuation: str) -> tuple[str, str, str, int] | None:
    """Shift the opening time of the first venue/day the reasoning quotes."""
    for venue in data if isinstance(data, list) else [data]:
        for day, hours in (venue.get("openhours") or {}).items():
            old = str((hours or {}).get("open") or "")
            if not old or _restated_at(continuation, old) is None:
                continue
            try:
                hh, mm = (int(part) for part in old.split(":"))
            except ValueError:
                continue
            new = f"{(hh + _HOURS_SHIFT) % 24:02d}:{mm:02d}"
            if new == old or len(new) != len(old):
                continue
            return f"{venue.get('name')}.{day}.open", old, new, _replace_everywhere(data, old, new)
    return None


def _corrupt_messages(
    messages: list[dict], spec: dict, continuation: str,
) -> tuple[list[dict], str, str, str, int] | None:
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
    if spec.get("mode") == "hours":
        change = _shift_hours(data, continuation)
    else:
        path = spec.get("path")
        change = _swap_number(data, [path] if isinstance(path, str) else path, continuation)
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


def _prepare(model, tokenizer, row: dict, spec: dict) -> dict | None:
    """Build the aligned clean/corrupted pair and the region positions, or None to skip."""
    import torch

    row_id = row.get("id")
    cot = row.get("cot") or ""
    response = row.get("response") or ""
    reasoning_prefix = f"{cot}\n</think>\n\n" if cot else ""
    continuation = reasoning_prefix + response
    if not continuation.strip():
        logger.warning("SKIP id=%s: empty continuation, nothing to replay", row_id)
        return None

    # the corruption is chosen so the model's own text restates it — otherwise
    # there is no position at which the swap can show an effect
    corrupted = _corrupt_messages(row["messages"], spec, continuation)
    if corrupted is None:
        logger.warning("SKIP id=%s: no same-length swap whose value the reasoning restates (%s)",
                       row_id, spec)
        return None
    corrupt_messages, field, old_value, new_value, n_occurrences = corrupted

    template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt", "return_dict": False}
    clean_prompt = tokenizer.apply_chat_template(row["messages"], **template_kwargs).to(model.device)
    corrupt_prompt = tokenizer.apply_chat_template(corrupt_messages, **template_kwargs).to(model.device)
    if clean_prompt.shape[1] != corrupt_prompt.shape[1]:
        logger.warning("SKIP id=%s: swap changed the prompt token count — positions can't align", row_id)
        return None

    encoded = tokenizer(
        continuation, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"][0].tolist()
    cont_ids = encoded["input_ids"].to(model.device)

    char_span = _restated_at(continuation, old_value)
    changed = _first_changed_offset(old_value, new_value)
    if changed is None:
        logger.warning("SKIP id=%s: swap changes no character of the quoted form", row_id)
        return None
    # read where the swap first alters the text, not at the start of the value
    read_char = char_span[0] + changed
    token_span = _char_span_to_token_span(offsets, read_char, read_char + 1)
    if token_span is None:
        logger.warning("SKIP id=%s: fact restatement didn't map to any token", row_id)
        return None

    n_prompt = clean_prompt.shape[1]
    answer_char_start = len(reasoning_prefix)
    reasoning_len = sum(1 for start, _ in offsets if start < answer_char_start)
    answer_start = min(n_prompt + reasoning_len, n_prompt + cont_ids.shape[1])
    fact_start = n_prompt + token_span[0]

    clean_ids = torch.cat([clean_prompt, cont_ids], dim=1)
    corrupt_ids = torch.cat([corrupt_prompt, cont_ids], dim=1)
    fact_positions = (clean_ids[0] != corrupt_ids[0]).nonzero().flatten().tolist()
    if not fact_positions:
        logger.warning("SKIP id=%s: swap produced identical tokens", row_id)
        return None

    # causal masking makes positions after the read position inert; the read
    # position itself is included, where a patch acts most directly
    limit = fact_start
    regions = {
        "fact": [p for p in fact_positions if p < limit],
        "prompt": list(range(0, min(n_prompt, limit))),
        "reasoning": list(range(n_prompt, min(answer_start, limit))),
        "answer": list(range(answer_start, limit)),
    }
    if not regions["fact"]:
        logger.warning("SKIP id=%s: the swapped tokens sit after the restatement", row_id)
        return None
    return {
        "clean_ids": clean_ids,
        "corrupt_ids": corrupt_ids,
        "regions": {name: pos for name, pos in regions.items() if pos},
        "fact_start": fact_start,
        "restated_in": "reasoning" if fact_start < answer_start else "answer",
        "n_prompt": n_prompt,
        "corrupted_field": field,
        "corrupted_from": old_value,
        "corrupted_to": new_value,
        "corrupted_occurrences": n_occurrences,
    }


def _patch_example(model, tokenizer, row: dict, spec: dict) -> dict | None:
    prepared = _prepare(model, tokenizer, row, spec)
    if prepared is None:
        return None

    row_id = row.get("id")
    clean_ids, corrupt_ids = prepared["clean_ids"], prepared["corrupt_ids"]
    read_pos = prepared["fact_start"] - 1

    clean_cache, corrupt_cache = {}, {}
    # run the clean and corrupted prompts
    clean_logits = _run(model, clean_ids, read_pos, cache=clean_cache)
    corrupt_logits = _run(model, corrupt_ids, read_pos, cache=corrupt_cache)

    clean_token = int(clean_ids[0, prepared["fact_start"]])
    corrupt_token = int(corrupt_logits.argmax())
    if corrupt_token == clean_token:
        logger.warning("SKIP id=%s: corruption didn't move the restated fact — nothing to trace", row_id)
        return None

    def logit_diff(logits) -> float:
        return float(logits[clean_token] - logits[corrupt_token])

    ld_clean, ld_corrupt = logit_diff(clean_logits), logit_diff(corrupt_logits)
    denom = ld_clean - ld_corrupt
    if denom <= 0:
        logger.warning("SKIP id=%s: clean run doesn't prefer the token it actually wrote", row_id)
        return None

    effects = []
    # run 256 patched run: 2 directions × 4 regions × 32 layers, reading the logit difference at the reusing the token from the prompt
    for direction in ("denoising", "noising"):
        base_ids = corrupt_ids if direction == "denoising" else clean_ids
        source = clean_cache if direction == "denoising" else corrupt_cache
        for region, positions in prepared["regions"].items():
            for layer in range(len(_layers(model))):
                logits = _run(model, base_ids, read_pos, patch=(layer, positions, source))
                patched = logit_diff(logits)
                score = (patched - ld_corrupt) if direction == "denoising" else (ld_clean - patched)
                effects.append({
                    "direction": direction,
                    "region": region,
                    "layer": layer,
                    "score": round(score / denom, 5),
                    # a logit difference can rise either by favouring the written token or
                    # merely by damaging its rival, so keep both sides separable
                    # (Heimersheim & Nanda 2024, sec. 4.2)
                    "logit_clean_token": round(float(logits[clean_token]), 4),
                    "logit_corrupt_token": round(float(logits[corrupt_token]), 4),
                })

    return {
        "id": row_id,
        "setup_id": row.get("setup_id"),
        "task_name": row.get("task_name"),
        "model": row.get("model"),
        "corrupted_field": prepared["corrupted_field"],
        "corrupted_from": prepared["corrupted_from"],
        "corrupted_to": prepared["corrupted_to"],
        "corrupted_occurrences": prepared["corrupted_occurrences"],
        "n_prompt_tokens": prepared["n_prompt"],
        "fact_start": prepared["fact_start"],
        "restated_in": prepared["restated_in"],
        "clean_token": tokenizer.decode([clean_token]),
        "corrupt_token": tokenizer.decode([corrupt_token]),
        "logit_diff_clean": round(ld_clean, 4),
        "logit_diff_corrupt": round(ld_corrupt, 4),
        "logit_clean_token_clean_run": round(float(clean_logits[clean_token]), 4),
        "logit_corrupt_token_clean_run": round(float(clean_logits[corrupt_token]), 4),
        "logit_clean_token_corrupt_run": round(float(corrupt_logits[clean_token]), 4),
        "logit_corrupt_token_corrupt_run": round(float(corrupt_logits[corrupt_token]), 4),
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
            traces = [t for row in rows if (t := _patch_example(model, tokenizer, row, spec))]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for trace in traces:
                    f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            logger.info("wrote %d/%d patching traces -> %s", len(traces), len(rows), out_path)
