"""B3 — Context masking (Terentowicz et al., 2025).

Removes a key constraint word from the **user turns** of the saved conversation
and reruns inference, per example. The word is derived from the benchmark
metadata (`configs/masks/context_masks.yaml` lists which field to mask, e.g.
`restaurant_params.food` → "gastropub"), located case-insensitively in the user
turns, and deleted — no hand-picked keywords. Values that are paraphrased and
don't appear verbatim (e.g. area "centre" written as "central") are recorded as
`found: false` and not rerun.

Each rerun is compared to the Step A answer (`answer_changed`). Runs on **both
setups**: `dialogue` has two observables (did the tool-call arguments change? did
the answer change?), `dialogue-no-tools` has one (the answer). The unfaithfulness
signal (choice unchanged but the explanation cited the removed word) is assessed
downstream in Step C against these reruns.

Reads `results/step_a/<task>-<model>-<setup>.jsonl`; writes
`results/context_masking/<task>-<model>-<setup>-<field>.jsonl`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from dialexp.config import Config
from dialexp.hf_client import HFClient, HFResponseParser
from dialexp.inference import load_datasets  # bridge re-export of boulder.inference
from dialexp.tool_runtime import build_tools_for_setup, load_dbs, make_runtime

logger = logging.getLogger(__name__)


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_nested(data: dict | None, dotted_path: str):
    cur = data
    for key in dotted_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def mask_value_in_messages(messages: list[dict], value: str) -> tuple[list[dict], bool]:
    """Delete whole-word, case-insensitive occurrences of `value` from user turns.

    Returns the (possibly) modified messages and whether the value was found.
    """
    pattern = re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)
    found = False
    masked = []
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            new_content, n = pattern.subn("", message["content"])
            if n:
                found = True
                new_content = re.sub(r"\s{2,}", " ", new_content).strip()
                message = {**message, "content": new_content}
        masked.append(message)
    return masked, found


def _is_dialogue(setup_id: str) -> bool:
    return "no-tools" not in setup_id


def run_b3(config: Config, client: HFClient | None = None) -> None:
    if client is None:
        client = HFClient(
            config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
        )
    parser = HFResponseParser(client)

    with open(config.masks["context"]) as f:
        specs = yaml.safe_load(f) or {}

    for task_name in config.tasks:
        fields = specs.get(task_name) or []
        if not fields:
            continue
        dataset = load_datasets([str(config.benchmark_path(task_name))])[0]
        meta_by_id = {ex["id"]: ex for ex in dataset["examples"]}
        dbs = load_dbs(task_name, config.paths["db_dir"])

        for setup_id in config.setups:
            src = config.result_path(task_name, setup_id)
            if not src.exists():
                logger.warning("MISSING Step A input: %s — run Step A first", src)
                continue
            rows_ref = _load_rows(src)
            if any("parsed_answer" not in r for r in rows_ref):
                logger.warning("%s has unparsed rows — run scripts/run_parser.py first", src)

            if _is_dialogue(setup_id):
                tools = build_tools_for_setup(
                    setup_id, dbs, include_arithmetic=config.include_arithmetic_tools,
                )
                schemas, handler = make_runtime(tools)
            else:
                schemas, handler = None, None

            for field in fields:
                tag = field.replace(".", "_")
                out_path = config.context_masking_path(task_name, setup_id, tag)
                if out_path.exists():
                    logger.warning("SKIP (exists): %s — delete the file to regenerate", out_path)
                    continue

                out_rows = []
                for ref in rows_ref:
                    value = _get_nested(meta_by_id.get(ref["id"]), field)
                    row = {
                        "id": ref["id"],
                        "setup_id": setup_id,
                        "model": config.model,
                        "task_name": task_name,
                        "masked_field": field,
                        "masked_value": value,
                        "ref_parsed_answer": ref.get("parsed_answer"),
                        "target": ref.get("target"),
                    }
                    if not isinstance(value, str) or not value:
                        out_rows.append({**row, "found": False, "answer_changed": None})
                        continue
                    masked_messages, found = mask_value_in_messages(ref["messages"], value)
                    if not found:
                        out_rows.append({**row, "found": False, "answer_changed": None})
                        continue

                    chat_kwargs = {"messages": masked_messages}
                    if schemas:
                        chat_kwargs.update(
                            tool_schemas=schemas,
                            tool_handler=handler,
                            max_tool_iterations=config.max_tool_iterations,
                        )
                    result = client.chat(**chat_kwargs)
                    rerun_parsed = parser.parse_answer(
                        result.content or "", ref["answer_type"], context=ref.get("parser_enum"),
                    )
                    ref_parsed = ref.get("parsed_answer")
                    out_rows.append({
                        **row,
                        "found": True,
                        "response": result.content,
                        "cot": result.reasoning,
                        "finish_reason": getattr(result, "finish_reason", None),
                        "tool_calls": result.tool_calls_made,
                        "parsed_answer": rerun_parsed,
                        "answer_changed": (rerun_parsed != ref_parsed)
                        if ref_parsed is not None and rerun_parsed is not None else None,
                    })

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    for row in out_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_found = sum(bool(r.get("found")) for r in out_rows)
                n_changed = sum(bool(r.get("answer_changed")) for r in out_rows)
                logger.info(
                    "wrote %d rows -> %s (masked: %d, answer changed: %d)",
                    len(out_rows), out_path, n_found, n_changed,
                )
