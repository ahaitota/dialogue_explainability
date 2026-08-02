"""B4 — Logic masking (tool-output intervention).

Reruns the `dialogue` setup with **one tool's output intercepted** while leaving
the prompt and tool schemas unchanged, so the model still issues the same tool
calls. Two interventions (`configs/masks/logic_masks.yaml`):

- **disable** — the tool returns an error (does the model recover or hallucinate?).
- **scale**   — the tool's numbers are multiplied by `factor` (does the final
  answer track the corrupted value? — the strongest causal-reliance test).

Each rerun is compared to the Step A answer (`answer_changed`). A faithful
explanation that cites a tool should be sensitive to that tool's output; if the
answer is unchanged when the tool is disabled/corrupted, the model did not
actually rely on it.

Only the arithmetic tools are genuinely called by the model (domain retrieval is
pre-baked into the dialogue history), so B4 requires a Step A run with
`include_arithmetic_tools: true`. Reads `results/step_a/<task>-<model>-dialogue.jsonl`;
writes `results/logic_masking/<task>-<model>-dialogue-<tag>.jsonl`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from dialexp.config import Config
from dialexp.hf_client import HFClient, HFResponseParser
from dialexp.tool_runtime import build_tools_for_setup, load_dbs, make_masked_runtime

logger = logging.getLogger(__name__)

SETUP = "dialogue"  # logic masking is impossible without tools


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_b4(config: Config, client: HFClient | None = None) -> None:
    if client is None:
        client = HFClient(
            config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
        )
    parser = HFResponseParser(client)

    with open(config.masks["logic"]) as f:
        specs = yaml.safe_load(f) or {}

    if not config.include_arithmetic_tools:
        logger.warning(
            "include_arithmetic_tools is false — the model calls no arithmetic tools, "
            "so there is nothing to mask. Run Step A and B4 with include_arithmetic_tools: true.",
        )

    for task_name in config.tasks:
        masks = specs.get(task_name) or []
        if not masks:
            continue
        src = config.result_path(task_name, SETUP)
        if not src.exists():
            logger.warning("MISSING Step A input: %s — run Step A (dialogue) first", src)
            continue
        rows_ref = _load_rows(src)
        if any("parsed_answer" not in r for r in rows_ref):
            logger.warning("%s has unparsed rows — run scripts/run_parser.py first", src)

        dbs = load_dbs(task_name, config.paths["db_dir"])
        tools = build_tools_for_setup(
            SETUP, dbs, include_arithmetic=config.include_arithmetic_tools,
        )

        for mask in masks:
            tag = mask["tag"]
            out_path = config.logic_masking_path(task_name, tag)
            if out_path.exists():
                logger.warning("SKIP (exists): %s — delete the file to regenerate", out_path)
                continue

            factor = float(mask.get("factor", 2.0))
            schemas, handler = make_masked_runtime(tools, mask["tool"], mask["mode"], factor)

            out_rows = []
            for ref in rows_ref:
                result = client.chat(
                    messages=list(ref["messages"]),
                    tool_schemas=schemas,
                    tool_handler=handler,
                    max_tool_iterations=config.max_tool_iterations,
                )
                rerun_parsed = parser.parse_answer(
                    result.content or "", ref["answer_type"], context=ref.get("parser_enum"),
                )
                ref_parsed = ref.get("parsed_answer")
                masked_called = any(tc["name"] == mask["tool"] for tc in result.tool_calls_made)
                out_rows.append({
                    "id": ref["id"],
                    "setup_id": SETUP,
                    "model": config.model,
                    "task_name": task_name,
                    "mask": {"tool": mask["tool"], "mode": mask["mode"], "factor": factor},
                    "masked_tool_called": masked_called,
                    "response": result.content,
                    "cot": result.reasoning,
                    "finish_reason": getattr(result, "finish_reason", None),
                    "tool_calls": result.tool_calls_made,
                    "parsed_answer": rerun_parsed,
                    "ref_parsed_answer": ref_parsed,
                    "answer_changed": (rerun_parsed != ref_parsed)
                    if ref_parsed is not None and rerun_parsed is not None else None,
                    "target": ref.get("target"),
                })

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_called = sum(r["masked_tool_called"] for r in out_rows)
            n_changed = sum(bool(r["answer_changed"]) for r in out_rows)
            logger.info(
                "wrote %d rows -> %s (masked tool called: %d, answer changed: %d)",
                len(out_rows), out_path, n_called, n_changed,
            )
