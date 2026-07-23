"""Ask-why baseline.

Replays each saved Step A conversation as `question -> answer -> "why?"` and asks
the SAME model to justify its own answer. The reconstructed assistant turn carries
only the final answer (`response`), never the saved `<think>` reasoning (`cot`) or
the final-turn tool calls — so the model must re-derive a justification from
scratch, reproducing what a real deployed thinking model sees (prior-turn
reasoning is stripped from history). 

Reads `results/step_a/{task}-{model}-{setup}.jsonl` and writes
`results/ask_why/{task}-{model}-{setup}.jsonl` (idempotent: skips existing files).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from dialexp.config import Config
from dialexp.hf_client import HFClient

logger = logging.getLogger(__name__)


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_ask_why(config: Config, client: HFClient | None = None) -> None:
    if client is None:
        client = HFClient(
            config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
        )
    why_prompt = config.ask_why["prompt"]

    for task_name in config.tasks:
        for setup_id in config.setups:
            src_path = config.result_path(task_name, setup_id)
            if not src_path.exists():
                logger.warning(
                    "MISSING Step A input: %s — run Step A first", src_path,
                )
                continue

            out_path = config.ask_why_path(task_name, setup_id)
            if out_path.exists():
                logger.warning(
                    "SKIP (exists): %s — delete the file to regenerate", out_path,
                )
                continue

            rows = []
            for row in _load_rows(src_path):
                answer = row.get("response")
                if not answer:
                    logger.warning(
                        "skip example %s (%s/%s): empty response, nothing to explain",
                        row.get("id"), task_name, setup_id,
                    )
                    continue

                # Replay question -> answer(only) -> "why?". No tools,
                # no prior reasoning — the model re-derives its own justification.
                messages = list(row["messages"]) + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": why_prompt},
                ]
                result = client.chat(messages=messages)
                rows.append({
                    "id": row["id"],
                    "setup_id": setup_id,
                    "model": config.model,
                    "task_name": task_name,
                    "why_prompt": why_prompt,
                    "explanation": result.content,
                    "explanation_cot": result.reasoning,
                    "source_response": answer,
                    "target": row.get("target"),
                })

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("wrote %d rows -> %s", len(rows), out_path)
