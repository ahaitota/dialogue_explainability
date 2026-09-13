"""Step C, phase 2 — grounded explanation synthesis.

Writes an explanation from the Step A trace **plus** the verified causes distilled
in `evidence.py`, instructed to cite only what the interventions established. This
is the experimental arm; the ask-why baseline is the control.

The two arms are built the same way on purpose: both replay the original Step A
conversation — same system prompt, same dialogue, same assistant answer — and both
end with the ask-why prompt verbatim. The findings arrive as an ordinary user turn
in that conversation rather than as a separate rulebook, so the only difference
between the arms is whether the causal evidence is present. An earlier version
replaced the system prompt with numbered rules; the model answered it as a task
specification, restating the rules instead of explaining (27 of 28 rows), and the
format constraints confounded any comparison with the free-form baseline.

The model is deliberately the *same* one used everywhere else in the pipeline. That
keeps model capability constant between the two arms, so any faithfulness gap is
attributable to the causal evidence rather than to a stronger explainer.

Reads results/evidence/<task>-<model>-<setup>.jsonl and the matching Step A rows;
writes results/explanations/<task>-<model>-<setup>.jsonl.
"""
from __future__ import annotations

import json
import logging

from dialexp.config import Config
from dialexp.evidence import render_evidence
from dialexp.hf_client import HFClient

logger = logging.getLogger(__name__)

_FINDINGS_TURN = (
    "One more thing before I ask — we ran controlled experiments on the answer you just "
    "gave, re-running it with parts of the input changed, to see what actually drove it. "
    "Here is what came back:\n\n{findings}\n\n"
    "Please rely only on what these results confirmed, and if something looks like it should "
    "have mattered but the results show it did not, say so plainly. Answer as you would to me "
    "as a customer — I do not know any experiments were run, so do not mention them."
)
_ACK = "Understood."


def _messages(base: dict, row: dict, question: str) -> list[dict]:
    """`question` is the ask-why prompt verbatim and stays the final turn, so the arms
    differ only in the findings turn that precedes it."""
    return [
        *base["messages"],
        {"role": "assistant", "content": base.get("response") or row.get("answer") or ""},
        {"role": "user", "content": _FINDINGS_TURN.format(findings=render_evidence(row))},
        {"role": "assistant", "content": _ACK},
        {"role": "user", "content": question},
    ]


def _rows_by_id(path) -> dict:
    with open(path) as f:
        return {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}


def run_c1(config: Config, client: HFClient | None = None) -> None:
    if client is None:
        client = HFClient(
            config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
        )
    question = config.ask_why["prompt"]

    for task_name in config.tasks:
        for setup_id in config.setups:
            src = config.evidence_path(task_name, setup_id)
            if not src.exists():
                logger.warning("MISSING evidence: %s — run the evidence phase first", src)
                continue
            step_a_path = config.result_path(task_name, setup_id)
            if not step_a_path.exists():
                logger.warning("MISSING Step A input: %s — the dialogue is replayed from it",
                               step_a_path)
                continue
            step_a = _rows_by_id(step_a_path)
            out_path = config.explanation_path(task_name, setup_id)
            if out_path.exists():
                logger.warning("SKIP (exists): %s — delete the file to regenerate", out_path)
                continue

            out_rows = []
            with open(src) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    base = step_a.get(row["id"])
                    if base is None:
                        logger.warning("SKIP id=%s (%s/%s): no Step A row to replay",
                                       row["id"], task_name, setup_id)
                        continue
                    messages = _messages(base, row, question)
                    result = client.chat(messages=messages)
                    out_rows.append({
                        "id": row["id"],
                        "task_name": task_name,
                        "setup_id": setup_id,
                        "model": config.model,
                        "sources": row.get("sources"),
                        "explanation": result.content,
                        "explanation_cot": result.reasoning,
                        "finish_reason": getattr(result, "finish_reason", None),
                        "why_prompt": question,
                        "prompt": messages[-3]["content"],
                        "messages": messages,
                    })

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for out_row in out_rows:
                    f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            logger.info("wrote %d grounded explanations -> %s", len(out_rows), out_path)
