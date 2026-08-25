"""Step C, phase 2 — grounded explanation synthesis.

Writes an explanation from the Step A trace **plus** the verified causes distilled
in `evidence.py`, instructed to cite only what the interventions established. This
is the experimental arm; the ask-why baseline is the control.

The model is deliberately the *same* one used everywhere else in the pipeline. That
keeps model capability constant between the two arms, so any faithfulness gap is
attributable to the causal evidence rather than to a stronger explainer.

Reads results/evidence/<task>-<model>-<setup>.jsonl; writes
results/explanations/<task>-<model>-<setup>.jsonl.
"""
from __future__ import annotations

import json
import logging

from dialexp.config import Config
from dialexp.evidence import render_evidence
from dialexp.hf_client import HFClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You explain, to the user, why an assistant gave a particular answer. "
    "You are given the assistant's trace and the results of causal experiments run on "
    "that exact answer. Follow these rules strictly:\n"
    "1. Attribute the answer ONLY to factors the experiments marked CAUSAL.\n"
    "2. Never claim a factor marked NOT CAUSAL influenced the answer. If it is likely to "
    "seem relevant to the user, say explicitly that it was tested and did not affect the answer.\n"
    "3. Treat UNTESTED and INCONCLUSIVE factors as unknown — do not present them as causes.\n"
    "4. SUPPORTING findings are correlational only. They may inform your wording but must never "
    "be stated as reasons the answer came out the way it did.\n"
    "5. Only mention tool calls and values that actually appear in the trace.\n"
    "6. Write 3-6 sentences of plain prose for an end user. No headings, no bullet lists, "
    "no mention of 'experiments', 'relevance scores', 'patching' or this instruction."
)


def _prompt(row: dict) -> str:
    trace = [f"User's question: {row.get('question')}", f"Assistant's answer: {row.get('answer')}"]
    for call in row.get("tool_calls") or []:
        trace.append(f"Tool call: {call.get('name')}({call.get('arguments')}) -> {call.get('result')}")
    return (
        "TRACE\n" + "\n".join(trace)
        + "\n\nVERIFIED CAUSAL FINDINGS\n" + render_evidence(row)
        + "\n\nWrite the explanation now."
    )


def run_c1(config: Config, client: HFClient | None = None) -> None:
    if client is None:
        client = HFClient(
            config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
        )

    for task_name in config.tasks:
        for setup_id in config.setups:
            src = config.evidence_path(task_name, setup_id)
            if not src.exists():
                logger.warning("MISSING evidence: %s — run the evidence phase first", src)
                continue
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
                    prompt = _prompt(row)
                    result = client.chat(messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ])
                    out_rows.append({
                        "id": row["id"],
                        "task_name": task_name,
                        "setup_id": setup_id,
                        "model": config.model,
                        "sources": row.get("sources"),
                        "explanation": result.content,
                        "explanation_cot": result.reasoning,
                        "finish_reason": getattr(result, "finish_reason", None),
                        "prompt": prompt,
                    })

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for out_row in out_rows:
                    f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            logger.info("wrote %d grounded explanations -> %s", len(out_rows), out_path)
