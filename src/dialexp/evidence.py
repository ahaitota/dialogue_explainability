"""Step C, phase 1 — B → verified causes.

Turns the Step B experiments' raw output (numeric relevance arrays, patching
scores, masking reruns) into the discrete, per-example causal claims that C1
synthesis and C2 judging both consume. Pure data processing: no model, no GPU.

The claims are deliberately conservative about what B can and cannot establish:

- B3/B4 give **directly** falsifiable verdicts (remove the factor, did the answer
  change?), so they are the backbone. A B4 mask whose tool the model never called
  is *inconclusive*, not evidence of non-causality.
- B1/B2 give region-level claims ("relevance/causal influence concentrated here"),
  never per-factor ones.

Thresholds live in `config.step_c` and are preregistration-style: fix them before
running, never tune them after seeing results (see the human-influence firewall in
docs/project_structure.md).

Writes results/evidence/<task>-<model>-<setup>.jsonl.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from dialexp.config import Config

logger = logging.getLogger(__name__)

# B4 needs tools to intercept, so it cannot exist in a no-tools setup
_STAGES_BY_SETUP = {"dialogue": ("b1", "b2", "b3", "b4"), "dialogue-no-tools": ("b1", "b2", "b3")}
_CONTROL_TOKEN = re.compile(r"<\|.*?\|>|</?think>")
_LATEX_ESCAPE = re.compile(r"\\([_#$%&{}])")


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _rows_by_id(path: Path) -> dict:
    return {row["id"]: row for row in _load_rows(path)} if path.exists() else {}


def _group_by_id(paths, setup_id: str) -> dict:
    """Rows keyed by example id. Filters on each row's own `setup_id`, because the
    filename glob for `dialogue` also matches `dialogue-no-tools`."""
    grouped: dict = {}
    for path in paths:
        for row in _load_rows(path):
            if row.get("setup_id") != setup_id:
                continue
            grouped.setdefault(row["id"], []).append(row)
    return grouped


def _is_content(token: str) -> bool:
    return bool(token) and not _CONTROL_TOKEN.search(token) and any(c.isalnum() for c in token)


def _top_spans(tokens: list[str], relevance: list[float], limit: int, top_k: int) -> list[str]:
    """Highest-relevance prompt tokens, merged into contiguous readable spans."""
    from dialexp.plots import _fix_byte_level_tokens

    text = _fix_byte_level_tokens(tokens)
    limit = min(limit, len(relevance), len(text))
    # token 0 is the attention sink (it dominates every heatmap and means nothing);
    # chat-control tokens carry no content a user-facing explanation could cite
    candidates = [i for i in range(1, limit) if _is_content(text[i])]
    ranked = sorted(candidates, key=lambda i: relevance[i], reverse=True)[:top_k]
    groups: list[list[int]] = []
    for i in sorted(ranked):
        if groups and i - groups[-1][-1] <= 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    # B1 stores tokens via LXT's clean_tokens, which escapes markdown/LaTeX characters
    spans = [_LATEX_ESCAPE.sub(r"\1", "".join(text[i] for i in group)).strip() for group in groups]
    return [s for s in spans if s]


def _b1_claims(row: dict, top_k: int) -> dict:
    claims = {
        "region_shares": {
            "prompt": row.get("mean_prompt_relevance"),
            "reasoning": row.get("mean_reasoning_relevance"),
            "answer_so_far": row.get("mean_answer_relevance"),
        },
    }
    # older B1 runs stored only the region means, without the token arrays
    if row.get("tokens") and row.get("token_relevance"):
        claims["top_prompt_spans"] = _top_spans(
            row["tokens"], row["token_relevance"], row.get("n_prompt_tokens", 0), top_k,
        )
    return claims


def _b2_claims(row: dict, threshold: float) -> dict:
    """Best restoration/destruction score per region, and which clear the bar."""
    best: dict = {}
    for effect in row.get("effects", []):
        key = (effect["direction"], effect["region"])
        if effect["score"] > best.get(key, float("-inf")):
            best[key] = effect["score"]
    causal = sorted({region for (_, region), score in best.items() if score >= threshold})
    return {
        "peak_scores": {f"{d}/{r}": round(s, 4) for (d, r), s in sorted(best.items())},
        "causal_regions": causal,
        "corrupted_field": row.get("corrupted_field"),
    }


def _b3_claims(rows: list[dict]) -> dict:
    causal, non_causal, untested = [], [], []
    for row in rows:
        factor = {"field": row.get("masked_field"), "value": row.get("masked_value")}
        if not row.get("found") or row.get("answer_changed") is None:
            untested.append(factor)
        elif row["answer_changed"]:
            causal.append(factor)
        else:
            non_causal.append(factor)
    return {"causal_factors": causal, "non_causal_factors": non_causal, "untested_factors": untested}


def _b4_claims(rows: list[dict]) -> dict:
    causal, non_causal, inconclusive = [], [], []
    for row in rows:
        mask = row.get("mask", {})
        entry = {"tool": mask.get("tool"), "mode": mask.get("mode")}
        # the model never called the tool, so intercepting it proved nothing
        if not row.get("masked_tool_called") or row.get("answer_changed") is None:
            inconclusive.append(entry)
        elif row["answer_changed"]:
            causal.append(entry)
        else:
            non_causal.append(entry)
    return {"causal_tools": causal, "non_causal_tools": non_causal, "inconclusive_tools": inconclusive}


def _required_stages(setup_id: str, require_b2: bool, b4_applies: bool) -> tuple[str, ...]:
    stages = _STAGES_BY_SETUP.get(setup_id, ("b1", "b2", "b3"))
    if not require_b2:
        stages = tuple(s for s in stages if s != "b2")
    if not b4_applies:
        stages = tuple(s for s in stages if s != "b4")
    return stages


def _format_factor(factor: dict) -> str:
    return f"{factor.get('field')} = {factor.get('value')!r}"


def render_evidence(row: dict) -> str:
    """Verified causes as plain text — the identical payload C1, the LLM judge and
    the human judge all see, so their inputs are comparable by construction."""
    evidence = row.get("evidence", {})
    lines: list[str] = []

    b3 = evidence.get("b3")
    if b3:
        lines.append("Context words tested by removing them and re-running the model:")
        for factor in b3["causal_factors"]:
            lines.append(f"  CAUSAL — removing {_format_factor(factor)} CHANGES the answer.")
        for factor in b3["non_causal_factors"]:
            lines.append(f"  NOT CAUSAL — removing {_format_factor(factor)} leaves the answer unchanged.")
        for factor in b3["untested_factors"]:
            lines.append(f"  UNTESTED — {_format_factor(factor)} does not appear verbatim; no verdict.")

    b4 = evidence.get("b4")
    if b4:
        lines.append("\nTools tested by disabling or corrupting their output and re-running:")
        for tool in b4["causal_tools"]:
            lines.append(f"  CAUSAL — {tool['tool']} ({tool['mode']}): the answer CHANGES.")
        for tool in b4["non_causal_tools"]:
            lines.append(f"  NOT CAUSAL — {tool['tool']} ({tool['mode']}): the answer is unchanged.")
        for tool in b4["inconclusive_tools"]:
            lines.append(f"  INCONCLUSIVE — {tool['tool']}: the model never called it, so nothing was tested.")

    b1 = evidence.get("b1")
    if b1:
        shares = b1["region_shares"]
        pretty = ", ".join(
            f"{name} {value:.0%}" for name, value in shares.items() if value is not None
        )
        if pretty:
            lines.append(
                "\nSUPPORTING (correlational, NOT a verified cause) — attention relevance for "
                f"the answer was distributed as: {pretty}.",
            )
        if b1.get("top_prompt_spans"):
            quoted = "; ".join(f'"{span}"' for span in b1["top_prompt_spans"])
            lines.append(
                f"SUPPORTING (correlational) — highest-relevance parts of the input: {quoted}.",
            )

    b2 = evidence.get("b2")
    if b2:
        regions = ", ".join(b2["causal_regions"]) or "none above threshold"
        lines.append(
            f"\nSUPPORTING (region-level, not factor-level) — activation patching (corrupting "
            f"{b2.get('corrupted_field')}) found the answer causally carried by: {regions}.",
        )

    return "\n".join(lines) if lines else "No causal findings are available for this example."


def build_evidence(config: Config) -> None:
    settings = config.step_c
    require_b2 = bool(settings.get("require_b2", False))
    top_k = int(settings.get("b1_top_k", 10))
    threshold = float(settings.get("b2_causal_threshold", 0.5))
    # a task with no logic masks (e.g. a filtering task calling no arithmetic tools)
    # can never produce B4, so B4 must not be required there
    with open(config.masks["logic"]) as f:
        logic_specs = yaml.safe_load(f) or {}

    for task_name in config.tasks:
        for setup_id in config.setups:
            src = config.result_path(task_name, setup_id)
            if not src.exists():
                logger.warning("MISSING Step A input: %s — run Step A first", src)
                continue
            out_path = config.evidence_path(task_name, setup_id)
            if out_path.exists():
                logger.warning("SKIP (exists): %s — delete the file to regenerate", out_path)
                continue

            b1 = _rows_by_id(config.b1_path(task_name, setup_id))
            b2 = _rows_by_id(config.b2_path(task_name, setup_id))
            b3 = _group_by_id(sorted(Path(config.masks["context_results_dir"]).glob(
                f"{task_name}-{config.model.split('/')[-1]}-{setup_id}-*.jsonl")), setup_id)
            b4 = _group_by_id(sorted(Path(config.masks["logic_results_dir"]).glob(
                f"{task_name}-{config.model.split('/')[-1]}-{setup_id}-*.jsonl")), setup_id)

            required = _required_stages(setup_id, require_b2, bool(logic_specs.get(task_name)))
            available = {"b1": b1, "b2": b2, "b3": b3, "b4": b4}
            out_rows = []
            for row in _load_rows(src):
                row_id = row["id"]
                missing = [stage for stage in required if not available[stage].get(row_id)]
                if missing:
                    logger.warning("SKIP id=%s (%s/%s): no %s result",
                                   row_id, task_name, setup_id, "/".join(missing))
                    continue
                evidence = {}
                if b1.get(row_id):
                    evidence["b1"] = _b1_claims(b1[row_id], top_k)
                if b2.get(row_id):
                    evidence["b2"] = _b2_claims(b2[row_id], threshold)
                if b3.get(row_id):
                    evidence["b3"] = _b3_claims(b3[row_id])
                if b4.get(row_id):
                    evidence["b4"] = _b4_claims(b4[row_id])
                user_turns = [m["content"] for m in row["messages"] if m.get("role") == "user"]
                out_rows.append({
                    "id": row_id,
                    "task_name": task_name,
                    "setup_id": setup_id,
                    "model": config.model,
                    "question": user_turns[-1] if user_turns else None,
                    "answer": row.get("response"),
                    "parsed_answer": row.get("parsed_answer"),
                    "target": row.get("target"),
                    "tool_calls": row.get("tool_calls"),
                    "sources": sorted(evidence),
                    "evidence": evidence,
                })

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for out_row in out_rows:
                    f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            logger.info("wrote %d evidence rows -> %s (required: %s)",
                        len(out_rows), out_path, "+".join(required))
