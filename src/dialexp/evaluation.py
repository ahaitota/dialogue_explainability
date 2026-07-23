"""Step A accuracy evaluation — per task, per setup, paired across setups.

Reuses BOULDER's per-task scorers (`boulder.evaluation.evaluators.TASK_EVALUATORS`,
dispatched by `answer_type`) and adds the project's paired layer: the two setups
(`dialogue` vs `dialogue-no-tools`) are compared on the SAME example ids, and
truncated examples are excluded in pairs (`results/step_a/excluded.json`, or a
live `finish_reason` scan as fallback). This is descriptive task accuracy — NOT
the Step C faithfulness judge.

Reads `results/step_a/*.jsonl` (needs `parsed_answer` — run the parser first) and
writes `results/evaluation/step_a_metrics.csv` (+ `step_a_paired.csv`).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from boulder.evaluation.evaluators import TASK_EVALUATORS

from dialexp.config import Config
from dialexp.significance import paired_bootstrap
from dialexp.truncation import find_truncated

logger = logging.getLogger(__name__)


def _round(x, ndigits: int = 4):
    return round(x, ndigits) if x is not None else None


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_exclusions(config: Config) -> dict[str, set]:
    manifest = Path(config.paths["results_dir"]) / "excluded.json"
    if manifest.exists():
        raw = json.loads(manifest.read_text())
        return {task: set(ids) for task, ids in raw.items()}
    # No manifest written yet — derive truncated ids live so pairs still balance.
    return find_truncated(config)


def _eval_dir(config: Config) -> Path:
    return Path(config.paths["results_dir"]).parent / "evaluation"


def run_evaluation(config: Config, write: bool = True, plots: bool = True) -> tuple[list[dict], list[dict]]:
    exclusions = _load_exclusions(config)
    summary: list[dict] = []
    paired: list[dict] = []

    for task_name in config.tasks:
        excluded = exclusions.get(task_name, set())
        scores_by_setup: dict[str, dict] = {}

        for setup_id in config.setups:
            path = config.result_path(task_name, setup_id)
            if not path.exists():
                logger.warning("MISSING: %s — run Step A first", path)
                continue
            rows = _load_rows(path)
            if any("parsed_answer" not in r for r in rows):
                logger.warning(
                    "%s has unparsed rows — run scripts/run_parser.py first", path,
                )
            items = [r for r in rows if r["id"] not in excluded]
            if not items:
                continue
            answer_type = items[0]["answer_type"]
            evaluator = TASK_EVALUATORS.get(answer_type)
            if evaluator is None:
                logger.warning("no evaluator for answer_type=%s (%s)", answer_type, task_name)
                continue
            result = evaluator(items)
            scores_by_setup[setup_id] = {it["id"]: float(s) for it, s in zip(items, result.scores)}
            summary.append({
                "task": task_name,
                "setup": setup_id,
                "answer_type": answer_type,
                "metric": result.metric,
                "n": len(items),
                "n_excluded": len(excluded),
                "score": round(float(result.average_score), 4),
            })

        # Paired significance between the two configured setups on shared ids only.
        if len(scores_by_setup) == 2:
            (a_id, a), (b_id, b) = scores_by_setup.items()
            common = sorted(set(a) & set(b))
            if common:
                test = paired_bootstrap([a[i] - b[i] for i in common])
                paired.append({
                    "task": task_name,
                    "setup_a": a_id,
                    "setup_b": b_id,
                    "n_paired": len(common),
                    "mean_a": round(sum(a[i] for i in common) / len(common), 4),
                    "mean_b": round(sum(b[i] for i in common) / len(common), 4),
                    "mean_delta_a_minus_b": round(test["delta"], 4),
                    "ci_lo": _round(test["ci_lo"]),
                    "ci_hi": _round(test["ci_hi"]),
                    "p_value": _round(test["p_value"]),
                    "sig": test["sig"],
                })

    _report(summary, paired)
    if write:
        _write(config, summary, paired)
    if plots and summary:
        from dialexp.plots import plot_step_a

        paths = plot_step_a(summary, paired, _eval_dir(config))
        for path in paths:
            logger.info("wrote chart -> %s", path)
    return summary, paired


def _report(summary: list[dict], paired: list[dict]) -> None:
    print("\n=== Step A metrics (per task/setup) ===")
    print(f"{'task':<28} {'setup':<20} {'metric':<9} {'n':>3} {'excl':>4} {'score':>8}")
    for r in summary:
        print(
            f"{r['task']:<28} {r['setup']:<20} {r['metric']:<9} "
            f"{r['n']:>3} {r['n_excluded']:>4} {r['score']:>8}",
        )
    if paired:
        print("\n=== Paired (same examples, delta = setup_a - setup_b) ===")
        print("(accuracy/precision: higher is better; mae: lower is better; "
              "95% CI + bootstrap p, H0: delta=0)")
        for p in paired:
            ci = (f"[{p['ci_lo']:+}, {p['ci_hi']:+}]" if p["ci_lo"] is not None else "[n/a]")
            pv = f"{p['p_value']}" if p["p_value"] is not None else "n/a"
            print(
                f"{p['task']:<28} {p['setup_a']} vs {p['setup_b']}: "
                f"n={p['n_paired']} delta={p['mean_delta_a_minus_b']:+} "
                f"95%CI={ci} p={pv} {p['sig']}",
            )


def _write(config: Config, summary: list[dict], paired: list[dict]) -> None:
    out_dir = _eval_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "step_a_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task", "setup", "answer_type", "metric", "n", "n_excluded", "score"],
        )
        writer.writeheader()
        writer.writerows(summary)

    if paired:
        paired_csv = out_dir / "step_a_paired.csv"
        with open(paired_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task", "setup_a", "setup_b", "n_paired",
                    "mean_a", "mean_b", "mean_delta_a_minus_b",
                    "ci_lo", "ci_hi", "p_value", "sig",
                ],
            )
            writer.writeheader()
            writer.writerows(paired)
    logger.info("wrote metrics -> %s", metrics_csv)
