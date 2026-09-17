"""Re-read saved judge replies with the current parser, no model needed.

The judge frequently states its verdict in prose rather than the JSON it was asked
for. Those replies were kept on the row (`judge_calls[arm].raw_judge_output`), so a
parser improvement can recover them without spending GPU time re-judging. Rewrites
results/judgements/*.jsonl in place and regenerates summary.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dialexp.c2_judge import ARMS, _has_distractor, _parse_scores, _summarise
from dialexp.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)
    judged_all: list[dict] = []
    recovered = still_failed = 0

    for task_name in config.tasks:
        for setup_id in config.setups:
            path = config.judgement_path(task_name, setup_id)
            if not path.exists():
                continue
            evidence = {}
            evidence_path = config.evidence_path(task_name, setup_id)
            if evidence_path.exists():
                with open(evidence_path) as f:
                    evidence = {json.loads(line)["id"]: json.loads(line)
                                for line in f if line.strip()}

            rows = [json.loads(line) for line in open(path) if line.strip()]
            for row in rows:
                if row.get("scores"):
                    continue
                calls = row.get("judge_calls") or {}
                scores = dict(row.get("partial_scores") or {})
                for arm in ARMS:
                    call = calls.get(arm) or {}
                    if call.get("raw_judge_output") is None:
                        continue  # nothing saved for this arm
                    parsed = _parse_scores(call["raw_judge_output"])
                    if parsed:
                        scores[arm] = parsed
                if len(scores) < len(ARMS):
                    still_failed += 1
                    continue
                row["scores"] = scores
                if "has_distractor" not in row and row["id"] in evidence:
                    row["has_distractor"] = _has_distractor(evidence[row["id"]])
                for arm in ARMS:
                    calls[arm]["recovered_from_prose"] = True
                recovered += 1

            if not args.dry_run:
                with open(path, "w") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            judged_all.extend(rows)

    complete = sum(1 for r in judged_all if r.get("scores"))
    logger.info("recovered %d rows from saved replies; %d still unparsable", recovered, still_failed)
    logger.info("complete rows: %d of %d", complete, len(judged_all))

    if args.dry_run:
        logger.warning("dry run — nothing written")
        return
    summary_path = Path(config.step_c["judgements_dir"]) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_summarise(judged_all), f, indent=2)
    logger.info("rewrote %s", summary_path)


if __name__ == "__main__":
    main()
