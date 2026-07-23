"""Detect truncated Step A generations and exclude them in pairs.

A row is truncated when its final generation hit `max_new_tokens` instead of
stopping naturally (`finish_reason == "length"`; rows generated before that field
existed fall back to `cot is None`, the thinking-model proxy for "never closed
`<think>`"). Because every analysis is a paired comparison across setups, a
truncated example is dropped from BOTH setups (see docs/project_structure.md
decoding policy).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from dialexp.config import Config

logger = logging.getLogger(__name__)


def _is_truncated(row: dict) -> bool:
    finish_reason = row.get("finish_reason")
    if finish_reason is not None:
        return finish_reason == "length"
    return row.get("cot") is None  # fallback for rows generated before finish_reason


def find_truncated(config: Config) -> dict[str, set]:
    """Return {task_name: {example_id, ...}} truncated in ANY setup."""
    truncated: dict[str, set] = defaultdict(set)
    for task_name in config.tasks:
        for setup_id in config.setups:
            path = config.result_path(task_name, setup_id)
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if _is_truncated(row):
                        truncated[task_name].add(row["id"])
                        logger.debug(
                            "truncated: %s id=%s setup=%s", task_name, row["id"], setup_id,
                        )
    return {task: ids for task, ids in truncated.items() if ids}


def write_exclusions(config: Config, truncated: dict[str, set]) -> Path:
    out = Path(config.paths["results_dir"]) / "excluded.json"
    out.write_text(json.dumps({t: sorted(ids) for t, ids in truncated.items()}, indent=2))
    return out
