"""Standalone answer parsing over saved Step A results.

Step A can parse inline (`parser.enabled: true`), but when results were generated
without it, this stage adds `parsed_answer` to each saved row IN PLACE using the
same `HFResponseParser` (BOULDER's task-specific templates on the HuggingFace
backend). The parsed answer is what evaluation and the B3/B4 answer-change checks
compare, so it must exist before those stages.

Reads/updates `results/step_a/{task}-{model}-{setup}.jsonl` (idempotent: skips
files whose rows already carry `parsed_answer`; `force=True` re-parses).
"""
from __future__ import annotations

import json
import logging

from boulder.inference import parse_results

from dialexp.config import Config
from dialexp.hf_client import HFClient, HFResponseParser

logger = logging.getLogger(__name__)


def _needs_parsing(path) -> bool:
    with open(path) as f:
        for line in f:
            if line.strip() and "parsed_answer" not in json.loads(line):
                return True
    return False


def _answer_type(path) -> str:
    with open(path) as f:
        for line in f:
            if line.strip():
                return json.loads(line)["answer_type"]
    raise ValueError(f"empty results file: {path}")


def run_parser(config: Config, force: bool = False) -> None:
    todo = []
    for task_name in config.tasks:
        for setup_id in config.setups:
            path = config.result_path(task_name, setup_id)
            if not path.exists():
                logger.warning("MISSING Step A input: %s — run Step A first", path)
                continue
            if not force and not _needs_parsing(path):
                logger.warning(
                    "SKIP (already parsed): %s — use --force to re-parse", path,
                )
                continue
            todo.append(path)

    if not todo:
        logger.info("nothing to parse")
        return

    # Only load the (expensive) model once we know there is work to do.
    parser_model = config.parser.get("model")
    client = HFClient(
        config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
    )
    parser_client = (
        client if not parser_model or parser_model == config.model
        else HFClient(parser_model, dtype=config.dtype, device=config.device, decoding=config.decoding)
    )
    parser = HFResponseParser(parser_client)

    for path in todo:
        parse_results(str(path), _answer_type(path), parser)
        logger.info("parsed answers -> %s", path)
