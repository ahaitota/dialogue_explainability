"""Step C CLI: uv run python scripts/run_step_c.py configs/experiment.yaml [--phase P]

Phases run in order and each depends on the previous one:
  evidence  — B results -> verified causes (no model, no GPU; safe to run anywhere)
  synthesis — grounded explanations from trace + evidence (needs the model)
  judge     — blind paired scoring vs the evidence, plus paired stats (needs the model)

The two model phases share one loaded HFClient when both are selected.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.c1_synthesis import run_c1
from dialexp.c2_judge import run_c2
from dialexp.config import load_config
from dialexp.evidence import build_evidence

PHASES = ("evidence", "synthesis", "judge")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Step C: evidence -> synthesis -> judging")
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    parser.add_argument("--task", help="restrict to this one task (default: all tasks in the config)")
    parser.add_argument("--setup", help="restrict to this one setup (default: all setups in the config)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    config = load_config(args.config)
    if args.task:
        if args.task not in config.tasks:
            raise ValueError(f"--task {args.task!r} not in config.tasks: {config.tasks}")
        config.tasks = [args.task]
    if args.setup:
        if args.setup not in config.setups:
            raise ValueError(f"--setup {args.setup!r} not in config.setups: {config.setups}")
        config.setups = [args.setup]

    selected = PHASES if args.phase == "all" else (args.phase,)
    if "evidence" in selected:
        build_evidence(config)

    client = None
    if {"synthesis", "judge"} & set(selected):
        from dialexp.hf_client import HFClient
        client = HFClient(
            config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
        )
    if "synthesis" in selected:
        run_c1(config, client=client)
    if "judge" in selected:
        run_c2(config, client=client)


if __name__ == "__main__":
    main()
