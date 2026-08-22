"""B2 activation patching CLI: uv run python scripts/run_b2.py configs/experiment.yaml

Teacher-forced clean/corrupted replay over saved Step A results, patching whole
decoder-layer outputs. Needs GPU + the model. Use --task/--setup to restrict a run
to a single (task, setup) pair.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.b2_patching import run_b2
from dialexp.config import load_config


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run B2 activation patching over saved Step A results")
    parser.add_argument("config", help="Path to the experiment YAML config")
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
    run_b2(config)


if __name__ == "__main__":
    main()
