"""B1 AttnLRP CLI: uv run python scripts/run_b1.py configs/experiment.yaml

Teacher-forced AttnLRP replay over saved Step A results (pilot scale). Needs GPU +
the model; uses LXT via the transformers-5.x compatibility patch. See docs for
caveats (experimental Qwen3 attribution; reconstructed sequences).

Use --task/--setup to restrict a run to a single (task, setup) pair.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.b1_attnlrp import run_b1
from dialexp.config import load_config


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run B1 AttnLRP attribution over saved Step A results")
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
    run_b1(config)


if __name__ == "__main__":
    main()
