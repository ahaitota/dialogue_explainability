"""B3 context-masking CLI: uv run python scripts/run_b3.py configs/experiment.yaml

Reruns both setups with a benchmark-derived constraint word removed from the user
turns; compares each rerun to Step A. Needs parsed Step A results.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.b3_context_masking import run_b3
from dialexp.config import load_config


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run B3 context masking over saved Step A results")
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_b3(load_config(args.config))


if __name__ == "__main__":
    main()
