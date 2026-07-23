"""B4 logic-masking CLI: uv run python scripts/run_b4.py configs/experiment.yaml

Reruns the dialogue setup with one tool's output disabled/corrupted; compares
each rerun to Step A. Requires a Step A run with include_arithmetic_tools: true
and parsed results.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.b4_logic_masking import run_b4
from dialexp.config import load_config


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run B4 logic masking over saved Step A results")
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_b4(load_config(args.config))


if __name__ == "__main__":
    main()
