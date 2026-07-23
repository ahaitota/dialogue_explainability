"""Step A evaluation CLI: uv run python scripts/run_evaluation.py configs/experiment.yaml

Descriptive task accuracy per setup, paired across setups, over saved Step A
results. Needs `parsed_answer` — run scripts/run_parser.py first if Step A was
generated with the parser disabled.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.config import load_config
from dialexp.evaluation import run_evaluation


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate Step A accuracy per task/setup (paired across setups)",
    )
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("--no-plots", action="store_true", help="skip PNG charts")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_evaluation(load_config(args.config), plots=not args.no_plots)


if __name__ == "__main__":
    main()
