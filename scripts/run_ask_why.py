"""Ask-why baseline CLI: uv run python scripts/run_ask_why.py configs/experiment.yaml"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.ask_why import run_ask_why
from dialexp.config import load_config


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run the ask-why self-explanation baseline over saved Step A results",
    )
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_ask_why(load_config(args.config))


if __name__ == "__main__":
    main()
