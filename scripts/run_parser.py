"""Standalone parser CLI: uv run python scripts/run_parser.py configs/experiment.yaml [--force]

Adds `parsed_answer` to saved Step A results in place. Run after Step A and before
evaluation / B3 / B4 when Step A was generated with `parser.enabled: false`.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.config import load_config
from dialexp.parsing import run_parser


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Parse structured answers into saved Step A results (in place)",
    )
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument(
        "--force", action="store_true",
        help="re-parse even if parsed_answer is already present",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_parser(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
