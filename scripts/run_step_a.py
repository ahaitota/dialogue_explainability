"""Step A CLI: uv run python scripts/run_step_a.py configs/experiment.yaml [--subset dev]"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.config import load_config
from dialexp.inference import run_step_a


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Step A dialogue response generation")
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("--subset", choices=["dev"], help="Override subset (dev = first dev_size examples)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    if args.subset:
        config.subset = args.subset
    run_step_a(config)


if __name__ == "__main__":
    main()
