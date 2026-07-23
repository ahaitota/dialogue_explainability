"""Run the pipeline stages in one process, loading the model once.

Currently chains **Step A generation → ask-why baseline**. Ask-why reads Step A's
saved results, so order matters; both stages share a single `HFClient`, so the
(expensive) model is loaded only once instead of once per script.

uv run python scripts/run_all.py configs/experiment.yaml [--subset dev]
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.ask_why import run_ask_why
from dialexp.config import load_config
from dialexp.hf_client import HFClient
from dialexp.inference import run_step_a


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run Step A then the ask-why baseline in one process (model loaded once)",
    )
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument(
        "--subset", choices=["dev"], help="Override subset (dev = first dev_size examples)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    if args.subset:
        config.subset = args.subset

    client = HFClient(
        config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
    )
    run_step_a(config, client=client)
    run_ask_why(config, client=client)


if __name__ == "__main__":
    main()
