"""B1 AttnLRP CLI: uv run python scripts/run_b1.py configs/experiment.yaml

Teacher-forced AttnLRP replay over saved Step A results (pilot scale). Needs GPU +
the model; uses LXT via the transformers-5.x compatibility patch. See docs for
caveats (experimental Qwen3 attribution; reconstructed sequences).
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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_b1(load_config(args.config))


if __name__ == "__main__":
    main()
