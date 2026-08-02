"""Run the pipeline stages in one process, loading the model once.

Chains the implemented stages in dependency order and shares a single `HFClient`
so the (expensive) model is loaded only once. Each stage is idempotent (skips
outputs that already exist), so re-running resumes where a previous run stopped.

    step_a → parser → ask_why → b3 → b4 → evaluation

    uv run python scripts/run_all.py configs/experiment.yaml [--subset dev]
    uv run python scripts/run_all.py configs/experiment.yaml --stages b3 b4
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.ask_why import run_ask_why
from dialexp.b3_context_masking import run_b3
from dialexp.b4_logic_masking import run_b4
from dialexp.config import load_config
from dialexp.evaluation import run_evaluation
from dialexp.hf_client import HFClient
from dialexp.inference import run_step_a
from dialexp.parsing import run_parser

# Canonical order. `evaluation` is the only stage that needs no model.
STAGE_ORDER = ["step_a", "parser", "ask_why", "b3", "b4", "evaluation"]
_MODEL_STAGES = {"step_a", "parser", "ask_why", "b3", "b4"}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run pipeline stages in one process (model loaded once)",
    )
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_ORDER,
        help="subset of stages to run (default: all, in canonical order)",
    )
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

    selected = set(args.stages) if args.stages else set(STAGE_ORDER)

    # Load the model once, only if a model-using stage is selected.
    client = None
    if selected & _MODEL_STAGES:
        client = HFClient(
            config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
        )

    runners = {
        "step_a": lambda: run_step_a(config, client=client),
        "parser": lambda: run_parser(config, client=client),
        "ask_why": lambda: run_ask_why(config, client=client),
        "b3": lambda: run_b3(config, client=client),
        "b4": lambda: run_b4(config, client=client),
        "evaluation": lambda: run_evaluation(config),
    }
    log = logging.getLogger(__name__)
    for stage in STAGE_ORDER:  # always run in canonical order
        if stage in selected:
            log.info("=== stage: %s ===", stage)
            runners[stage]()


if __name__ == "__main__":
    main()
