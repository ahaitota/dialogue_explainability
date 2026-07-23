"""List (and optionally exclude) truncated Step A examples, paired across setups.

uv run python scripts/find_truncated.py configs/experiment.yaml [--write]

`--write` emits results/step_a/excluded.json — {task: [ids]} to drop from BOTH
setups so paired comparisons stay balanced.
"""
import argparse
import logging

from dotenv import load_dotenv

from dialexp.config import load_config
from dialexp.truncation import find_truncated, write_exclusions


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Find truncated Step A rows and exclude them in pairs",
    )
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument(
        "--write", action="store_true",
        help="write results/step_a/excluded.json (pairs to drop from both setups)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    truncated = find_truncated(config)
    total = sum(len(ids) for ids in truncated.values())
    if not total:
        print("No truncated examples found.")
        return

    print("Truncated examples (excluded from BOTH setups):")
    for task, ids in sorted(truncated.items()):
        print(f"  {task}: {sorted(ids)}")
    print(f"Total: {total} example(s) across {len(truncated)} task(s)")

    if args.write:
        out = write_exclusions(config, truncated)
        print(f"Wrote exclusion manifest -> {out}")


if __name__ == "__main__":
    main()
