"""B1 AttnLRP heatmaps CLI: uv run python scripts/plot_b1.py configs/experiment.yaml

Renders task × setup heatmaps (mean prompt/reasoning/answer relevance fraction)
from existing results/attnlrp/*.jsonl files. No model needed.

Add --token-heatmaps to also render whole-text token-level heatmaps (needs
`token_relevance` in the results, added after 2026-08-12 — older results must be
re-run first).
"""
import argparse

from dialexp.config import load_config
from dialexp.plots import plot_b1, plot_b1_token_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Render B1 AttnLRP relevance heatmaps")
    parser.add_argument("config", help="Path to the experiment YAML config")
    parser.add_argument("--token-heatmaps", action="store_true", help="also render whole-text token heatmaps")
    parser.add_argument("--limit", type=int, default=3, help="examples per (task, setup) for --token-heatmaps")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = plot_b1(config)
    if args.token_heatmaps:
        paths += plot_b1_token_heatmaps(config, limit=args.limit)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

