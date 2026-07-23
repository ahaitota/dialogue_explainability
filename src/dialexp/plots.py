"""Charts for the Step A accuracy evaluation.

Turns the `run_evaluation` summary/paired tables into PNGs that are easy to show:
a grouped bar chart (score per task × setup, one chart per metric) and a paired
delta chart (`dialogue − dialogue-no-tools` per task). Uses a headless backend so
it works on the cluster without a display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402

_HIGHER_IS_BETTER = {"accuracy", "precision"}


def _grouped_bars(summary: list[dict], out_dir: Path) -> list[Path]:
    by_metric: dict[str, list[dict]] = {}
    for row in summary:
        by_metric.setdefault(row["metric"], []).append(row)

    paths = []
    for metric, rows in by_metric.items():
        tasks = sorted({r["task"] for r in rows})
        setups = sorted({r["setup"] for r in rows})
        score = {(r["task"], r["setup"]): r["score"] for r in rows}

        width = 0.8 / max(len(setups), 1)
        fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(tasks) + 2), 4.5))
        for i, setup in enumerate(setups):
            offsets = [t + i * width for t in range(len(tasks))]
            bars = ax.bar(offsets, [score.get((t, setup), 0.0) for t in tasks], width, label=setup)
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

        ax.set_xticks([t + width * (len(setups) - 1) / 2 for t in range(len(tasks))])
        ax.set_xticklabels(tasks, rotation=20, ha="right")
        ax.set_ylabel(metric)
        if metric in _HIGHER_IS_BETTER:
            ax.set_ylim(0, 1.18)  # headroom for value labels + legend
        ax.set_title(f"Step A {metric} by task and setup")
        ax.legend(title="setup", loc="upper right", framealpha=0.9)
        fig.tight_layout()

        path = out_dir / f"step_a_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _paired_delta(paired: list[dict], out_dir: Path) -> list[Path]:
    if not paired:
        return []
    tasks = [p["task"] for p in paired]
    deltas = [p["mean_delta_a_minus_b"] for p in paired]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    # 95% CI whiskers when available (a whisker crossing 0 = not significant).
    has_ci = all(p.get("ci_lo") is not None and p.get("ci_hi") is not None for p in paired)
    xerr = None
    if has_ci:
        lower = [p["mean_delta_a_minus_b"] - p["ci_lo"] for p in paired]
        upper = [p["ci_hi"] - p["mean_delta_a_minus_b"] for p in paired]
        xerr = [lower, upper]

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.6 * len(tasks) + 1.5)))
    ax.barh(tasks, deltas, color=colors, xerr=xerr,
            error_kw={"ecolor": "#333333", "capsize": 3})
    ax.axvline(0, color="black", linewidth=0.8)
    extent = [abs(d) for d in deltas]
    if has_ci:
        extent += [abs(p["ci_lo"]) for p in paired] + [abs(p["ci_hi"]) for p in paired]
    span = max(extent, default=0.1) or 0.1
    ax.set_xlim(-span * 1.55 - 0.08, span * 1.55 + 0.08)  # room for labels past whiskers
    pad = span * 0.05 + 0.015
    for i, p in enumerate(paired):
        d = p["mean_delta_a_minus_b"]
        end = (p["ci_hi"] if d >= 0 else p["ci_lo"]) if has_ci else d
        if d >= 0:
            ax.text(end + pad, i, f"+{d:.2f}", va="center", ha="left", fontsize=9)
        else:
            ax.text(end - pad, i, f"{d:.2f}", va="center", ha="right", fontsize=9)

    setup_a = paired[0]["setup_a"]
    setup_b = paired[0]["setup_b"]
    ax.set_xlabel(f"mean paired delta  ({setup_a} − {setup_b})")
    ax.set_title("Step A paired delta per task")
    fig.tight_layout()

    path = out_dir / "step_a_paired_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def plot_step_a(summary: list[dict], paired: list[dict], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return _grouped_bars(summary, out_dir) + _paired_delta(paired, out_dir)
