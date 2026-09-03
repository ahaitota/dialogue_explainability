"""Figures for the inspection UI, rendered on demand from result rows.

B1 token heatmaps reuse `plots.plot_b1_token_heatmap` via a temp file so the
tested rendering path is not duplicated. B2 has no plotting in `plots.py`, so its
figures are built here.
"""
from __future__ import annotations

import collections
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

REGION_COLORS = {"fact": "#d62728", "prompt": "#1f77b4", "reasoning": "#2ca02c", "answer": "#ff7f0e"}


@st.cache_data
def b1_token_heatmap_png(row: dict, variant: str) -> str | None:
    """Render one example's whole-text relevance heatmap; returns a PNG path.

    `variant` keys the temp file: the B1 runs share task/setup/id, so without it
    they overwrite each other's PNG and the wrong heatmap is displayed.
    """
    from dialexp.plots import plot_b1_token_heatmap

    if not row.get("tokens") or not row.get("token_relevance"):
        return None
    slug = "".join(c if c.isalnum() else "_" for c in variant)
    out = Path(tempfile.gettempdir()) / (
        f"b1_{slug}_{row.get('task_name')}_{row.get('setup_id')}_{row['id']}.png"
    )
    plot_b1_token_heatmap(row, out)
    return str(out)


def b1_region_figure(rows: list[dict]):
    """Mean prompt/reasoning/answer relevance per example, as stacked bars."""
    ids = [r["id"] for r in rows]
    regions = ("mean_prompt_relevance", "mean_reasoning_relevance", "mean_answer_relevance")
    labels = ("prompt", "reasoning", "answer-so-far")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bottom = [0.0] * len(rows)
    for key, label in zip(regions, labels):
        values = [(r.get(key) or 0.0) for r in rows]
        ax.bar([str(i) for i in ids], values, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xlabel("example id")
    ax.set_ylabel("share of relevance")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def b2_layer_figure(row: dict):
    """Patching score per layer, one line per region, one panel per direction."""
    directions = ("denoising", "noising")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, direction in zip(axes, directions):
        series = collections.defaultdict(list)
        for effect in row["effects"]:
            if effect["direction"] == direction:
                series[effect["region"]].append((effect["layer"], effect["score"]))
        for region, points in series.items():
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker="o", markersize=3, label=region, color=REGION_COLORS.get(region))
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.axhline(1, color="grey", linewidth=0.6, linestyle=":")
        ax.set_title(f"{direction} — {'restores' if direction == 'denoising' else 'destroys'}", fontsize=10)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("normalised effect")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def b2_logit_figure(row: dict, region: str):
    """Heimersheim & Nanda sec. 4.2 check: did the written token's logit rise, or
    did its rival's merely fall?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, direction in zip(axes, ("denoising", "noising")):
        points = sorted(
            (e["layer"], e.get("logit_clean_token"), e.get("logit_corrupt_token"))
            for e in row["effects"]
            if e["direction"] == direction and e["region"] == region
        )
        if not points or points[0][1] is None:
            ax.text(0.5, 0.5, "per-layer logits not recorded\n(re-run B2)",
                    ha="center", va="center", fontsize=9)
            ax.set_axis_off()
            continue
        layers = [p[0] for p in points]
        ax.plot(layers, [p[1] for p in points], marker="o", markersize=3,
                label=f"written token {row.get('clean_token')!r}", color="#2ca02c")
        ax.plot(layers, [p[2] for p in points], marker="o", markersize=3,
                label=f"rival token {row.get('corrupt_token')!r}", color="#d62728")
        ax.set_title(f"{direction} · {region}", fontsize=10)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("logit")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def b3_change_figure(rows: list[dict]):
    """How often removing each field changed the answer."""
    by_field = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        field = row.get("masked_field", "?")
        if not row.get("found") or row.get("answer_changed") is None:
            by_field[field]["no verdict"] += 1
        elif row["answer_changed"]:
            by_field[field]["changed"] += 1
        else:
            by_field[field]["unchanged"] += 1
    fields = sorted(by_field)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(fields) + 1.4))
    bottom = [0] * len(fields)
    for key, color in (("changed", "#2ca02c"), ("unchanged", "#c7c7c7"), ("no verdict", "#f0a3a3")):
        values = [by_field[f][key] for f in fields]
        ax.barh(fields, values, left=bottom, label=key, color=color)
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xlabel("examples")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
