"""Charts for the Step A accuracy evaluation.

Turns the `run_evaluation` summary/paired tables into PNGs that are easy to show:
a grouped bar chart (score per task × setup, one chart per metric) and a paired
delta chart (`dialogue − dialogue-no-tools` per task). Uses a headless backend so
it works on the cluster without a display.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

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


def _load_jsonl(path: Path) -> list[dict]:
    import json

    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _heatmap(matrix, tasks: list[str], setups: list[str], metric: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(setups) + 3.0), max(3.0, 0.8 * len(tasks) + 1.2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(setups)))
    ax.set_xticklabels(setups, rotation=20, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    for i in range(len(tasks)):
        for j in range(len(setups)):
            value = matrix[i][j]
            if value is not None:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label=f"mean {metric} relevance")
    ax.set_title(f"B1 AttnLRP: mean {metric} relevance by task and setup", fontsize=11)
    fig.tight_layout()

    path = out_dir / f"attnlrp_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_b1(config, out_dir: Path | None = None) -> list[Path]:
    """Task × setup heatmaps of mean prompt/reasoning/answer(self) relevance
    fractions, read directly from existing results/attnlrp/*.jsonl files."""
    out_dir = Path(out_dir) if out_dir else Path(config.b1["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(config.tasks)
    setups = list(config.setups)
    metrics = ["prompt", "reasoning", "answer"]
    matrices = {metric: [[None] * len(setups) for _ in tasks] for metric in metrics}

    for i, task in enumerate(tasks):
        for j, setup in enumerate(setups):
            path = config.b1_path(task, setup)
            if not path.exists():
                continue
            rows = _load_jsonl(path)
            if not rows:
                continue
            for metric in metrics:
                values = [r[f"mean_{metric}_relevance"] for r in rows if r.get(f"mean_{metric}_relevance") is not None]
                if values:
                    matrices[metric][i][j] = sum(values) / len(values)

    return [_heatmap(matrices[metric], tasks, setups, metric, out_dir) for metric in metrics]


def plot_b1_token_heatmap(row: dict, out_path: Path, fig_width: float = 11.0, fontsize: int = 9) -> Path:
    """Whole-text heatmap for one B1 example: every input token (prompt +
    reasoning + answer) colored by its relevance to the explained answer tokens,
    wrapped across lines like running text. Rendered directly in matplotlib (no
    LaTeX/xelatex, unlike LXT's own `lxt.utils.pdf_heatmap`).

    Needs `row["tokens"]` / `row["token_relevance"]`, added to b1_attnlrp's output
    -- older results (from before this field existed) must be re-run.
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    tokens = row["tokens"]
    values = row["token_relevance"]
    n_prompt = row["n_prompt_tokens"]
    n_reasoning = row["n_reasoning_tokens"]
    value_start = row.get("value_start")
    value_end = row.get("value_end")

    # mark region boundaries as plain (uncolored) labels inline with the text
    items: list[tuple[str, float | None]] = []
    for i, (tok, val) in enumerate(zip(tokens, values)):
        if i == n_prompt:
            items.append(("[REASONING]", None))
        if i == n_prompt + n_reasoning:
            items.append(("[ANSWER]", None))
        if value_start is not None and i == value_start:
            items.append(("[VALUE→]", None))
        items.append((tok.replace("\n", "\\n") or " ", val))
        if value_end is not None and i == value_end - 1:
            items.append(("[←VALUE]", None))

    # monospace char width (~0.6x fontsize in points -> inches) sizes the wrap width
    char_w_in = 0.6 * fontsize / 72
    margin_in = 0.6
    chars_per_line = max(20, int((fig_width - margin_in) / char_w_in))

    lines: list[list[tuple[str, float | None]]] = [[]]
    line_len = 0
    for text, val in items:
        w = len(text) + 1
        if line_len + w > chars_per_line and lines[-1]:
            lines.append([])
            line_len = 0
        lines[-1].append((text, val))
        line_len += w

    # exclude token 0 (BOS/<|im_start|>) from the color scale: attention sinks
    # make the first token disproportionately "relevant" (LXT's own docs note
    # Qwen3 attribution is "skewed toward first token"), which otherwise washes
    # out contrast across the rest of the heatmap. It still renders, just
    # clipped to the same color as the new max.
    vmax = max((v for v in values[1:] if v), default=1.0) or 1.0
    cmap = mpl.colormaps["YlOrRd"]
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    line_h_in = fontsize / 72 * 2.2
    fig_h = max(1.5, line_h_in * len(lines) + 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    ax.set_xlim(0, chars_per_line)
    ax.set_ylim(0, len(lines))
    ax.axis("off")

    for row_i, line in enumerate(lines):
        y = len(lines) - row_i - 0.5
        x = 0.0
        for text, val in line:
            if val is None:
                ax.text(x, y, text, fontsize=fontsize, family="monospace", va="center", ha="left",
                         color="#555555", fontstyle="italic", fontweight="bold")
            else:
                ax.text(x, y, text, fontsize=fontsize, family="monospace", va="center", ha="left",
                         bbox={"facecolor": cmap(norm(val)), "edgecolor": "none", "pad": 1.0})
            x += len(text) + 1

    task = row.get("task_name", "")
    setup = row.get("setup_id", "")
    ex_id = row.get("id", "")
    ax.set_title(f"B1 AttnLRP token relevance — {task} / {setup} / id={ex_id}", fontsize=11)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_b1_token_heatmaps(config, limit: int = 3, out_dir: Path | None = None) -> list[Path]:
    """Renders `plot_b1_token_heatmap` for the first `limit` examples of every
    (task, setup) in `results/attnlrp/*.jsonl`. Rows missing `token_relevance`
    (results from before that field existed) are skipped with a warning."""
    out_dir = Path(out_dir) if out_dir else Path(config.b1["results_dir"]) / "token_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.model.split("/")[-1]
    paths = []
    for task in config.tasks:
        for setup in config.setups:
            src = config.b1_path(task, setup)
            if not src.exists():
                continue
            rows = _load_jsonl(src)
            for row in rows[:limit]:
                if not row.get("token_relevance"):
                    logger.warning(
                        "SKIP token heatmap (no token_relevance): %s id=%s — re-run B1 to regenerate",
                        src, row.get("id"),
                    )
                    continue
                out_path = out_dir / f"{task}-{model_name}-{setup}-id{row['id']}.png"
                paths.append(plot_b1_token_heatmap(row, out_path))
    return paths
