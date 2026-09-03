"""Data loading for the inspection UI: locates result files and caches their rows.

Every loader tolerates missing files, because stages are run at different times
and B2 / Step C may not exist yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dialexp.config import Config, load_config

# `attnlrp/` predates the token-level fields and cannot render heatmaps, so it is
# not offered here; delete it once the newer runs are confirmed good.
B1_VARIANTS = {
    "answer_value (value-targeted)": "results/attnlrp_answer_value",
    "full (whole answer)": "results/attnlrp_full",
}


@st.cache_data
def get_config(path: str = "configs/experiment.yaml") -> Config:
    return load_config(path)


@st.cache_data
def load_rows(path_str: str) -> list[dict]:
    path = Path(path_str)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def model_name(config: Config) -> str:
    return config.model.split("/")[-1]


def stem(config: Config, task: str, setup: str) -> str:
    return f"{task}-{model_name(config)}-{setup}"


def stage_path(config: Config, directory: str, task: str, setup: str) -> Path:
    return Path(directory) / f"{stem(config, task, setup)}.jsonl"


def masked_paths(config: Config, directory: str, task: str, setup: str) -> list[Path]:
    """B3/B4 write one file per mask; the dialogue glob also matches no-tools."""
    return sorted(Path(directory).glob(f"{stem(config, task, setup)}-*.jsonl"))


def load_masked(config: Config, directory: str, task: str, setup: str) -> list[dict]:
    rows: list[dict] = []
    for path in masked_paths(config, directory, task, setup):
        rows += [r for r in load_rows(str(path)) if r.get("setup_id") == setup]
    return rows


def row_by_id(rows: list[dict], row_id) -> dict | None:
    return next((r for r in rows if r.get("id") == row_id), None)


def example_ids(config: Config, task: str, setup: str) -> list:
    return [r["id"] for r in load_rows(str(stage_path(config, "results/step_a", task, setup)))]


def inventory(config: Config) -> list[dict]:
    """One row per (stage, task, setup) with how many records exist."""
    stages = [
        ("Step A", "results/step_a", "file"),
        ("Ask-why", "results/ask_why", "file"),
        ("B1 AttnLRP", B1_VARIANTS["answer_value (value-targeted)"], "file"),
        ("B2 patching", config.b2["results_dir"], "file"),
        ("B3 context masking", config.masks["context_results_dir"], "glob"),
        ("B4 logic masking", config.masks["logic_results_dir"], "glob"),
        ("C evidence", config.step_c["evidence_dir"], "file"),
        ("C explanations", config.step_c["explanations_dir"], "file"),
        ("C judgements", config.step_c["judgements_dir"], "file"),
    ]
    out = []
    for label, directory, kind in stages:
        for task in config.tasks:
            for setup in config.setups:
                if kind == "file":
                    n = len(load_rows(str(stage_path(config, directory, task, setup))))
                else:
                    n = len(load_masked(config, directory, task, setup))
                out.append({"stage": label, "task": task, "setup": setup, "rows": n})
    return out
