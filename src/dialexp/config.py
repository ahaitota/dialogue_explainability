"""Minimal experiment config: YAML -> typed dataclass. HuggingFace-only backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    model: str = "Qwen/Qwen3.5-4B"   # HF model id — one model, shared by all steps
    dtype: str = "bfloat16"                     # bfloat16 | float16 | float32
    device: str = "auto"                        # transformers device_map (auto | cpu | cuda | mps)
    tasks: list[str] = field(default_factory=list)
    setups: list[str] = field(default_factory=lambda: ["dialogue", "dialogue-no-tools"])
    include_arithmetic_tools: bool = False
    subset: str | None = None                 # None = full data; "dev" = first dev_size
    dev_size: int = 20
    decoding: dict = field(default_factory=lambda: {"temperature": 0.0, "max_new_tokens": 2048})
    max_tool_iterations: int = 10
    paths: dict = field(default_factory=lambda: {
        "benchmark_dir": "data/benchmark",
        "results_dir": "results/step_a",
        "db_dir": "boulder/data/db",
    })
    # parse answers after generation with the same HFClient (null model = reuse the main model)
    parser: dict = field(default_factory=lambda: {"enabled": False, "model": None})

    def benchmark_path(self, task: str) -> Path:
        return Path(self.paths["benchmark_dir"]) / f"{task}.json"

    def result_path(self, task: str, setup: str) -> Path:
        model_name = self.model.split("/")[-1]
        return Path(self.paths["results_dir"]) / f"{task}-{model_name}-{setup}.jsonl"


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in Config.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**raw)
