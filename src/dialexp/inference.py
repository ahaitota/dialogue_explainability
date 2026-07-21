"""Step A — dialogue response generation (project side, HuggingFace-only).

Reuses BOULDER's prompt construction, tool handler and response-parser logic; the
model backend is the project's unified `HFClient` (the same engine every step
uses). The custom arithmetic tools are injected via `dialexp.tool_runtime`. The
frozen benchmark under `paths.benchmark_dir` is read-only and never regenerated.
"""
from __future__ import annotations

import json
import logging

from boulder.inference import (
    build_chat_messages,
    extract_targets,
    get_prompt_templates,
    load_datasets,
    parse_results,
)

from dialexp.config import Config
from dialexp.hf_client import HFClient, HFResponseParser
from dialexp.tool_runtime import build_tools_for_setup, load_dbs, make_runtime

logger = logging.getLogger(__name__)


def _select_examples(config: Config, examples: list[dict]) -> list[dict]:
    if config.subset == "dev":
        return examples[: config.dev_size]
    return examples


def run_step_a(config: Config) -> None:
    client = HFClient(
        config.model, dtype=config.dtype, device=config.device, decoding=config.decoding,
    )

    parser = None
    if config.parser.get("enabled"):
        parser_model = config.parser.get("model")
        parser_client = (
            client if not parser_model or parser_model == config.model
            else HFClient(parser_model, dtype=config.dtype, device=config.device, decoding=config.decoding)
        )
        parser = HFResponseParser(parser_client)

    results_dir = config.result_path(config.tasks[0], config.setups[0]).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    selected_setups = get_prompt_templates(config.setups)

    for task_name in config.tasks:
        dataset = load_datasets([str(config.benchmark_path(task_name))])[0]
        answer_type = dataset["answer_type"]
        examples = _select_examples(config, dataset["examples"])
        dbs = load_dbs(task_name, config.paths["db_dir"])

        for setup_id, dialogue_template, _is_dialogue in selected_setups:
            out_path = config.result_path(task_name, setup_id)
            if out_path.exists():
                logger.info("skip (exists): %s", out_path)
                continue

            tools = build_tools_for_setup(
                setup_id, dbs, include_arithmetic=config.include_arithmetic_tools,
            )
            tool_schemas, tool_handler = make_runtime(tools)

            rows = []
            for example in examples:
                _prompt, chat_messages, _is_dlg = build_chat_messages(
                    example, setup_id, dialogue_template, answer_type,
                )
                chat_kwargs = dict(messages=chat_messages)
                if tool_schemas:
                    chat_kwargs.update(
                        tool_schemas=tool_schemas,
                        tool_handler=tool_handler,
                        max_tool_iterations=config.max_tool_iterations,
                    )
                result = client.chat(**chat_kwargs)
                targets, parser_enum = extract_targets(example, answer_type)
                rows.append({
                    "id": example["id"],
                    "setup_id": setup_id,
                    "model": config.model,
                    "task_name": task_name,
                    "answer_type": answer_type,
                    "messages": chat_messages,
                    "cot": result.reasoning,
                    "response": result.content,
                    "target": targets,
                    "tool_calls": result.tool_calls_made,
                    "parser_enum": parser_enum,
                })

            with open(out_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("wrote %d rows -> %s", len(rows), out_path)

            if parser is not None:
                parse_results(str(out_path), answer_type, parser)
                logger.info("parsed answers -> %s", out_path)

