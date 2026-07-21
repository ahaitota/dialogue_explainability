"""Unified HuggingFace inference for the whole project.

A single `HFClient` serves **every** step — Step A generation, the ask-why
baseline, the B3/B4 reruns, Step C synthesis, and answer parsing — so all steps
share the exact same generation path (same model, tokenizer, decoding, tool
loop). Ollama/OpenRouter are not used anywhere in the project.

`HFClient.chat(...)` mirrors BOULDER's client signature and returns BOULDER's
`LLMResult`, so it is a drop-in wherever a BOULDER client was expected (it still
uses BOULDER's `create_tool_handler` to execute and record tool calls).

Tool-call parsing targets the `<tool_call>{...}</tool_call>` convention emitted
by Qwen/Hermes-style chat templates (the project's chosen models), with a fenced
/ bare-JSON fallback. Reasoning is extracted from `<think>...</think>` spans.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable

from boulder.llm.clients import LLMResult
from boulder.response_parser import ResponseParser

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class HFClient:
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device: str = "auto",
        decoding: dict | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.decoding = decoding or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(torch, _DTYPES.get(dtype, "bfloat16"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
        )
        self.model.eval()

    # ---- generation ------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        model: str | None = None,          # ignored; kept for signature parity
        think: bool = False,               # noqa: ARG002 — parity with boulder
        options: dict | None = None,       # noqa: ARG002 — decoding comes from config
        tool_schemas: list[dict] | None = None,
        tool_handler: Callable | None = None,
        max_tool_iterations: int = 10,
    ) -> LLMResult:
        messages = list(messages)
        tool_calls_made: list[dict] = []

        text = ""
        for iteration in range(max_tool_iterations + 1):
            text = self._generate(messages, tool_schemas)
            calls = self._parse_tool_calls(text)
            can_loop = tool_schemas and tool_handler and calls and iteration < max_tool_iterations
            if not can_loop:
                break
            for tool_call in calls:
                name, args, _tid, result = tool_handler(tool_call, messages)
                tool_calls_made.append({"name": name, "arguments": args, "result": result})

        content, reasoning = self._split_reasoning(text)
        return LLMResult(content=content, reasoning=reasoning, tool_calls_made=tool_calls_made)

    def _generate(self, messages: list[dict], tool_schemas: list[dict] | None) -> str:
        torch = self._torch
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas or None,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(**inputs, **self._gen_kwargs())
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _gen_kwargs(self) -> dict[str, Any]:
        temperature = float(self.decoding.get("temperature", 0.0))
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.decoding.get("max_new_tokens", 2048)),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)  # greedy — deterministic paired comparisons
        return kwargs

    # ---- parsing helpers -------------------------------------------------
    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict]:
        calls: list[dict] = []
        for match in _TOOL_CALL_RE.finditer(text):
            try:
                obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            arguments = obj.get("arguments", obj.get("parameters", {}))
            if name:
                calls.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
        return calls

    @staticmethod
    def _split_reasoning(text: str) -> tuple[str, str | None]:
        match = _THINK_RE.search(text)
        if match:
            reasoning = match.group(1).strip()
            content = _THINK_RE.sub("", text).strip()
            return content, reasoning
        return text.strip(), None


class HFResponseParser(ResponseParser):
    """BOULDER's response parser with the HuggingFace backend.

    Reuses every task-specific `parse_*` method and Jinja template; only the LLM
    call (`_parse_with_llm`) is swapped to the shared `HFClient`.
    """

    def __init__(self, hf_client: HFClient, retries: int = 3):
        self.hf = hf_client
        self.temperature = 0.0
        self.retries = retries

    def _parse_with_llm(self, prompt: str, json_field: str | None = None):
        messages = [{"role": "user", "content": prompt}]
        for _ in range(self.retries):
            result = self.hf.chat(messages=messages)
            content = (result.content or "").strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            try:
                obj = json.loads(content)
            except json.JSONDecodeError:
                continue
            return obj.get(json_field) if json_field else obj
        return None
