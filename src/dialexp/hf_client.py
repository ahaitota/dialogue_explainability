"""Unified HuggingFace inference for the whole project.

A single `HFClient` serves **every** step — Step A generation, the ask-why
baseline, the B3/B4 reruns, Step C synthesis, and answer parsing — so all steps
share the exact same generation path (same model, tokenizer, decoding, tool
loop). Ollama/OpenRouter are not used anywhere in the project.

`HFClient.chat(...)` mirrors BOULDER's client signature and returns BOULDER's
`LLMResult`, so it is a drop-in wherever a BOULDER client was expected (it still
uses BOULDER's `create_tool_handler` to execute and record tool calls).

Tool-call parsing handles both the Qwen3.5 XML convention
(`<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`)
and the JSON convention (`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`)
emitted by Hermes-style chat templates. Reasoning is extracted from
`<think>...</think>` spans.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Callable

from boulder.llm.clients import LLMResult
from boulder.response_parser import ResponseParser

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}

# searches for a "total" phrase followed by a currency amount (ex "Total: $101.24")
_TOTAL_AMOUNT_RE = re.compile(r"total[^\n£$]{0,40}[£$]\s*([\d,]+\.?\d*)", re.IGNORECASE)
# searches for a **bold** currency amount (ex "**$17.10**")
_BOLD_AMOUNT_RE = re.compile(r"\*\*[^*\n]*?[£$]\s*([\d,]+\.?\d*)[^*\n]*?\*\*")# list answers are usually a markdown table whose first column holds the venue name
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^[\s|:-]+$")
_MD_EMPHASIS_RE = re.compile(r"\*+")


def restaurant_fallback(text: str | None) -> list[str] | None:
    """Venue names from a markdown table, recording what the model listed — right or wrong.

    Deliberately not filtered against the benchmark target: keeping only names that
    happen to be correct would turn a partly-wrong answer into a perfect one.
    """
    if not text:
        return None
    rows = [(i, m.group(1)) for i, line in enumerate(text.splitlines())
            for m in [_TABLE_ROW_RE.match(line)] if m]
    separators = {i for i, cells in rows if _TABLE_SEP_RE.match(cells)}
    if not separators:
        return None
    headers = {i - 1 for i in separators}
    names = []
    for i, cells in rows:
        if i in separators or i in headers:
            continue
        name = _MD_EMPHASIS_RE.sub("", cells.split("|")[0]).strip().lower()
        if name and name not in names:
            names.append(name)
    return names or None
def amount_fallback(text: str | None) -> float | None:
    """Manually recover a final total from an amount response when the LLM parser fails.

    Relies only on a clear cue — an amount right after a "total" phrase, else the last
    **bold** currency amount.
    """
    if not text:
        return None
    for regex in (_TOTAL_AMOUNT_RE, _BOLD_AMOUNT_RE):
        matches = regex.findall(text)
        if matches:
            try:
                return float(matches[-1].replace(",", ""))
            except ValueError:
                continue
    return None


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
        model_dtype = getattr(torch, _DTYPES.get(dtype, "bfloat16"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=model_dtype, device_map=device,
        )
        self.model.eval()
        self._eos_ids = self._collect_eos_ids()

    # ---- generation ------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        model: str | None = None,          # ignored; kept for signature parity
        think: bool = False,               # noqa: ARG002 — parity with boulder
        options: dict | None = None,       # only max_new_tokens is honoured
        tool_schemas: list[dict] | None = None,
        tool_handler: Callable | None = None,
        max_tool_iterations: int = 10,
    ) -> LLMResult:
        messages = list(messages)
        tool_calls_made: list[dict] = []

        text = ""
        finish_reason = "stop"
        for iteration in range(max_tool_iterations + 1):
            text, finish_reason = self._generate(messages, tool_schemas, options)
            calls = self._parse_tool_calls(text)
            can_loop = tool_schemas and tool_handler and calls and iteration < max_tool_iterations
            if not can_loop:
                break
            for tool_call in calls:
                name, args, _tid, result = tool_handler(tool_call, messages)
                tool_calls_made.append({"name": name, "arguments": args, "result": result})

        content, reasoning = self._split_reasoning(text)
        llm_result = LLMResult(content=content, reasoning=reasoning, tool_calls_made=tool_calls_made)
        # "length" = the final generation hit max_new_tokens (truncated) instead of
        # stopping naturally; recorded per row so truncated pairs can be excluded.
        llm_result.finish_reason = finish_reason
        return llm_result

    def _generate(self, messages: list[dict], tool_schemas: list[dict] | None,
                  options: dict | None = None) -> tuple[str, str]:
        torch = self._torch
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas or None,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = self._gen_kwargs(options)
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        finish_reason = self._finish_reason(new_tokens, gen_kwargs["max_new_tokens"])
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True), finish_reason

    def _collect_eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for source in (self.tokenizer.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)):
            if isinstance(source, (list, tuple)):
                ids.update(int(i) for i in source)
            elif source is not None:
                ids.add(int(source))
        return ids

    def _finish_reason(self, new_tokens, max_new_tokens: int) -> str:
        # Natural stop tokens appear before the cap; only a run that reached the cap
        # without ending on a stop token was actually truncated.
        if len(new_tokens) >= max_new_tokens and int(new_tokens[-1]) not in self._eos_ids:
            return "length"
        return "stop"

    def _gen_kwargs(self, options: dict | None = None) -> dict[str, Any]:
        temperature = float(self.decoding.get("temperature", 0.0))
        max_new_tokens = (options or {}).get("max_new_tokens") or self.decoding.get("max_new_tokens", 2048)
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        # Deterministic anti-repetition: penalises already-generated tokens to break
        # greedy repetition loops without introducing sampling (kept identical across
        # all runs/setups so paired comparisons stay comparable).
        repetition_penalty = float(self.decoding.get("repetition_penalty", 1.0))
        if repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = repetition_penalty
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)  # greedy — deterministic paired comparisons
        return kwargs

    # ---- parsing helpers -------------------------------------------------
    @staticmethod
    def _coerce(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict]:
        blocks = _TOOL_CALL_RE.findall(text)
        # Some models emit bare <function=...> without a <tool_call> wrapper.
        if not blocks and "<function=" in text:
            blocks = [text]

        calls: list[dict] = []
        for block in blocks:
            block = block.strip()
            name: str | None = None
            arguments: dict = {}

            if block.startswith("{"):
                # JSON convention: {"name": ..., "arguments": {...}}.
                try:
                    obj = json.loads(block)
                except json.JSONDecodeError:
                    continue
                name = obj.get("name")
                arguments = obj.get("arguments", obj.get("parameters", {})) or {}
            else:
                # Qwen3.5 XML convention:
                # <function=NAME><parameter=KEY>VALUE</parameter>...</function>
                fmatch = _FUNCTION_RE.search(block)
                if not fmatch:
                    continue
                name = fmatch.group(1)
                arguments = {
                    key: HFClient._coerce(val)
                    for key, val in _PARAMETER_RE.findall(fmatch.group(2))
                }

            if name:
                calls.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
        return calls

    @staticmethod
    def _split_reasoning(text: str) -> tuple[str, str | None]:
        # Explicit <think>...</think> span anywhere in the text.
        match = _THINK_RE.search(text)
        if match:
            reasoning = match.group(1).strip()
            content = _THINK_RE.sub("", text).strip()
            return content, reasoning
        # Thinking models whose chat template already opened <think> in the prompt
        # emit only the closing </think>: everything before it is the reasoning.
        if "</think>" in text:
            reasoning, _, content = text.partition("</think>")
            return content.strip(), reasoning.strip()
        return text.strip(), None


class HFResponseParser(ResponseParser):
    """BOULDER's response parser with the HuggingFace backend.

    Reuses every task-specific `parse_*` method and Jinja template; only the LLM
    call (`_parse_with_llm`) is swapped to the shared `HFClient`.
    """

    def __init__(self, hf_client: HFClient, retries: int = 3, max_new_tokens: int = 4096):
        self.hf = hf_client
        self.temperature = 0.0
        self.retries = retries
        self.max_new_tokens = max_new_tokens

    def parse_answer(self, answer, answer_type, context=None):
        # Deterministic fallback for amount tasks when the LLM parser returns None
        # (recovers a clearly-stated total; leaves genuine non-answers as None).
        result = super().parse_answer(answer, answer_type, context=context)
        if result is None and answer_type == "amount":
            return amount_fallback(answer)
        if result is None and answer_type == "restaurants":
            return restaurant_fallback(answer)
        return result

    def _parse_with_llm(self, prompt: str, json_field: str | None = None):
        messages = [{"role": "user", "content": prompt}]
        last_content, last_finish = "", None
        for attempt in range(self.retries):
            # decoding is greedy, so a repeat of the same call returns the same text;
            # only a larger budget can turn a truncated reply into a parsable one
            result = self.hf.chat(
                messages=messages,
                options={"max_new_tokens": self.max_new_tokens * (attempt + 1)},
            )
            content = (result.content or "").strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            last_content, last_finish = content, getattr(result, "finish_reason", None)
            try:
                obj = json.loads(content)
            except json.JSONDecodeError:
                continue
            return obj.get(json_field) if json_field else obj
        logger.warning("parser produced no JSON after %d attempts (finish=%s) — %r",
                       self.retries, last_finish, last_content[:300])
        return None
