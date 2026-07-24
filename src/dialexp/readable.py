"""Render result JSONL rows into readable Markdown.

Any `results/**/*.jsonl` file has a different schema (Step A, ask-why, B3, B4, B1),
so the renderer is schema-agnostic: it shows the conversation (system prompt
collapsed), the reasoning/answer/explanation text fields, tool calls, and a table
of the remaining scalar fields. Used by `scripts/view_results.py`.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

# Text fields rendered as their own section (label, key, collapsed?).
_TEXT_FIELDS = [
    ("Reasoning (cot)", "cot", True),
    ("Reasoning", "reasoning", True),
    ("Answer", "response", False),
    ("Explanation", "explanation", False),
    ("Explanation reasoning", "explanation_cot", True),
    ("Source answer (explained)", "source_response", False),
]
_HANDLED = {"messages", "tool_calls", *[k for _, k, _ in _TEXT_FIELDS]}


def _fmt_args(args) -> str:
    if isinstance(args, dict):
        return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return str(args)


def _short(text, limit: int = 400) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " …"


def _compact(value) -> str:
    """One-line summary of a list/dict field (restaurant lists → names)."""
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "name" in value[0]:
            names = [v.get("name") for v in value]
            return f"{len(value)} items: " + ", ".join(str(n) for n in names[:8]) + (" …" if len(names) > 8 else "")
        return _short(json.dumps(value, ensure_ascii=False))
    if isinstance(value, dict):
        return _short(json.dumps(value, ensure_ascii=False))
    return str(value)


def _pre(text) -> str:
    return f"<pre>{html.escape(str(text))}</pre>"


def _render_conversation(messages: list) -> list[str]:
    out = ["#### Conversation", ""]
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if role == "system":
            out.append(f"<details><summary>🖥️ <b>system prompt</b></summary>\n\n{_pre(content)}\n\n</details>\n")
        elif role == "user":
            out.append(f"**👤 user:** {content}\n")
        elif role == "assistant":
            if content:
                out.append(f"**🤖 assistant:** {content}\n")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                out.append(f"**🤖 assistant → call:** `{fn.get('name')}({_fmt_args(fn.get('arguments'))})`\n")
        elif role == "tool":
            out.append(f"**🔧 tool result:** `{_short(m.get('content'))}`\n")
    return out


def render_row(row: dict) -> str:
    rid = row.get("id")
    task, setup = row.get("task_name"), row.get("setup_id")
    header = f"### id {rid}"
    if task or setup:
        header += f" — {task or ''}{' / ' + setup if setup else ''}"
    lines = [header, ""]

    messages = row.get("messages")
    if isinstance(messages, list):
        lines += _render_conversation(messages)
        if row.get("_conversation_from_step_a"):
            lines.append("_(conversation shown from the matching Step A row)_\n")

    for label, key, collapsed in _TEXT_FIELDS:
        value = row.get(key)
        if not value:
            continue
        if collapsed:
            lines.append(f"<details><summary>🧠 <b>{label}</b></summary>\n\n{_pre(value)}\n\n</details>\n")
        else:
            lines.append(f"**{label}:**\n\n{value}\n")

    tool_calls = row.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict) and "name" in tool_calls[0]:
        lines.append("**🔧 tool calls made:**\n")
        for tc in tool_calls:
            result = _short(json.dumps(tc.get("result"), ensure_ascii=False))
            lines.append(f"- `{tc.get('name')}({_fmt_args(tc.get('arguments'))})` → `{result}`")
        lines.append("")

    scalars = {k: v for k, v in row.items()
               if k not in _HANDLED and not isinstance(v, (list, dict))
               and not k.startswith("_") and k not in ("id", "task_name", "setup_id")}
    if scalars:
        lines += ["| field | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in scalars.items()]
        lines.append("")

    others = {k: v for k, v in row.items()
              if k not in _HANDLED and isinstance(v, (list, dict)) and not k.startswith("_")}
    for k, v in others.items():
        lines.append(f"- **{k}:** {_compact(v)}")
    if others:
        lines.append("")

    lines.append("---\n")
    return "\n".join(lines)


def _attach_conversations(rows: list[dict], step_a_dir: Path) -> None:
    """For rows without `messages`, pull the conversation from the matching Step A
    row (same task/model/setup/id), so every stage's view shows the input."""
    cache: dict = {}
    for row in rows:
        if row.get("messages") or row.get("id") is None:
            continue
        if not all(row.get(k) for k in ("task_name", "model", "setup_id")):
            continue
        model_name = str(row["model"]).split("/")[-1]
        key = (row["task_name"], model_name, row["setup_id"])
        if key not in cache:
            src = step_a_dir / f"{key[0]}-{key[1]}-{key[2]}.jsonl"
            index: dict = {}
            if src.exists():
                with open(src) as f:
                    for line in f:
                        if line.strip():
                            sr = json.loads(line)
                            index[sr.get("id")] = sr.get("messages")
            cache[key] = index
        messages = cache[key].get(row["id"])
        if messages:
            row["messages"] = messages
            row["_conversation_from_step_a"] = True


def render_file(path, ids: set | None = None, limit: int | None = None, step_a_dir=None) -> str:
    path = Path(path)
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if ids is not None:
        rows = [r for r in rows if r.get("id") in ids]
    if limit is not None:
        rows = rows[:limit]
    if any("messages" not in r for r in rows):
        step_a = Path(step_a_dir) if step_a_dir else path.parent.parent / "step_a"
        if step_a.exists():
            _attach_conversations(rows, step_a)
    body = "\n".join(render_row(r) for r in rows)
    return f"# {path}\n\n_{len(rows)} row(s)_\n\n{body}"
