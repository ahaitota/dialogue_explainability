"""Tool runtime for the project-side inference pipeline.

Reuses BOULDER's native tools and tool handler, and injects the project's custom
arithmetic tools (`dialexp.tools`) at inference time. The frozen benchmark is
tool-agnostic — tools only exist here, when the model runs. This is the single
place where the custom-tools condition is switched on.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from boulder.domains.db import AttractionDB, HotelDB, RestaurantDB, TrainDB
from boulder.domains.tools import BaseTool
from boulder.inference import get_tools_for_prompt
from boulder.llm import create_tool_handler
from boulder.llm.utils import parse_tool_arguments

from dialexp.tools import build_arithmetic_tools

# project_root/src/dialexp/tool_runtime.py -> project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = _PROJECT_ROOT / "boulder" / "data" / "db"


def load_dbs(task_name: str, db_dir: str | Path = DEFAULT_DB_DIR) -> dict:
    """Load the four BOULDER domain DBs. The frequency task needs the extended train DB."""
    db_dir = Path(db_dir)
    train_file = "train_db-extended.json" if "frequency" in task_name else "train_db.json"
    return {
        "attraction_db": AttractionDB.from_json(str(db_dir / "attraction_db.json")),
        "hotel_db": HotelDB.from_json(str(db_dir / "hotel_db.json")),
        "restaurant_db": RestaurantDB.from_json(str(db_dir / "restaurant_db.json")),
        "train_db": TrainDB.from_json(str(db_dir / train_file)),
    }


def build_tools_for_setup(
    setup_id: str,
    dbs: dict,
    domains: list[str] | None = None,
    include_arithmetic: bool = False,
) -> list[BaseTool]:
    """Native BOULDER tools for the setup, plus custom arithmetic tools when enabled.

    No-tools setups (`dialogue-no-tools`, etc.) return an empty list, and the
    arithmetic tools are never added to them — they are a tool-condition extension.
    """
    tools = get_tools_for_prompt(
        setup_id,
        dbs["attraction_db"],
        dbs["hotel_db"],
        dbs["restaurant_db"],
        dbs["train_db"],
        domains=domains,
    )
    if include_arithmetic and tools:
        tools = tools + build_arithmetic_tools()
    return tools


def make_runtime(tools: list[BaseTool]):
    """Return `(tool_schemas, tool_handler)` ready to pass to a BOULDER LLM client."""
    tools_dict = {tool.name: tool for tool in tools}
    tool_schemas = [tool.get_tool_schema() for tool in tools]
    tool_handler = create_tool_handler(tools_dict)
    return tool_schemas, tool_handler


# ---- B4 logic masking -----------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _scale_token(text_number: str, factor: float) -> str:
    value = float(text_number) * factor
    return f"{value:.2f}" if "." in text_number else str(int(round(value)))


def corrupt_result(result, factor: float):
    """Multiply every number in a tool result by `factor` (recursively).

    Handles clean numerics (`{"result": 6.84}` from the arithmetic tools) and
    numbers embedded in strings (`"17.10 pounds"` from the domain tools).
    """
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)):
        return result * factor
    if isinstance(result, str):
        return _NUMBER_RE.sub(lambda m: _scale_token(m.group(), factor), result)
    if isinstance(result, list):
        return [corrupt_result(x, factor) for x in result]
    if isinstance(result, dict):
        return {k: corrupt_result(v, factor) for k, v in result.items()}
    return result


def make_masked_handler(tools_dict: dict, masked_tool: str, mode: str, factor: float = 2.0):
    """Tool handler that intervenes on one tool (B4). Mirrors BOULDER's handler
    (same message bookkeeping) but for `masked_tool`:

    - ``disable``: returns an error without executing the tool.
    - ``scale``:  executes the tool, then multiplies every number in the result
      by ``factor`` (corrupted output — tests whether the answer tracks it).

    Every other tool behaves exactly as normal.
    """
    def handle_tool_call(tool_call: dict, messages: list[dict]):
        tool_id = tool_call.get("id", str(uuid.uuid4()))
        tool_name = tool_call["function"]["name"]
        tool_args = parse_tool_arguments(tool_call["function"]["arguments"])

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            }],
        })

        if tool_name == masked_tool and mode == "disable":
            result = {"error": f"Error executing tool {tool_name}: tool unavailable"}
        elif tool_name in tools_dict:
            tool = tools_dict[tool_name]
            try:
                result = tool(tool.parameters(**tool_args))
            except Exception as e:
                result = {"error": f"Error executing tool {tool_name}: {str(e)}"}
            if tool_name == masked_tool and mode == "scale" and "error" not in (result or {}):
                result = corrupt_result(result, factor)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        messages.append({"role": "tool", "id": tool_id, "content": json.dumps(result)})
        return tool_name, tool_args, tool_id, result

    return handle_tool_call


def make_masked_runtime(tools: list[BaseTool], masked_tool: str, mode: str, factor: float = 2.0):
    """Like `make_runtime`, but the handler intervenes on `masked_tool` (B4)."""
    tools_dict = {tool.name: tool for tool in tools}
    tool_schemas = [tool.get_tool_schema() for tool in tools]
    handler = make_masked_handler(tools_dict, masked_tool, mode, factor)
    return tool_schemas, handler
