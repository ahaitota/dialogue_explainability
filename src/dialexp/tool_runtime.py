"""Tool runtime for the project-side inference pipeline.

Reuses BOULDER's native tools and tool handler, and injects the project's custom
arithmetic tools (`dialexp.tools`) at inference time. The frozen benchmark is
tool-agnostic — tools only exist here, when the model runs. This is the single
place where the custom-tools condition is switched on.
"""
from __future__ import annotations

from pathlib import Path

from boulder.domains.db import AttractionDB, HotelDB, RestaurantDB, TrainDB
from boulder.domains.tools import BaseTool
from boulder.inference import get_tools_for_prompt
from boulder.llm import create_tool_handler

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
