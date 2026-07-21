"""Custom arithmetic tools for the dialogue explainability project.

These extend BOULDER's tool interface (`boulder.domains.tools.BaseTool`) so the
model can call them exactly like the native search/booking tools: each exposes an
OpenAI-compatible schema via `get_tool_schema()` and is dispatched by BOULDER's
`create_tool_handler`. Unlike the native tools they are stateless (no DB) and
perform the arithmetic that otherwise happens implicitly in natural language,
turning each computation step into an inspectable "code branch" for trace
analysis (Step B logic masking).

The BOULDER submodule is NOT modified — its inner `boulder` package is exposed
as a top-level import via the project's setuptools config (see pyproject.toml
`[tool.setuptools.packages.find]`), so `from boulder.domains.tools import
BaseTool` works from any directory once the project is installed.
"""
from __future__ import annotations

import pydantic

from boulder.domains.tools import BaseTool


class AddToolParameters(pydantic.BaseModel):
    numbers: list[float] = pydantic.Field(
        ...,
        description="The list of numbers to add together",
    )


class AddTool(BaseTool):
    def __init__(
        self,
        name: str = "add",
        description: str = "Add a list of numbers together and return their sum",
    ):
        super().__init__(name, description, AddToolParameters)

    def __call__(self, parameters: AddToolParameters) -> dict:
        return {"result": float(sum(parameters.numbers))}


class MultiplyToolParameters(pydantic.BaseModel):
    numbers: list[float] = pydantic.Field(
        ...,
        description="The list of numbers to multiply together",
    )


class MultiplyTool(BaseTool):
    def __init__(
        self,
        name: str = "multiply",
        description: str = "Multiply a list of numbers together and return their product",
    ):
        super().__init__(name, description, MultiplyToolParameters)

    def __call__(self, parameters: MultiplyToolParameters) -> dict:
        product = 1.0
        for value in parameters.numbers:
            product *= value
        return {"result": float(product)}


class ApplyDiscountToolParameters(pydantic.BaseModel):
    amount: float = pydantic.Field(
        ...,
        description="The original amount before the discount",
    )
    discount_rate: float = pydantic.Field(
        ...,
        description="The fractional discount to apply, e.g. 0.2 for a 20% discount",
    )


class ApplyDiscountTool(BaseTool):
    def __init__(
        self,
        name: str = "apply_discount",
        description: str = (
            "Apply a fractional discount to an amount and return the discounted "
            "value (amount * (1 - discount_rate))"
        ),
    ):
        super().__init__(name, description, ApplyDiscountToolParameters)

    def __call__(self, parameters: ApplyDiscountToolParameters) -> dict:
        discounted = parameters.amount * (1.0 - parameters.discount_rate)
        return {"result": float(discounted)}


ARITHMETIC_TOOLS = [AddTool, MultiplyTool, ApplyDiscountTool]


def build_arithmetic_tools() -> list[BaseTool]:
    """Instantiate the custom arithmetic tools."""
    return [tool_cls() for tool_cls in ARITHMETIC_TOOLS]
