from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .intents import IndustrialIntent


class IndustrialCommand(BaseModel):
    """原始工业指令 → 结构化输出。"""

    raw_text: str
    intent: str = IndustrialIntent.OTHER.value
    action: str | None = None
    obj: str | None = Field(default=None, serialization_alias="object", validation_alias="object")
    tool: str | None = None
    target: str | None = None
    position: str | None = None
    direction: str | None = None
    confidence: float | None = None
    slots: dict[str, str] = Field(default_factory=dict)
    parser_backend: str = "rule"

    model_config = {"populate_by_name": True}

    def model_dump_json_ready(self) -> dict[str, Any]:
        d = self.model_dump(by_alias=True, exclude_none=True)
        d.pop("parser_backend", None)
        return d
