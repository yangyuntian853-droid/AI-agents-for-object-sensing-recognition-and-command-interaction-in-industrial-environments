from __future__ import annotations

from enum import Enum


class IndustrialIntent(str, Enum):
    """工业口语指令意图分类。"""

    RETRIEVE = "retrieve"
    PLACE = "place"
    MOVE = "move"
    GRASP = "grasp"
    ASSEMBLE = "assemble"
    OTHER = "other"


def intent_label_list() -> list[str]:
    return [m.value for m in IndustrialIntent]
