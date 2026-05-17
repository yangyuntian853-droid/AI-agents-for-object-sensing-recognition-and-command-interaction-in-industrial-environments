from __future__ import annotations

import re

from ..schemas.instruction import IndustrialCommand
from ..schemas.intents import IndustrialIntent

# 意图关键词 → intent / action
_INTENT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"取出|拿|取|给我|递"), IndustrialIntent.RETRIEVE.value, "retrieve"),
    (re.compile(r"放到|放入|放置|放回|搁"), IndustrialIntent.PLACE.value, "place"),
    (re.compile(r"移动|搬|移"), IndustrialIntent.MOVE.value, "move"),
    (re.compile(r"抓|抓取|夹"), IndustrialIntent.GRASP.value, "grasp"),
    (re.compile(r"装配|安装|拧|紧固"), IndustrialIntent.ASSEMBLE.value, "assemble"),
]

_TOOL_PATTERNS = [
  (re.compile(r"扳手"), "扳手"),
  (re.compile(r"螺丝刀"), "螺丝刀"),
  (re.compile(r"滚柱"), "滚柱"),
]

_DIRECTION_PATTERNS = [
    (re.compile(r"左侧|左边"), "左侧"),
    (re.compile(r"右侧|右边"), "右侧"),
]

_POSITION_RE = re.compile(r"第\s*(\d+)\s*格")
_POSITION_CN_RE = re.compile(r"第([一二三四五六七八九十两]+)个?格子?")
_CN_NUM = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10", "两": "2"}


class RuleBasedParser:
    """无权重时的启发式解析（demo / fallback）。"""

    def parse(self, text: str) -> IndustrialCommand:
        text = text.strip()
        intent = IndustrialIntent.OTHER.value
        action: str | None = None
        for pat, int_val, act in _INTENT_PATTERNS:
            if pat.search(text):
                intent = int_val
                action = act
                break

        tool: str | None = None
        obj: str | None = None
        for pat, name in _TOOL_PATTERNS:
            if pat.search(text):
                if name in ("扳手", "螺丝刀"):
                    tool = name
                else:
                    obj = name

        direction: str | None = None
        for pat, val in _DIRECTION_PATTERNS:
            if pat.search(text):
                direction = val
                break

        position: str | None = None
        target: str | None = None
        m_pos = _POSITION_RE.search(text)
        if m_pos:
            position = f"第{m_pos.group(1)}格"
        else:
            m_cn = _POSITION_CN_RE.search(text)
            if m_cn:
                cn = m_cn.group(1)
                num = cn if cn.isdigit() else _CN_NUM.get(cn, cn)
                position = f"第{num}格"
        if "料箱" in text:
            target = "料箱"

        # 滚柱等物体
        if "滚柱" in text and obj is None:
            obj = "滚柱"

        slots: dict[str, str] = {}
        if action:
            slots["action"] = action
        if obj:
            slots["object"] = obj
        if tool:
            slots["tool"] = tool
        if target:
            slots["target"] = target
        if position:
            slots["position"] = position
        if direction:
            slots["direction"] = direction

        return IndustrialCommand(
            raw_text=text,
            intent=intent,
            action=action,
            obj=obj,
            tool=tool,
            target=target,
            position=position,
            direction=direction,
            slots=slots,
            parser_backend="rule",
        )
