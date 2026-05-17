from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class SFTRecord(TypedDict):
    """
    SFT JSONL 单条记录（Tinytron chat 微调）。

    示例::

        {"messages": [
          {"role": "system", "content": "将工业口语指令解析为 JSON。"},
          {"role": "user", "content": "帮我把滚柱放到料箱的第三个格子中"},
          {"role": "assistant", "content": "{\\"intent\\":\\"place\\",...}"}
        ]}
    """

    messages: list[ChatMessage]


class DPORecord(TypedDict):
    """
    DPO JSONL 单条记录。

    示例::

        {"prompt": "帮我把滚柱放到料箱的第三个格子中",
         "chosen": "{...正确 JSON...}",
         "rejected": "{...错误槽位...}"}
    """

    prompt: str
    chosen: str
    rejected: str


class IndustrialRecord(TypedDict):
    """
    工业指令标注（BERT intent + slot）。

    示例::

        {"text": "帮我把滚柱放到料箱的第三个格子中",
         "intent": "place",
         "slots": {"action": "place", "object": "滚柱", "target": "料箱", "position": "第3格"}}
    """

    text: str
    intent: str
    slots: dict[str, str]
    bio_spans: NotRequired[list[list[str]]]


class BioNerRecord(TypedDict):
    """
    可选 BIO 标注（字符/词级 span）。

    示例::

        {"text": "帮我把滚柱放到料箱的第三个格子中",
         "labels": [["滚柱", "B-object"], ["料箱", "B-target"]]}
    """

    text: str
    labels: list[list[str]]


def validate_sft_record(record: dict[str, Any]) -> SFTRecord:
    if "messages" not in record or not isinstance(record["messages"], list):
        raise ValueError("SFT 记录必须包含非空 messages 列表")
    for msg in record["messages"]:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError("每条 message 需含 role 与 content")
    return record  # type: ignore[return-value]


def validate_dpo_record(record: dict[str, Any]) -> DPORecord:
    for key in ("prompt", "chosen", "rejected"):
        if key not in record or not isinstance(record[key], str):
            raise ValueError(f"DPO 记录必须包含字符串字段 {key!r}")
    return record  # type: ignore[return-value]


def validate_industrial_record(record: dict[str, Any]) -> IndustrialRecord:
    if "text" not in record or not isinstance(record["text"], str):
        raise ValueError("Industrial 记录必须包含 text 字符串")
    if "intent" not in record or not isinstance(record["intent"], str):
        raise ValueError("Industrial 记录必须包含 intent 字符串")
    slots = record.get("slots")
    if slots is not None and not isinstance(slots, dict):
        raise ValueError("slots 必须为 dict[str, str]")
    return record  # type: ignore[return-value]
