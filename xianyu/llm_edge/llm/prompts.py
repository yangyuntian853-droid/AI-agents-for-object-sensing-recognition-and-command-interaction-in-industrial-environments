from __future__ import annotations

INDUSTRIAL_SYSTEM_PROMPT = """你是工业场景自然语言指令解析助手。
将用户的口语指令解析为 JSON，字段包括：
intent（retrieve|place|move|grasp|assemble|other）、
action、object、tool、target、position、direction。
只输出合法 JSON，不要多余说明。"""


def build_sft_messages(user_text: str, assistant_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INDUSTRIAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_json},
    ]
