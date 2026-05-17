from __future__ import annotations

import re
from pathlib import Path

from torch.utils.data import Dataset

from ..schemas.slots import SLOT_NAMES
from .protocols import IndustrialRecord, validate_industrial_record
from ._jsonl import load_jsonl

INDUSTRIAL_FORMAT_HINT = (
    "Industrial JSONL 每行: "
    '{"text":"...", "intent":"place", "slots":{"action":"place","object":"滚柱",...}}'
)


def industrial_to_bio(text: str, slots: dict[str, str]) -> list[tuple[str, str]]:
    """
    由 slots 值在 text 中做子串匹配，生成 (span_text, BIO_label) 列表（启发式，非完美对齐）。
    """
    spans: list[tuple[int, int, str]] = []
    for slot_name, value in slots.items():
        if not value or slot_name not in SLOT_NAMES:
            continue
        for m in re.finditer(re.escape(value), text):
            spans.append((m.start(), m.end(), slot_name))
    spans.sort(key=lambda x: x[0])
    used: list[tuple[int, int]] = []
    result: list[tuple[str, str]] = []
    for start, end, slot in spans:
        if any(not (end <= s or start >= e) for s, e in used):
            continue
        used.append((start, end))
        span_text = text[start:end]
        result.append((span_text, f"B-{slot}"))
    return result


class JsonlIndustrialDataset(Dataset):
    def __init__(self, records: list[IndustrialRecord]) -> None:
        self.records = records

    @classmethod
    def from_path(cls, path: Path | None) -> JsonlIndustrialDataset:
        rows = load_jsonl(path, hint=INDUSTRIAL_FORMAT_HINT, validator=validate_industrial_record)
        return cls(rows)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> IndustrialRecord:
        return self.records[idx]

    def intent_labels(self) -> list[str]:
        return sorted({r["intent"] for r in self.records})

    def to_hf_dataset(self):
        from datasets import Dataset

        return Dataset.from_list(self.records)
