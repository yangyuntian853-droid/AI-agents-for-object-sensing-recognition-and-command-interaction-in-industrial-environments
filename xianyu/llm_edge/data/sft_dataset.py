from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .protocols import SFTRecord, validate_sft_record
from ._jsonl import load_jsonl

SFT_FORMAT_HINT = (
    "SFT JSONL 每行: "
    '{"messages": [{"role":"system|user|assistant","content":"..."}, ...]}'
)


class SFTDatasetBase(Dataset):
    """SFT 数据集抽象基类。"""

    def formatting_func(self, example: dict[str, Any]) -> str:
        raise NotImplementedError


class JsonlSFTDataset(SFTDatasetBase):
    def __init__(self, records: list[SFTRecord]) -> None:
        self.records = records

    @classmethod
    def from_path(cls, path: Path | None) -> JsonlSFTDataset:
        rows = load_jsonl(path, hint=SFT_FORMAT_HINT, validator=validate_sft_record)
        return cls(rows)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> SFTRecord:
        return self.records[idx]

    def to_hf_dataset(self):
        from datasets import Dataset

        return Dataset.from_list(self.records)
