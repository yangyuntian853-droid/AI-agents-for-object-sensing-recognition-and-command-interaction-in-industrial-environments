from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .protocols import DPORecord, validate_dpo_record
from ._jsonl import load_jsonl

DPO_FORMAT_HINT = (
    "DPO JSONL 每行: "
    '{"prompt":"...", "chosen":"...", "rejected":"..."}'
)


class JsonlDPODataset(Dataset):
    def __init__(self, records: list[DPORecord]) -> None:
        self.records = records

    @classmethod
    def from_path(cls, path: Path | None) -> JsonlDPODataset:
        rows = load_jsonl(path, hint=DPO_FORMAT_HINT, validator=validate_dpo_record)
        return cls(rows)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DPORecord:
        return self.records[idx]

    def to_hf_dataset(self):
        from datasets import Dataset

        return Dataset.from_list(self.records)
