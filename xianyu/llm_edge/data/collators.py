from __future__ import annotations

from typing import Any


def build_intent_collator(tokenizer: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [b["text"] for b in batch]
        labels = [b["label_id"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        import torch

        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


def build_slot_collator(tokenizer: Any, label_pad_id: int = -100):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [b["text"] for b in batch]
        tag_ids = [b["tag_ids"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            is_split_into_words=False,
            return_tensors="pt",
        )
        import torch

        max_len = enc["input_ids"].shape[1]
        padded = []
        for i, tags in enumerate(tag_ids):
            word_ids = enc.word_ids(batch_index=i)
            aligned = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(label_pad_id)
                elif wid != prev:
                    aligned.append(tags[wid] if wid < len(tags) else label_pad_id)
                else:
                    aligned.append(label_pad_id)
                prev = wid
            if len(aligned) < max_len:
                aligned.extend([label_pad_id] * (max_len - len(aligned)))
            else:
                aligned = aligned[:max_len]
            padded.append(aligned)
        enc["labels"] = torch.tensor(padded, dtype=torch.long)
        return enc

    return collate
