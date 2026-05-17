from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import InferenceConfig
from ..presets import DEFAULT_BERT_BACKBONE
from ..schemas.slots import SLOT_NAMES, bio_label_list


class BertSlotTagger:
    def __init__(
        self,
        *,
        backbone: str | None = None,
        checkpoint: Path | None = None,
        label2id: dict[str, int] | None = None,
        device: str = "cpu",
    ) -> None:
        self.backbone = backbone or DEFAULT_BERT_BACKBONE
        self.checkpoint = checkpoint
        self.device = device
        labels = bio_label_list()
        self.label2id = label2id or {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        path = str(self.checkpoint) if self.checkpoint and Path(self.checkpoint).exists() else self.backbone
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self.checkpoint and Path(self.checkpoint).exists():
            self._model = AutoModelForTokenClassification.from_pretrained(str(self.checkpoint))
        else:
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.backbone,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        self._model.to(self.device)
        self._model.eval()

    def predict_tags(self, text: str) -> list[tuple[str, str]]:
        """返回 (token, bio_label) 列表（字级/词级取决于 tokenizer）。"""
        if self._model is None:
            self.load()
        import torch

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits[0]
            pred_ids = logits.argmax(dim=-1).cpu().tolist()

        tokens = self._tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
        pairs: list[tuple[str, str]] = []
        for tok, pid, (s, e) in zip(tokens, pred_ids, offsets):
            if s == 0 and e == 0:
                continue
            label = self.id2label.get(pid, "O")
            span = text[s:e] if e > s else tok
            if label != "O" and span:
                pairs.append((span, label))
        return pairs

    def predict_slots(self, text: str) -> dict[str, str]:
        pairs = self.predict_tags(text)
        slots: dict[str, str] = {}
        for span, label in pairs:
            if label.startswith("B-") or label.startswith("I-"):
                slot_name = label[2:]
                if slot_name in SLOT_NAMES:
                    key = "object" if slot_name == "object" else slot_name
                    if slot_name not in slots or label.startswith("B-"):
                        slots[slot_name if slot_name != "object" else "object"] = span
        return slots

    @classmethod
    def from_inference_config(cls, cfg: InferenceConfig) -> BertSlotTagger | None:
        if cfg.slot_ckpt is None or not Path(cfg.slot_ckpt).exists():
            return None
        return cls(
            backbone=cfg.backbone or DEFAULT_BERT_BACKBONE,
            checkpoint=cfg.slot_ckpt,
            device=cfg.device,
        )
