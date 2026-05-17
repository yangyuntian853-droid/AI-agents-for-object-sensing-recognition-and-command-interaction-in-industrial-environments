from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import InferenceConfig
from ..presets import DEFAULT_BERT_BACKBONE
from ..schemas.intents import intent_label_list


class BertIntentClassifier:
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
        self.label2id = label2id or {l: i for i, l in enumerate(intent_label_list())}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = str(self.checkpoint) if self.checkpoint and Path(self.checkpoint).exists() else self.backbone
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        num_labels = len(self.label2id)
        if self.checkpoint and Path(self.checkpoint).exists():
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.checkpoint),
            )
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.backbone,
                num_labels=num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        self._model.to(self.device)
        self._model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        if self._model is None:
            self.load()
        import torch

        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
            probs = torch.softmax(out.logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            conf = float(probs[idx].item())
        return self.id2label.get(idx, "other"), conf

    @classmethod
    def from_inference_config(cls, cfg: InferenceConfig) -> BertIntentClassifier | None:
        if cfg.intent_ckpt is None or not Path(cfg.intent_ckpt).exists():
            return None
        return cls(
            backbone=cfg.backbone or DEFAULT_BERT_BACKBONE,
            checkpoint=cfg.intent_ckpt,
            device=cfg.device,
        )
