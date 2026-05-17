from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import NlpTrainConfig
from ..data.industrial_dataset import JsonlIndustrialDataset
from ..presets import DEFAULT_BERT_BACKBONE, nlp_intent_output_dir, nlp_slot_output_dir
from ..schemas.intents import intent_label_list
from ..schemas.slots import bio_label_list


def _resolve_device(device: str | int) -> str | int:
    if device != "auto":
        return device
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def train_intent(
    data_path: Path,
    config: NlpTrainConfig | None = None,
) -> Path:
    """微调 BERT 意图分类头。"""
    cfg = config or NlpTrainConfig()
    cfg.output_dir = cfg.output_dir / "intent" if cfg.output_dir.name != "intent" else cfg.output_dir
    if cfg.output_dir == Path("artifacts/nlp_runs"):
        cfg.output_dir = nlp_intent_output_dir()

    dataset = JsonlIndustrialDataset.from_path(data_path)
    labels = dataset.intent_labels() or intent_label_list()
    label2id = {l: i for i, l in enumerate(labels)}

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    backbone = cfg.backbone or DEFAULT_BERT_BACKBONE
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=len(label2id),
        id2label={i: l for l, i in label2id.items()},
        label2id=label2id,
    )

    rows = []
    for rec in dataset.records:
        rows.append(
            {
                "text": rec["text"],
                "label_id": label2id[rec["intent"]],
            }
        )

    from datasets import Dataset as HFDataset

    hf_ds = HFDataset.from_list(rows)
    split = hf_ds.train_test_split(test_size=0.1, seed=42)

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)
        enc["labels"] = batch["label_id"]
        return enc

    tokenized = split.map(preprocess, batched=True, remove_columns=split["train"].column_names)
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels_arr, preds),
            "f1_macro": f1_score(labels_arr, preds, average="macro", zero_division=0),
        }

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        **cfg.extra_train_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    out = Path(cfg.output_dir)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    return out


def train_slots(
    data_path: Path,
    config: NlpTrainConfig | None = None,
) -> Path:
    """微调 BERT 槽位序列标注头（由 slots 启发式对齐 BIO）。"""
    cfg = config or NlpTrainConfig()
    if str(cfg.output_dir).endswith("nlp_runs") or cfg.output_dir.name == "nlp_runs":
        cfg.output_dir = nlp_slot_output_dir()

    from ..data.industrial_dataset import industrial_to_bio
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    dataset = JsonlIndustrialDataset.from_path(data_path)
    bio_labels = bio_label_list()
    label2id = {l: i for i, l in enumerate(bio_labels)}
    label_pad_id = -100

    backbone = cfg.backbone or DEFAULT_BERT_BACKBONE
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    rows: list[dict[str, Any]] = []
    for rec in dataset.records:
        text = rec["text"]
        slots = rec.get("slots") or {}
        spans = industrial_to_bio(text, slots)
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=cfg.max_length)
        offsets = enc["offset_mapping"]
        tag_ids = [label2id["O"]] * len(offsets)
        for span_text, bio in spans:
            for i, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue
                piece = text[s:e]
                if piece and span_text.find(piece) >= 0 or piece in span_text:
                    tag_ids[i] = label2id.get(bio, label2id["O"])
        rows.append({"text": text, "tag_ids": tag_ids})

    from datasets import Dataset as HFDataset

    hf_ds = HFDataset.from_list(rows)
    split = hf_ds.train_test_split(test_size=0.1, seed=42)

    model = AutoModelForTokenClassification.from_pretrained(
        backbone,
        num_labels=len(bio_labels),
        id2label={i: l for l, i in label2id.items()},
        label2id=label2id,
    )

    from ..data.collators import build_slot_collator

    collator = build_slot_collator(tokenizer, label_pad_id=label_pad_id)

    def compute_metrics(eval_pred):
        try:
            from seqeval.metrics import f1_score as seq_f1
        except ImportError:
            return {}

        import numpy as np

        predictions, labels_arr = eval_pred
        preds = np.argmax(predictions, axis=2)
        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []
        id2label = {i: l for l, i in label2id.items()}
        for pred_row, label_row in zip(preds, labels_arr):
            t_seq, p_seq = [], []
            for p, lab in zip(pred_row, label_row):
                if lab == label_pad_id:
                    continue
                t_seq.append(id2label.get(int(lab), "O"))
                p_seq.append(id2label.get(int(p), "O"))
            true_labels.append(t_seq)
            pred_labels.append(p_seq)
        return {"f1": seq_f1(true_labels, pred_labels)}

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        **cfg.extra_train_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    out = Path(cfg.output_dir)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    return out
