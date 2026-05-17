from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TinytronTrainConfig:
    """Tinytron SFT 超参。"""

    model_id: str = ""  # 空则使用 presets.TINYTRON_HF_ID
    output_dir: Path = field(default_factory=lambda: Path("artifacts/tinytron_runs/sft"))
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 512
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    full_finetune: bool = False
    device: str | int = "auto"
    extra_train_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DpoTrainConfig:
    """Tinytron DPO 超参。"""

    model_id: str = ""
    sft_checkpoint: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("artifacts/tinytron_runs/dpo"))
    beta: float = 0.1
    epochs: int = 1
    learning_rate: float = 5e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 512
    use_lora: bool = True
    device: str | int = "auto"
    extra_train_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class NlpTrainConfig:
    """BERT 意图 / 槽位微调超参。"""

    backbone: str = ""  # 空则使用 presets.DEFAULT_BERT_BACKBONE
    output_dir: Path = field(default_factory=lambda: Path("artifacts/nlp_runs"))
    epochs: int = 5
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_length: int = 128
    device: str | int = "auto"
    extra_train_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    """在线 BERT 推理配置。"""

    intent_ckpt: Path | None = None
    slot_ckpt: Path | None = None
    backbone: str = ""
    use_onnx: bool = False
    onnx_intent_path: Path | None = None
    onnx_slot_path: Path | None = None
    device: str = "cpu"
    use_rule_fallback: bool = True
