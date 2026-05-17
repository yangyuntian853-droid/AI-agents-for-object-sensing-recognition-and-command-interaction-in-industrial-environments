from __future__ import annotations

import os
from pathlib import Path

TINYTRON_HF_ID = os.environ.get(
    "TINYTRON_HF_ID",
    "Agnuxo/Tinytron-Qwen-0.5B-Instruct_CODE_Python_Spanish_English_16bit",
)
DEFAULT_BERT_BACKBONE = os.environ.get("LLM_EDGE_BERT_BACKBONE", "hfl/chinese-bert-wwm-ext")
LIGHT_BERT_BACKBONE = "hfl/rbt3"

PACKAGE_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = PACKAGE_ROOT / "examples"


def default_artifacts_root() -> Path:
    raw = os.environ.get("LLM_EDGE_ARTIFACTS", "artifacts")
    return Path(raw).expanduser().resolve()


def tinytron_sft_output_dir() -> Path:
    return default_artifacts_root() / "tinytron_runs" / "sft"


def tinytron_dpo_output_dir() -> Path:
    return default_artifacts_root() / "tinytron_runs" / "dpo"


def nlp_intent_output_dir() -> Path:
    return default_artifacts_root() / "nlp_runs" / "intent"


def nlp_slot_output_dir() -> Path:
    return default_artifacts_root() / "nlp_runs" / "slot"
