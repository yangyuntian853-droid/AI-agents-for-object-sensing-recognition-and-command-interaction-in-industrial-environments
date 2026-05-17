from __future__ import annotations

from typing import Any

from ..config import DpoTrainConfig, TinytronTrainConfig
from ..presets import TINYTRON_HF_ID


def resolve_model_id(cfg: TinytronTrainConfig | DpoTrainConfig) -> str:
    return cfg.model_id or TINYTRON_HF_ID


def load_tokenizer(model_id: str | None = None, **kwargs: Any):
    from transformers import AutoTokenizer

    mid = model_id or TINYTRON_HF_ID
    return AutoTokenizer.from_pretrained(mid, trust_remote_code=True, **kwargs)


def load_causal_lm(model_id: str | None = None, **kwargs: Any):
    from transformers import AutoModelForCausalLM
    import torch

    mid = model_id or TINYTRON_HF_ID
    dtype = kwargs.pop("torch_dtype", torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    return AutoModelForCausalLM.from_pretrained(
        mid,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=kwargs.pop("device_map", "auto"),
        **kwargs,
    )


def apply_lora(model: Any, *, r: int = 8, alpha: int = 16) -> Any:
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)
