from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import TinytronTrainConfig
from ..data.sft_dataset import SFT_FORMAT_HINT, JsonlSFTDataset
from ..presets import tinytron_sft_output_dir
from .model_loader import apply_lora, load_causal_lm, load_tokenizer, resolve_model_id


class TinytronSFTTrainer:
    def __init__(self, config: TinytronTrainConfig | None = None) -> None:
        self.config = config or TinytronTrainConfig()

    def validate_config(self) -> None:
        if not self.config.output_dir:
            self.config.output_dir = tinytron_sft_output_dir()

    @staticmethod
    def print_format_hint() -> None:
        print(SFT_FORMAT_HINT)

    def run(self, data_path: Path | None = None) -> Path:
        if data_path is None:
            raise FileNotFoundError(
                "SFT 需要 --data 指向 JSONL 文件。\n" + SFT_FORMAT_HINT
            )
        self.validate_config()
        cfg = self.config
        model_id = resolve_model_id(cfg)
        dataset = JsonlSFTDataset.from_path(data_path)
        hf_ds = dataset.to_hf_dataset()

        tokenizer = load_tokenizer(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = load_causal_lm(model_id)
        if cfg.use_lora and not cfg.full_finetune:
            model = apply_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)

        def formatting_func(example: dict[str, Any]) -> str:
            messages = example["messages"]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        try:
            from trl import SFTConfig, SFTTrainer
        except ImportError as e:
            raise RuntimeError("SFT 需要安装 trl: pip install trl") from e

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        training_args = SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_length=cfg.max_seq_len,
            logging_steps=10,
            save_strategy="epoch",
            **cfg.extra_train_kwargs,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_ds,
            processing_class=tokenizer,
            formatting_func=formatting_func,
        )
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        return out_dir
