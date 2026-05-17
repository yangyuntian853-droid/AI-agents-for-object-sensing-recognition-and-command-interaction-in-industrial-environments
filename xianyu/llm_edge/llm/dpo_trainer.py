from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import DpoTrainConfig
from ..data.dpo_dataset import DPO_FORMAT_HINT, JsonlDPODataset
from ..presets import tinytron_dpo_output_dir
from .model_loader import apply_lora, load_causal_lm, load_tokenizer, resolve_model_id


class TinytronDPOTrainer:
    def __init__(self, config: DpoTrainConfig | None = None) -> None:
        self.config = config or DpoTrainConfig()

    def validate_config(self) -> None:
        if not self.config.output_dir:
            self.config.output_dir = tinytron_dpo_output_dir()

    @staticmethod
    def print_format_hint() -> None:
        print(DPO_FORMAT_HINT)

    def run(
        self,
        data_path: Path | None,
        *,
        sft_checkpoint: Path | None = None,
    ) -> Path:
        if data_path is None:
            raise FileNotFoundError(
                "DPO 需要 --data 指向 JSONL 文件。\n" + DPO_FORMAT_HINT
            )
        ckpt = sft_checkpoint or self.config.sft_checkpoint
        if ckpt is None or not Path(ckpt).exists():
            raise FileNotFoundError(
                "DPO 需要有效的 SFT checkpoint（--sft-checkpoint 或 config.sft_checkpoint）"
            )

        self.validate_config()
        cfg = self.config
        base_id = resolve_model_id(cfg)
        model_path = str(Path(ckpt).resolve())

        dataset = JsonlDPODataset.from_path(data_path)
        hf_ds = dataset.to_hf_dataset()

        tokenizer = load_tokenizer(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = load_causal_lm(model_path)
        ref_model = load_causal_lm(base_id)
        if cfg.use_lora:
            model = apply_lora(model)

        try:
            from trl import DPOConfig, DPOTrainer
        except ImportError as e:
            raise RuntimeError("DPO 需要安装 trl: pip install trl") from e

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        training_args = DPOConfig(
            output_dir=str(out_dir),
            beta=cfg.beta,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_length=cfg.max_seq_len,
            logging_steps=10,
            save_strategy="epoch",
            **cfg.extra_train_kwargs,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=hf_ds,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        return out_dir
