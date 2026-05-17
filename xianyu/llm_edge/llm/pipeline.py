from __future__ import annotations

from pathlib import Path

from ..config import DpoTrainConfig, TinytronTrainConfig
from ..data.dpo_dataset import DPO_FORMAT_HINT
from ..data.sft_dataset import SFT_FORMAT_HINT
from .dpo_trainer import TinytronDPOTrainer
from .sft_trainer import TinytronSFTTrainer


class TinytronFinetunePipeline:
    """
    Tinytron-Qwen-0.5B SFT → DPO 编排，与 YoloMasterDetectionPipeline 对称。
    """

    def __init__(
        self,
        sft_config: TinytronTrainConfig | None = None,
        dpo_config: DpoTrainConfig | None = None,
    ) -> None:
        self.sft_config = sft_config or TinytronTrainConfig()
        self.dpo_config = dpo_config or DpoTrainConfig()
        self._sft_trainer = TinytronSFTTrainer(self.sft_config)
        self._dpo_trainer = TinytronDPOTrainer(self.dpo_config)

    def validate_config(self) -> None:
        self._sft_trainer.validate_config()
        self._dpo_trainer.validate_config()

    def print_data_format_hints(self) -> None:
        print("=== SFT ===")
        print(SFT_FORMAT_HINT)
        print("\n=== DPO ===")
        print(DPO_FORMAT_HINT)

    def run_sft(self, data_path: Path | None = None, **kwargs) -> Path:
        if kwargs:
            for k, v in kwargs.items():
                if hasattr(self.sft_config, k):
                    setattr(self.sft_config, k, v)
        if data_path is None:
            self.print_data_format_hints()
            raise FileNotFoundError("未提供 SFT 数据路径（--data）")
        return self._sft_trainer.run(data_path)

    def run_dpo(
        self,
        sft_dir: Path,
        data_path: Path | None = None,
        **kwargs,
    ) -> Path:
        if kwargs:
            for k, v in kwargs.items():
                if hasattr(self.dpo_config, k):
                    setattr(self.dpo_config, k, v)
        self.dpo_config.sft_checkpoint = Path(sft_dir)
        if data_path is None:
            self.print_data_format_hints()
            raise FileNotFoundError("未提供 DPO 数据路径（--data）")
        return self._dpo_trainer.run(data_path, sft_checkpoint=Path(sft_dir))
