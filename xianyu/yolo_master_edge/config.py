from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class YoloMasterVariant(str, Enum):
    """v0.1 检测档位（与上游 `ultralytics/cfg/models/master/v0_1/det` 对齐）。"""

    V01_N = "v0.1-n"
    V01_S = "v0.1-s"
    V01_M = "v0.1-m"
    V01_L = "v0.1-l"
    V01_X = "v0.1-x"


@dataclass
class DetectionTrainConfig:
    """常用训练超参子集；其余通过 `extra_train_kwargs` 透传给 `model.train`。"""

    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str | int | list[int] = "0"
    project: Path = field(default_factory=lambda: Path("artifacts/yolo_master_runs"))
    name: str = "train"
    plots: bool = True
    extra_train_kwargs: dict[str, Any] = field(default_factory=dict)
