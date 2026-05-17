"""
基于 [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master) 的实时目标检测管线封装。

使用前请在 YOLO-Master 克隆目录内执行 `pip install -r requirements.txt` 与 `pip install -e .`。
上游许可证为 AGPL-3.0，商用请自行合规评估。
"""

from .config import DetectionTrainConfig, YoloMasterVariant
from .pipeline import YoloMasterDetectionPipeline
from .presets import UPSTREAM_REPO_URL, default_pretrained_weights, default_yolo_master_root, yaml_path_in_repo
from .plot_metrics import plot_run_metrics
from .results import detections_from_result

__all__ = [
    "UPSTREAM_REPO_URL",
    "YoloMasterVariant",
    "DetectionTrainConfig",
    "YoloMasterDetectionPipeline",
    "default_pretrained_weights",
    "default_yolo_master_root",
    "yaml_path_in_repo",
    "detections_from_result",
    "plot_run_metrics",
]
