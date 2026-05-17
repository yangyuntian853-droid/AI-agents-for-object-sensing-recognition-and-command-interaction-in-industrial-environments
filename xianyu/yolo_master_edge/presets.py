from __future__ import annotations

import os
from pathlib import Path

from .config import YoloMasterVariant

# 上游仓库：https://github.com/Tencent/YOLO-Master（AGPL-3.0）
UPSTREAM_REPO_URL = "https://github.com/Tencent/YOLO-Master"

# 官方 README 中的预训练权重命名约定（由 Ultralytics 自动拉取或本地 .pt）
_DEFAULT_WEIGHTS: dict[YoloMasterVariant, str] = {
    YoloMasterVariant.V01_N: "yolo_master_n.pt",
    YoloMasterVariant.V01_S: "yolo_master_s.pt",
    YoloMasterVariant.V01_M: "yolo_master_m.pt",
    YoloMasterVariant.V01_L: "yolo_master_l.pt",
    YoloMasterVariant.V01_X: "yolo_master_x.pt",
}

# 相对「YOLO-Master 克隆根目录」的 YAML 路径（从零训练或自定义结构时使用）
_YAML_REL: dict[YoloMasterVariant, str] = {
    YoloMasterVariant.V01_N: "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
    YoloMasterVariant.V01_S: "ultralytics/cfg/models/master/v0_1/det/yolo-master-s.yaml",
    YoloMasterVariant.V01_M: "ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml",
    YoloMasterVariant.V01_L: "ultralytics/cfg/models/master/v0_1/det/yolo-master-l.yaml",
    YoloMasterVariant.V01_X: "ultralytics/cfg/models/master/v0_1/det/yolo-master-x.yaml",
}


def default_pretrained_weights(variant: YoloMasterVariant) -> str:
    """返回传给 `YOLO(...)` 的默认权重标识（.pt 文件名或本地路径）。"""
    return _DEFAULT_WEIGHTS[variant]


def yaml_path_in_repo(variant: YoloMasterVariant, yolo_master_root: Path) -> Path:
    """给定克隆的 YOLO-Master 根目录，解析对应 YAML 绝对路径。"""
    return (yolo_master_root / _YAML_REL[variant]).resolve()


def default_yolo_master_root() -> Path | None:
    """环境变量 `YOLO_MASTER_ROOT` 指向的克隆根目录；未设置则返回 None。"""
    raw = os.environ.get("YOLO_MASTER_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()
