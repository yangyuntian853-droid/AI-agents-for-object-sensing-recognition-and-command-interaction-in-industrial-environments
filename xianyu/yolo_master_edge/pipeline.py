from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import DetectionTrainConfig, YoloMasterVariant
from .presets import UPSTREAM_REPO_URL, default_pretrained_weights, default_yolo_master_root, yaml_path_in_repo
from .results import detections_from_result


def _import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "未找到 `ultralytics`。请先在本地克隆 Tencent/YOLO-Master 并在该目录执行 "
            "`pip install -r requirements.txt` 与 `pip install -e .`，"
            "以便使用带 YOLO-Master（MoE 等）改动的 Ultralytics；勿仅 `pip install ultralytics` 覆盖。"
            f" 上游说明见 {UPSTREAM_REPO_URL}"
        ) from e
    return YOLO


class YoloMasterDetectionPipeline:
    """
    基于 [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master) 的目标检测管线封装：

    **加载权重 / YAML → predict / train / val / export**，与 `llm_edge` 并列、便于工业场景「检测 + 语言」组合。

    依赖：必须在环境中安装 **YOLO-Master 仓库内的 editable ultralytics**（见类文档字符串与 `presets.UPSTREAM_REPO_URL`）。
    """

    def __init__(
        self,
        variant: YoloMasterVariant = YoloMasterVariant.V01_N,
        *,
        model_spec: str | Path | None = None,
        yolo_master_repo: Path | None = None,
        train_defaults: DetectionTrainConfig | None = None,
    ) -> None:
        self.variant = variant
        self.yolo_master_repo = yolo_master_repo or default_yolo_master_root()
        self.train_defaults = train_defaults or DetectionTrainConfig()
        self._model_spec = model_spec
        self._model: Any = None

    def _resolved_spec(self) -> str:
        if self._model_spec is None:
            return default_pretrained_weights(self.variant)
        raw = Path(self._model_spec).expanduser()
        if raw.is_file():
            return str(raw.resolve())
        # 裸权重名（如 yolo_master_n.pt）或 Hub 标识；不要用 resolve() 误拼成不存在的 cwd 路径
        return str(self._model_spec)

    def load(self, *, model_spec: str | Path | None = None) -> None:
        """构建底层 `YOLO` 实例（首次推理 / 训练前调用）。"""
        YOLO = _import_yolo()
        if model_spec is not None:
            self._model_spec = model_spec
        self._model = YOLO(self._resolved_spec())

    def load_from_yaml(self, *, yolo_master_repo: Path | None = None) -> None:
        """从上游仓库内的 YAML 构建模型（适合从零训练；需本地已克隆 YOLO-Master）。"""
        root = yolo_master_repo or self.yolo_master_repo
        if root is None:
            raise ValueError(
                "使用 YAML 构建模型时请传入 `yolo_master_repo=...`，或设置环境变量 `YOLO_MASTER_ROOT`。"
            )
        path = yaml_path_in_repo(self.variant, Path(root))
        if not path.is_file():
            raise FileNotFoundError(
                f"未找到 YAML: {path}。请确认已克隆 {UPSTREAM_REPO_URL} 且路径为仓库根目录。"
            )
        YOLO = _import_yolo()
        self._model = YOLO(str(path))

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("请先调用 `load()` 或 `load_from_yaml()`。")
        return self._model

    def predict(self, source: str | Path | Any, **kwargs: Any) -> Any:
        """对图像 / 视频 / 目录等执行推理；`kwargs` 透传 `model.predict`（如 `conf=`, `device=`）。"""
        return self.model.predict(source=source, **kwargs)

    def predict_detections(
        self, source: str | Path | Any, **kwargs: Any
    ) -> list[list[dict[str, Any]]]:
        """推理并返回每张图的检测框列表（纯 dict，无 Tensor）。"""
        results = self.predict(source, **kwargs)
        return [detections_from_result(r) for r in results]

    def train(
        self,
        data: str | Path,
        *,
        train_config: DetectionTrainConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """在 `data`（Ultralytics 数据集 yaml）上训练；`kwargs` 与 `DetectionTrainConfig` 合并后传给 `model.train`。"""
        cfg = train_config or self.train_defaults
        merged: dict[str, Any] = {
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "device": cfg.device,
            "project": str(cfg.project),
            "name": cfg.name,
            "plots": cfg.plots,
        }
        merged.update(cfg.extra_train_kwargs)
        merged.update(kwargs)
        return self.model.train(data=str(data), **merged)

    def val(self, data: str | Path, **kwargs: Any) -> Any:
        """验证集评估。"""
        return self.model.val(data=str(data), **kwargs)

    def export(self, **kwargs: Any) -> Any:
        """导出 ONNX / TensorRT 等；`kwargs` 透传 `model.export`。"""
        return self.model.export(**kwargs)

    def benchmark_predict(
        self,
        source: str | Path | Any,
        *,
        runs: int = 10,
        warmup: int = 1,
        **kwargs: Any,
    ) -> dict[str, float]:
        """对同一 `source` 重复 `predict` 若干次，返回耗时统计（秒）。"""
        for _ in range(max(0, warmup)):
            self.predict(source, **kwargs)
        times: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.predict(source, **kwargs)
            times.append(time.perf_counter() - t0)
        return {
            "runs": float(runs),
            "mean_s": sum(times) / len(times) if times else 0.0,
            "min_s": min(times) if times else 0.0,
            "max_s": max(times) if times else 0.0,
        }
