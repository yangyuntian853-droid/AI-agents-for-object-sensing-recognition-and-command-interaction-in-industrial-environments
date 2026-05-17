from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def export_to_onnx(
    checkpoint: Path,
    output_path: Path,
    *,
    task: str = "intent",
) -> Path:
    """
    将 HuggingFace checkpoint 导出为 ONNX（需 optimum）。
    task: intent | slot
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForTokenClassification
    except ImportError as e:
        raise RuntimeError("导出 ONNX 需要安装: pip install optimum[onnxruntime]") from e

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt = str(checkpoint)
    if task == "intent":
        model = ORTModelForSequenceClassification.from_pretrained(ckpt, export=True)
    else:
        model = ORTModelForTokenClassification.from_pretrained(ckpt, export=True)
    model.save_pretrained(str(output_path))
    return output_path


def benchmark_parse(
    parse_fn: Any,
    text: str,
    *,
    warmup: int = 3,
    runs: int = 20,
) -> dict[str, float]:
    """简单延迟统计（毫秒）。"""
    for _ in range(warmup):
        parse_fn(text)
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        parse_fn(text)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }
