from __future__ import annotations

from typing import Any


def detections_from_result(result: Any) -> list[dict[str, Any]]:
    """
    将单张图的 Ultralytics `Results` 转为纯 Python 结构（便于 JSON / 下游 LLM）。

    依赖上游 `result.boxes` API；无检测框时返回空列表。
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names: dict[int, str] = getattr(result, "names", {}) or {}
    out: list[dict[str, Any]] = []

    def _row_tensor(attr: str) -> Any:
        t = getattr(boxes, attr)
        return t.cpu() if hasattr(t, "cpu") else t

    xyxy = _row_tensor("xyxy").tolist()
    confs = _row_tensor("conf").tolist()
    clss = _row_tensor("cls").tolist()
    for i in range(len(boxes)):
        cid = int(clss[i])
        out.append(
            {
                "xyxy": [float(x) for x in xyxy[i]],
                "confidence": float(confs[i]),
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
            }
        )
    return out
