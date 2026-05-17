from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator


def load_jsonl(
    path: Path | None,
    *,
    hint: str,
    validator: Callable[[dict[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    if path is None:
        raise FileNotFoundError(
            f"未提供数据路径。{hint}\n"
            f"可参考包内 examples/*.jsonl.example 的格式说明。"
        )
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"数据文件不存在: {path}\n{hint}"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 解析失败: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno} 每行须为 JSON 对象")
            if validator is not None:
                validator(obj)
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path} 无有效训练样本（空文件或仅含注释）")
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)
