from __future__ import annotations

SLOT_NAMES: tuple[str, ...] = (
    "action",
    "object",
    "tool",
    "target",
    "position",
    "direction",
)


def bio_label_list() -> list[str]:
    labels = ["O"]
    for slot in SLOT_NAMES:
        labels.append(f"B-{slot}")
        labels.append(f"I-{slot}")
    return labels
