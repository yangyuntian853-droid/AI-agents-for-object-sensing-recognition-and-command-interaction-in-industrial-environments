from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_edge.config import NlpTrainConfig
from llm_edge.data.industrial_dataset import INDUSTRIAL_FORMAT_HINT
from llm_edge.nlp.train import train_intent, train_slots


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="BERT 工业指令 intent / slot 微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=INDUSTRIAL_FORMAT_HINT,
    )
    p.add_argument("task", choices=["intent", "slot"], help="训练任务")
    p.add_argument("--data", type=str, default=None, help="Industrial JSONL 路径")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args(argv)

    if args.data is None:
        print("错误: 请提供 --data 指向 Industrial JSONL 文件。\n")
        print(INDUSTRIAL_FORMAT_HINT)
        return 1

    cfg = NlpTrainConfig(epochs=args.epochs)
    if args.backbone:
        cfg.backbone = args.backbone
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    data_path = Path(args.data)
    if args.task == "intent":
        out = train_intent(data_path, cfg)
    else:
        out = train_slots(data_path, cfg)
    print(f"{args.task} 训练完成，输出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
