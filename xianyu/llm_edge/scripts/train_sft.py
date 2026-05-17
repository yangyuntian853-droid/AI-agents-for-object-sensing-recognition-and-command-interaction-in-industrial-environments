from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_edge.config import TinytronTrainConfig
from llm_edge.data.sft_dataset import SFT_FORMAT_HINT
from llm_edge.llm.pipeline import TinytronFinetunePipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tinytron SFT 微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SFT_FORMAT_HINT,
    )
    p.add_argument("--data", type=str, default=None, help="SFT JSONL 路径")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--no-lora", action="store_true", help="禁用 LoRA，全参微调")
    args = p.parse_args(argv)

    if args.data is None:
        print("错误: 请提供 --data 指向 SFT JSONL 文件。\n")
        print(SFT_FORMAT_HINT)
        return 1

    cfg = TinytronTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_lora=not args.no_lora,
        full_finetune=args.no_lora,
    )
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    pipeline = TinytronFinetunePipeline(sft_config=cfg)
    out = pipeline.run_sft(Path(args.data))
    print(f"SFT 完成，输出目录: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
