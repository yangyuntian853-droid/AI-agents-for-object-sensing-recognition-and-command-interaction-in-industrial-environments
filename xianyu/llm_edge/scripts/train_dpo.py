from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_edge.config import DpoTrainConfig
from llm_edge.data.dpo_dataset import DPO_FORMAT_HINT
from llm_edge.llm.pipeline import TinytronFinetunePipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tinytron DPO 微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DPO_FORMAT_HINT,
    )
    p.add_argument("--data", type=str, default=None, help="DPO JSONL 路径")
    p.add_argument("--sft-checkpoint", type=str, required=False, help="SFT 输出目录")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args(argv)

    if args.data is None:
        print("错误: 请提供 --data 指向 DPO JSONL 文件。\n")
        print(DPO_FORMAT_HINT)
        return 1
    if args.sft_checkpoint is None:
        print("错误: 请提供 --sft-checkpoint（SFT 阶段输出目录）")
        return 1

    cfg = DpoTrainConfig(beta=args.beta, sft_checkpoint=Path(args.sft_checkpoint))
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    pipeline = TinytronFinetunePipeline(dpo_config=cfg)
    out = pipeline.run_dpo(Path(args.sft_checkpoint), Path(args.data))
    print(f"DPO 完成，输出目录: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
