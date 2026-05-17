from __future__ import annotations

import argparse
import json
import sys

from llm_edge.config import InferenceConfig
from llm_edge.nlp.parser import InstructionParser
from llm_edge.nlp.rule_parser import RuleBasedParser


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="工业指令解析 demo（规则 fallback 或 BERT checkpoint）")
    p.add_argument("--text", required=True, help="原始工业口语指令")
    p.add_argument("--intent-ckpt", type=str, default=None, help="意图模型目录")
    p.add_argument("--slot-ckpt", type=str, default=None, help="槽位模型目录")
    p.add_argument("--rule-only", action="store_true", help="强制使用规则解析")
    args = p.parse_args(argv)

    if args.rule_only:
        cmd = RuleBasedParser().parse(args.text)
    else:
        from pathlib import Path

        cfg = InferenceConfig(
            intent_ckpt=Path(args.intent_ckpt) if args.intent_ckpt else None,
            slot_ckpt=Path(args.slot_ckpt) if args.slot_ckpt else None,
            use_rule_fallback=True,
        )
        cmd = InstructionParser(cfg).parse(args.text)

    print(json.dumps(cmd.model_dump_json_ready(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
