"""
llm_edge：工业场景语言侧能力。

- **TinytronFinetunePipeline**：Tinytron-Qwen-0.5B SFT / DPO 训练管线（需自备 JSONL）
- **InstructionParser**：轻量 BERT 意图 + 槽位解析（无权重时规则 fallback）
"""

from .config import DpoTrainConfig, InferenceConfig, NlpTrainConfig, TinytronTrainConfig
from .llm.pipeline import TinytronFinetunePipeline
from .nlp.parser import InstructionParser
from .nlp.rule_parser import RuleBasedParser
from .presets import TINYTRON_HF_ID, default_artifacts_root
from .schemas.instruction import IndustrialCommand

__all__ = [
    "TINYTRON_HF_ID",
    "TinytronTrainConfig",
    "DpoTrainConfig",
    "NlpTrainConfig",
    "InferenceConfig",
    "TinytronFinetunePipeline",
    "InstructionParser",
    "RuleBasedParser",
    "IndustrialCommand",
    "default_artifacts_root",
]
