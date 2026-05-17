from __future__ import annotations

from ..config import InferenceConfig
from ..schemas.instruction import IndustrialCommand
from ..schemas.intents import IndustrialIntent
from ..schemas.slots import SLOT_NAMES
from .intent_model import BertIntentClassifier
from .rule_parser import RuleBasedParser
from .slot_model import BertSlotTagger


def _merge_slots_into_command(cmd: IndustrialCommand, slots: dict[str, str]) -> IndustrialCommand:
    updates: dict[str, str | None] = {}
    field_map = {
        "action": "action",
        "object": "obj",
        "tool": "tool",
        "target": "target",
        "position": "position",
        "direction": "direction",
    }
    merged_slots = dict(cmd.slots)
    merged_slots.update(slots)
    for slot_key, attr in field_map.items():
        if slot_key in slots and slots[slot_key]:
            updates[attr] = slots[slot_key]
    return cmd.model_copy(
        update={
            **updates,
            "slots": merged_slots,
            "parser_backend": "bert",
        },
    )


class InstructionParser:
    """
    工业指令解析入口：优先 BERT intent + slot；无 checkpoint 时回退规则解析。
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._rule = RuleBasedParser()
        self._intent: BertIntentClassifier | None = None
        self._slot: BertSlotTagger | None = None
        if self.config.intent_ckpt and self.config.intent_ckpt.exists():
            self._intent = BertIntentClassifier.from_inference_config(self.config)
        if self.config.slot_ckpt and self.config.slot_ckpt.exists():
            self._slot = BertSlotTagger.from_inference_config(self.config)

    def parse(self, text: str) -> IndustrialCommand:
        text = text.strip()
        if self._intent is None and self._slot is None:
            if self.config.use_rule_fallback:
                return self._rule.parse(text)
            return IndustrialCommand(raw_text=text, intent=IndustrialIntent.OTHER.value, parser_backend="none")

        intent = IndustrialIntent.OTHER.value
        confidence: float | None = None
        if self._intent is not None:
            intent, confidence = self._intent.predict(text)

        slots: dict[str, str] = {}
        if self._slot is not None:
            slots = self._slot.predict_slots(text)

        base = self._rule.parse(text)
        cmd = base.model_copy(
            update={
                "intent": intent,
                "confidence": confidence,
                "parser_backend": "bert",
            },
        )
        if slots:
            cmd = _merge_slots_into_command(cmd, slots)
        elif not cmd.slots:
            for name in SLOT_NAMES:
                val = getattr(cmd, "obj" if name == "object" else name, None)
                if val:
                    cmd.slots[name if name != "object" else "object"] = val
        return cmd
