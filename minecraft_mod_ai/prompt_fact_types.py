from __future__ import annotations

"""Atomic prompt fact types for fine-grained prompt decomposition.

Each prompt fact represents exactly one verified statement, relation, or constraint
extracted from a single clause of the user's prompt.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FactType(str, Enum):
    # Existence
    BLOCK_EXISTS = "BLOCK_EXISTS"
    ITEM_EXISTS = "ITEM_EXISTS"
    ENTITY_EXISTS = "ENTITY_EXISTS"

    # Block behaviors & relations
    BLOCK_DROP = "BLOCK_DROP"
    BLOCK_HARDNESS = "BLOCK_HARDNESS"
    BLOCK_RESISTANCE = "BLOCK_RESISTANCE"
    BLOCK_SOUND = "BLOCK_SOUND"
    BLOCK_LIGHT = "BLOCK_LIGHT"

    # Item properties & settings
    ITEM_STACK_LIMIT = "ITEM_STACK_LIMIT"
    ITEM_DURABILITY = "ITEM_DURABILITY"
    ITEM_FIREPROOF = "ITEM_FIREPROOF"
    ITEM_FOOD_NUTRITION = "ITEM_FOOD_NUTRITION"
    ITEM_FOOD_SATURATION = "ITEM_FOOD_SATURATION"
    ITEM_ATTACK_DAMAGE = "ITEM_ATTACK_DAMAGE"
    ITEM_ATTACK_SPEED = "ITEM_ATTACK_SPEED"

    # Visual & Aesthetic
    VISUAL_COLOR = "VISUAL_COLOR"
    VISUAL_FORM = "VISUAL_FORM"
    VISUAL_TEXTURE_REFERENCE = "VISUAL_TEXTURE_REFERENCE"

    # Recipes & Progression
    CRAFTING_RECIPE = "CRAFTING_RECIPE"
    SMELTING_RECIPE = "SMELTING_RECIPE"

    # External Reference
    EXTERNAL_REFERENCE = "EXTERNAL_REFERENCE"


@dataclass(frozen=True)
class PromptFact:
    fact_id: str
    fact_type: FactType
    subject: str
    predicate: str = ""
    object: str = ""
    value: Any = None
    unit: str = ""
    source_clause: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "value": self.value,
            "unit": self.unit,
            "source_clause": self.source_clause,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptFact:
        return cls(
            fact_id=str(data["fact_id"]),
            fact_type=FactType(data["fact_type"]),
            subject=str(data["subject"]),
            predicate=str(data.get("predicate", "")),
            object=str(data.get("object", "")),
            value=data.get("value"),
            unit=str(data.get("unit", "")),
            source_clause=str(data.get("source_clause", "")),
        )
