from __future__ import annotations

"""Single-slot atomic AI filler with bounded retry.

Resolves exactly ONE scalar, enum, or short identifier at a time without
asking the model to author complex records or document sections.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import (
    MAX_MODEL_FIELDS,
    MAX_MODEL_STRING_CHARS,
    MAX_SCHEMA_DEPTH,
)


class SlotFillError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlotDefinition:
    slot_id: str
    schema: dict[str, Any]
    description: str = ""
    default: Any = None

    def validate_schema(self) -> None:
        schema = self.schema
        if not isinstance(schema, Mapping):
            raise SlotFillError(f"SLOT_SCHEMA: Slot {self.slot_id} schema must be a mapping")
        # Ensure slot schema is narrow and atomic
        props = schema.get("properties", {})
        if len(props) > MAX_MODEL_FIELDS:
            raise SlotFillError(
                f"SLOT_ATOMICITY_VIOLATION: Slot {self.slot_id} has {len(props)} fields, "
                f"exceeding maximum of {MAX_MODEL_FIELDS}"
            )


def fill_one_slot(
    router: Any,
    slot: SlotDefinition | Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    max_retries: int = 2,
    role: str = "planner",
) -> Any:
    """Resolve a single slot value using a strictly bounded model call."""
    if isinstance(slot, SlotDefinition):
        slot_def = slot
    else:
        slot_def = SlotDefinition(
            slot_id=str(slot["id"] if "id" in slot else slot["slot_id"]),
            schema=dict(slot["schema"]),
            description=str(slot.get("description", "")),
            default=slot.get("default"),
        )

    slot_def.validate_schema()
    schema = slot_def.schema
    validator = Draft202012Validator(schema)

    messages = [
        {
            "role": "system",
            "content": (
                f"Provide exactly one value for property '{slot_def.slot_id}'.\n"
                f"Description: {slot_def.description or slot_def.slot_id}\n"
                "Return only the exact requested field."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\nResolve slot '{slot_def.slot_id}'.",
        },
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if router is None:
            # Deterministic fallback if default provided or test mode without model
            if slot_def.default is not None:
                return slot_def.default
            raise SlotFillError(f"SLOT_NO_ROUTER: No router provided to fill slot {slot_def.slot_id}")

        try:
            value = generate_fixed_template_value(
                router,
                role,
                messages,
                response_schema=schema,
                tool_name=f"resolve_{slot_def.slot_id}".replace(".", "_").replace("-", "_")[:64],
                description=f"Resolve single slot {slot_def.slot_id}",
                enable_tools=False,
            )
            validator.validate(value)
            # If wrapped in object with slot_id or 'value', extract
            if isinstance(value, Mapping):
                if slot_def.slot_id in value:
                    return value[slot_def.slot_id]
                if "value" in value:
                    return value["value"]
            return value
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Previous response for '{slot_def.slot_id}' was invalid: {exc}. Retry with valid value.",
                    }
                )

    if slot_def.default is not None:
        return slot_def.default
    raise SlotFillError(
        f"SLOT_RETRY_EXHAUSTED: Failed to fill slot {slot_def.slot_id} after {max_retries + 1} attempts: {last_error}"
    )
