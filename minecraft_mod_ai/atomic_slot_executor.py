from __future__ import annotations

"""Single-slot AI filler with fail-closed evidence and bounded context."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import assert_strict_atomicity_bounds


MAX_SLOT_CONTEXT_CHARS = 4096


class SlotFillError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlotDefinition:
    slot_id: str
    schema: dict[str, Any]
    description: str = ""
    default: Any = None

    def validate_schema(self) -> None:
        if not isinstance(self.schema, Mapping):
            raise SlotFillError(
                f"SLOT_SCHEMA: Slot {self.slot_id} schema must be a mapping"
            )
        if self.default is not None:
            raise SlotFillError(
                f"SLOT_DEFAULT_FORBIDDEN: Slot {self.slot_id} cannot invent a fallback value"
            )
        try:
            Draft202012Validator.check_schema(self.schema)
            assert_strict_atomicity_bounds(
                self.schema, surface=f"atomic slot {self.slot_id!r}"
            )
        except Exception as exc:
            raise SlotFillError(
                f"SLOT_ATOMICITY_VIOLATION: Slot {self.slot_id} is not a bounded atomic schema: {exc}"
            ) from exc


def _bounded_context(context: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(context),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception as exc:
        raise SlotFillError(f"SLOT_CONTEXT_INVALID: context is not serializable: {exc}") from exc
    if len(encoded) > MAX_SLOT_CONTEXT_CHARS:
        raise SlotFillError(
            f"SLOT_CONTEXT_TOO_LARGE: {len(encoded)} characters exceeds "
            f"{MAX_SLOT_CONTEXT_CHARS}; pass only the evidence slice needed for this slot"
        )
    return encoded


def fill_one_slot(
    router: Any,
    slot: SlotDefinition | Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    max_retries: int = 2,
    role: str = "planner",
) -> Any:
    """Resolve exactly one bounded value; missing evidence never becomes a default."""
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
    if router is None:
        raise SlotFillError(
            f"SLOT_NO_ROUTER: No router supplied to resolve {slot_def.slot_id}; "
            "leave the slot unresolved instead of inventing a value"
        )

    schema = slot_def.schema
    validator = Draft202012Validator(schema)
    context_text = _bounded_context(context)
    messages = [
        {
            "role": "system",
            "content": (
                f"Resolve exactly one bounded value for '{slot_def.slot_id}'.\n"
                f"Meaning: {slot_def.description or slot_def.slot_id}.\n"
                "Use only the supplied evidence context. Do not guess a missing value."
            ),
        },
        {
            "role": "user",
            "content": f"Evidence context:\n{context_text}",
        },
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            value = generate_fixed_template_value(
                router,
                role,
                messages,
                response_schema=schema,
                tool_name=(
                    f"resolve_{slot_def.slot_id}"
                    .replace(".", "_")
                    .replace("-", "_")[:64]
                ),
                description=f"Resolve one atomic slot: {slot_def.slot_id}",
                enable_tools=False,
            )
            validator.validate(value)
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
                        "content": (
                            f"The prior value violated the declared slot schema: "
                            f"{type(exc).__name__}. Return only a valid value supported "
                            "by the same evidence."
                        ),
                    }
                )

    raise SlotFillError(
        f"SLOT_RETRY_EXHAUSTED: Failed to resolve {slot_def.slot_id} after "
        f"{max_retries + 1} attempts: {last_error}"
    )
