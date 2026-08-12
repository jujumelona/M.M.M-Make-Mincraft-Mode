from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Any


def _json_native(value: Any) -> Any:
    """Convert dataclass output into values representable by JSON without coercion."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_native(item) for item in value]
    if isinstance(value, list):
        return [_json_native(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    return value


def install(spec_module: Any) -> None:
    """Guarantee ``Proposal.to_dict`` produces JSON-native containers."""

    cls = spec_module.Proposal
    current = cls.to_dict
    if getattr(current, "_mmm_json_native_serialization", False):
        return

    @wraps(current)
    def to_dict(self: Any) -> dict[str, Any]:
        value = _json_native(current(self))
        if not isinstance(value, dict):
            raise spec_module.SpecValidationError(
                "Proposal serialization must produce a JSON object."
            )
        return value

    to_dict._mmm_json_native_serialization = True  # type: ignore[attr-defined]
    to_dict.__wrapped__ = current  # type: ignore[attr-defined]
    cls.to_dict = to_dict


__all__ = ["install"]
