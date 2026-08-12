from __future__ import annotations

import json
from functools import wraps
from typing import Any, Sequence


def install(runtime_module: Any) -> None:
    """Require one complete JSON object on structured planner pages.

    The planner already has a bounded page-local retry. Accepting embedded prose,
    ``strict=False`` JSON, or auto-closed truncated objects bypasses that repair gate
    and can synthesize pagination state the model never actually completed. Structured
    decode output therefore has one contract: the entire response must parse as one
    strict JSON object whose top-level field set matches one declared host contract.
    """

    current = runtime_module._extract_with_safe_empty_defaults
    if getattr(current, "_mmm_strict_structured_json", False):
        return

    @wraps(current)
    def extract_strict(
        module: Any,
        text: str,
        *,
        expected_contracts: Sequence[frozenset[str]],
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise module.SpecValidationError("Structured planner returned empty JSON.")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise module.SpecValidationError(
                f"Structured planner did not return one complete strict JSON object: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise module.SpecValidationError(
                "Structured planner top-level JSON value must be an object."
            )
        fields = frozenset(str(key) for key in value)
        if fields not in tuple(expected_contracts):
            expected = [sorted(contract) for contract in expected_contracts]
            raise module.SpecValidationError(
                "Structured planner top-level fields do not match the host contract: "
                f"received={sorted(fields)}, expected_one_of={expected}"
            )
        return dict(value)

    extract_strict._mmm_strict_structured_json = True  # type: ignore[attr-defined]
    extract_strict.__wrapped__ = current  # type: ignore[attr-defined]
    runtime_module._extract_with_safe_empty_defaults = extract_strict


__all__ = ["install"]
