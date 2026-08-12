from __future__ import annotations

import json
from functools import wraps
from typing import Any, Sequence


def _unwrap_transport_json_fence(text: str) -> str:
    """Remove one transport-only ```json fence and nothing else.

    The Colab llama streaming path intentionally disables the server's generic JSON
    grammar because that grammar has previously stalled CUDA decoding.  Some models
    still wrap an otherwise exact JSON object in a Markdown JSON fence.  Treat that
    fence as transport syntax only when it encloses the *entire* response.  Prose,
    multiple fenced blocks, other fence languages and truncated JSON remain invalid.
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    newline = stripped.find("\n")
    if newline < 0:
        return stripped
    opener = stripped[:newline].strip().casefold()
    if opener != "```json" or not stripped.endswith("```"):
        return stripped

    inner = stripped[newline + 1 : -3].strip()
    return inner


def install(runtime_module: Any) -> None:
    """Require one complete JSON object on structured planner pages.

    The planner already has a bounded page-local retry. Accepting embedded prose,
    ``strict=False`` JSON, or auto-closed truncated objects bypasses that repair gate
    and can synthesize pagination state the model never actually completed. Structured
    decode output therefore has one contract: one complete strict JSON object whose
    top-level field set matches one declared host contract. A single response-wide
    `````json`` transport fence is ignored because the Colab streaming path cannot
    safely rely on the server's generic JSON grammar.
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
        candidate = _unwrap_transport_json_fence(text)
        try:
            value = json.loads(candidate)
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
