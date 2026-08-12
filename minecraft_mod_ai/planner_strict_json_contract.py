from __future__ import annotations

import json
from dataclasses import dataclass
from functools import wraps
from typing import Any, Sequence


@dataclass(frozen=True)
class _DecodedContainer:
    start: int
    end: int
    value: Any


def _outermost_complete_json_containers(text: str) -> list[_DecodedContainer]:
    """Return complete outermost JSON object/array values embedded in ``text``.

    Structured planner output can travel through several OpenAI-compatible/chat-template
    transports. Those transports may add Markdown fences, Qwen channel markers, a BOM,
    or short presentation text around an otherwise exact JSON value. Transport syntax
    must not decide semantic validity.

    This scanner is deliberately *not* a JSON repair routine. It only accepts values
    that ``json.JSONDecoder.raw_decode`` can already parse completely. It never closes
    braces, changes strictness, fills fields, aliases fields, or drops a second complete
    top-level JSON container. Arrays are retained here so an array wrapping the desired
    object cannot be mistaken for a valid top-level object merely because the nested
    object itself is parseable.
    """

    decoder = json.JSONDecoder()
    decoded: list[_DecodedContainer] = []
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, relative_end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, (dict, list)):
            continue
        decoded.append(
            _DecodedContainer(
                start=start,
                end=start + int(relative_end),
                value=value,
            )
        )

    outermost: list[_DecodedContainer] = []
    for candidate in sorted(decoded, key=lambda item: (item.start, -item.end)):
        if any(
            candidate.start >= parent.start and candidate.end <= parent.end
            for parent in outermost
        ):
            continue
        outermost.append(candidate)
    return outermost


def _extract_one_complete_object(text: str) -> dict[str, Any]:
    """Extract exactly one complete outermost JSON object without repairing it."""

    containers = _outermost_complete_json_containers(text)
    if len(containers) != 1:
        raise ValueError(
            "response must contain exactly one complete outermost JSON object; "
            f"found {len(containers)} complete outermost JSON containers"
        )
    value = containers[0].value
    if not isinstance(value, dict):
        raise ValueError("structured planner top-level JSON value must be an object")
    return dict(value)


def install(runtime_module: Any) -> None:
    """Require one complete, exact-contract JSON object on structured planner pages."""

    current = runtime_module._extract_with_safe_empty_defaults
    if not getattr(current, "_mmm_strict_structured_json", False):

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
                value = _extract_one_complete_object(text)
            except (ValueError, json.JSONDecodeError) as exc:
                raise module.SpecValidationError(
                    "Structured planner did not return exactly one complete strict JSON "
                    f"object: {exc}"
                ) from exc

            fields = frozenset(str(key) for key in value)
            if fields not in tuple(expected_contracts):
                expected = [sorted(contract) for contract in expected_contracts]
                raise module.SpecValidationError(
                    "Structured planner top-level fields do not match the host contract: "
                    f"received={sorted(fields)}, expected_one_of={expected}"
                )
            return value

        extract_strict._mmm_strict_structured_json = True  # type: ignore[attr-defined]
        extract_strict.__wrapped__ = current  # type: ignore[attr-defined]
        runtime_module._extract_with_safe_empty_defaults = extract_strict

    # The strict parser stays strict. Separately constrain the tiny production-outline
    # generation prompt/output budget so the model produces one object instead of
    # spending an 8k implementation budget emitting repeated candidate JSON objects.
    from .planner_outline_prompt_contract import install as install_outline_prompt

    install_outline_prompt(runtime_module)


__all__ = ["install"]
