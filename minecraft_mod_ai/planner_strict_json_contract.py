from __future__ import annotations

import json
from dataclasses import dataclass
from functools import wraps
from typing import Any, Sequence


_OUTLINE_FIELDS = frozenset({"production_batches", "complete", "next_cursor"})


@dataclass(frozen=True)
class _DecodedContainer:
    start: int
    end: int
    value: Any


def _outermost_complete_json_containers(text: str) -> list[_DecodedContainer]:
    """Return complete outermost JSON object/array values embedded in ``text``.

    Structured planner output can travel through several OpenAI-compatible/chat-template
    transports. Those transports may add Markdown fences, Qwen channel markers, a BOM,
    or short presentation text around otherwise valid JSON values. Transport syntax
    must not decide semantic validity.

    This scanner is deliberately not a JSON repair routine. It only accepts values
    that ``json.JSONDecoder.raw_decode`` can already parse completely. It never closes
    braces, changes strictness, fills fields, or aliases fields.
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


def _extract_outline_sequence(text: str) -> dict[str, Any]:
    """Consume one or more complete production-outline JSON pages in order.

    Production outlines are intrinsically paginated. A large model response may
    contain several consecutive outline page objects; treating that as malformed
    throws away valid planning work and creates an artificial plan-size ceiling.
    Every outermost JSON value must therefore be an exact outline page, while the
    host concatenates their batch payloads and carries the terminal page state
    forward. No page is repaired or synthesized.
    """

    containers = _outermost_complete_json_containers(text)
    if not containers:
        raise ValueError("response contains no complete production-outline JSON page")

    pages: list[dict[str, Any]] = []
    for index, container in enumerate(containers):
        if not isinstance(container.value, dict):
            raise ValueError(
                f"production-outline page {index + 1} must be a JSON object"
            )
        page = dict(container.value)
        fields = frozenset(str(key) for key in page)
        if fields != _OUTLINE_FIELDS:
            raise ValueError(
                "production-outline page fields are invalid: "
                f"received={sorted(fields)}"
            )
        if not isinstance(page["production_batches"], list):
            raise ValueError(
                f"production-outline page {index + 1} production_batches must be a list"
            )
        if type(page["complete"]) is not bool:
            raise ValueError(
                f"production-outline page {index + 1} complete must be boolean"
            )
        if not isinstance(page["next_cursor"], str):
            raise ValueError(
                f"production-outline page {index + 1} next_cursor must be a string"
            )

        final_emitted_page = index == len(containers) - 1
        if not final_emitted_page:
            if page["complete"]:
                raise ValueError(
                    "a non-final emitted production-outline page may not declare complete=true"
                )
            if not page["next_cursor"]:
                raise ValueError(
                    "a non-final emitted production-outline page requires next_cursor"
                )
        else:
            if page["complete"] and page["next_cursor"]:
                raise ValueError(
                    "a complete production-outline page may not carry next_cursor"
                )
            if not page["complete"] and not page["next_cursor"]:
                raise ValueError(
                    "an incomplete production-outline page requires next_cursor"
                )
        pages.append(page)

    combined_batches: list[Any] = []
    for page in pages:
        combined_batches.extend(page["production_batches"])

    terminal = pages[-1]
    return {
        "production_batches": combined_batches,
        "complete": terminal["complete"],
        "next_cursor": terminal["next_cursor"],
    }


def install(runtime_module: Any) -> None:
    """Install strict structured JSON parsing with scalable outline pagination."""

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

            outline_sequence = (
                len(expected_contracts) == 1
                and expected_contracts[0] == _OUTLINE_FIELDS
            )
            try:
                value = (
                    _extract_outline_sequence(text)
                    if outline_sequence
                    else _extract_one_complete_object(text)
                )
            except (ValueError, json.JSONDecodeError) as exc:
                expectation = (
                    "one or more complete sequential outline JSON pages"
                    if outline_sequence
                    else "exactly one complete strict JSON object"
                )
                raise module.SpecValidationError(
                    f"Structured planner did not return {expectation}: {exc}"
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

    # Outline prompting is independent of strict semantic validation. It gives the
    # model freedom to choose page size while preserving explicit continuation state.
    from .planner_outline_prompt_contract import install as install_outline_prompt

    install_outline_prompt(runtime_module)


__all__ = ["install"]
