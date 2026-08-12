from __future__ import annotations

import json
from dataclasses import dataclass
from functools import wraps
from typing import Any, Sequence


_OUTLINE_FIELDS = frozenset({"production_batches", "complete", "next_cursor"})
_IGNORABLE_RESPONSE_METADATA = frozenset({"notes", "generation_metadata"})


@dataclass(frozen=True)
class _DecodedContainer:
    start: int
    end: int
    value: Any


def _outermost_complete_json_containers(text: str) -> list[_DecodedContainer]:
    """Return complete outermost JSON object/array values embedded in ``text``.

    This scanner never repairs JSON. It only exposes containers that Python's strict
    JSON decoder can already parse completely. Nested containers are removed so a
    model may intentionally emit several consecutive top-level pages without nested
    objects being mistaken for additional pages.
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


def _extract_unique_contract_object(
    containers: Sequence[_DecodedContainer],
    expected_contracts: Sequence[frozenset[str]],
) -> dict[str, Any]:
    """Select one contract-shaped object while ignoring unrelated scratch JSON.

    Some local reasoning models emit a small top-level scratch object before the final
    structured answer. It is ignored only when exactly one outer object contains one
    host contract and any extra fields are inert response metadata. Multiple matching
    objects remain ambiguous and fail closed.
    """

    matches: list[dict[str, Any]] = []
    for container in containers:
        value = container.value
        if not isinstance(value, dict):
            continue
        fields = frozenset(str(key) for key in value)
        for expected in expected_contracts:
            if expected <= fields and (fields - expected) <= _IGNORABLE_RESPONSE_METADATA:
                matches.append({key: value[key] for key in expected})
                break
    if len(matches) != 1:
        raise ValueError(
            "response must contain exactly one complete contract-shaped JSON object; "
            f"found {len(matches)} matching objects among {len(containers)} outer containers"
        )
    return matches[0]


def _is_exact_outline_container(container: _DecodedContainer) -> bool:
    value = container.value
    return isinstance(value, dict) and frozenset(str(key) for key in value) == _OUTLINE_FIELDS


def _valid_outline_page(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"production-outline page {index} must be a JSON object")
    page = dict(value)
    fields = frozenset(str(key) for key in page)
    if fields != _OUTLINE_FIELDS:
        raise ValueError(
            f"production-outline page {index} fields are invalid: received={sorted(fields)}"
        )
    if not isinstance(page["production_batches"], list):
        raise ValueError(
            f"production-outline page {index} production_batches must be a list"
        )
    if type(page["complete"]) is not bool:
        raise ValueError(f"production-outline page {index} complete must be boolean")
    if not isinstance(page["next_cursor"], str):
        raise ValueError(f"production-outline page {index} next_cursor must be a string")
    return page


def _aggregate_outline_pages(pages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not pages:
        raise ValueError("response contains no complete production-outline JSON page")
    combined_batches: list[Any] = []
    for page in pages:
        combined_batches.extend(page["production_batches"])

    terminal = pages[-1]
    cursor = terminal["next_cursor"]
    complete = bool(terminal["complete"]) and not cursor
    if not complete and not cursor:
        raise ValueError(
            "terminal production-outline page is incomplete but has no next_cursor"
        )
    return {
        "production_batches": combined_batches,
        "complete": complete,
        "next_cursor": "" if complete else cursor,
    }


def _extract_outline_sequence(text: str) -> dict[str, Any]:
    """Consume every complete consecutive production-outline JSON page in a response."""

    containers = _outermost_complete_json_containers(text)
    if not containers:
        raise ValueError("response contains no complete production-outline JSON page")
    pages = [
        _valid_outline_page(container.value, index)
        for index, container in enumerate(containers, start=1)
    ]
    return _aggregate_outline_pages(pages)


def _extract_outline_prefix(text: str) -> tuple[dict[str, Any] | None, str]:
    """Return the valid outline prefix before the first bad page."""

    containers = _outermost_complete_json_containers(text)
    if not containers:
        return None, "no complete outermost JSON container"

    pages: list[dict[str, Any]] = []
    for index, container in enumerate(containers, start=1):
        try:
            page = _valid_outline_page(container.value, index)
        except ValueError as exc:
            if not pages:
                return None, str(exc)
            batches: list[Any] = []
            for accepted in pages:
                batches.extend(accepted["production_batches"])
            cursor = pages[-1]["next_cursor"]
            return (
                {
                    "production_batches": batches,
                    "complete": False,
                    "next_cursor": cursor,
                },
                str(exc),
            )
        pages.append(page)

    try:
        return _aggregate_outline_pages(pages), ""
    except ValueError as exc:
        batches: list[Any] = []
        for accepted in pages:
            batches.extend(accepted["production_batches"])
        cursor = pages[-1]["next_cursor"] if pages else ""
        return (
            {
                "production_batches": batches,
                "complete": False,
                "next_cursor": cursor,
            }
            if pages
            else None,
            str(exc),
        )


def install(runtime_module: Any) -> None:
    """Install strict JSON parsing plus scalable multi-page outline consumption."""

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

        containers = _outermost_complete_json_containers(text)
        outline_allowed = _OUTLINE_FIELDS in tuple(expected_contracts)
        outline_count = sum(1 for container in containers if _is_exact_outline_container(container))
        all_outline = bool(containers) and outline_count == len(containers)
        try:
            if outline_allowed and all_outline:
                value = _extract_outline_sequence(text)
            elif outline_allowed and outline_count:
                # Once a response contains a production-outline page, every other
                # outer JSON container must be another outline page. Silently
                # discarding an unrelated object would make continuation ambiguous.
                raise ValueError(
                    "production-outline response mixed valid outline pages with unrelated "
                    f"outer JSON containers ({outline_count}/{len(containers)} outline pages)"
                )
            elif len(containers) == 1:
                value = _extract_one_complete_object(text)
            else:
                # Non-outline structured calls may still tolerate one scratch object
                # from a local reasoning model when exactly one final contract object
                # remains unambiguous.
                value = _extract_unique_contract_object(containers, expected_contracts)
        except (ValueError, json.JSONDecodeError) as exc:
            expectation = (
                "valid sequential production-outline JSON pages"
                if outline_allowed and outline_count
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


__all__ = [
    "install",
    "_extract_one_complete_object",
    "_extract_outline_sequence",
    "_extract_outline_prefix",
    "_outermost_complete_json_containers",
]
