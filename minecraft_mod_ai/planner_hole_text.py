"""Host-numbered text blocks avoid asking a small model to escape nested JSON."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_HEADER = re.compile(r"^### Hole ([1-9][0-9]*)\s*$", re.MULTILINE)
_FIELD = re.compile(
    r"^(Decision|Steps|Bindings|References|Verification|Uncertainties):[ \t]*(.*)$"
)
_LIST_FIELDS = {"Steps", "Bindings", "References", "Uncertainties"}
_KEYS = {
    "Decision": "implementation_decision",
    "Steps": "local_steps",
    "Bindings": "code_bindings",
    "References": "reference_uses",
    "Verification": "verification_intent",
    "Uncertainties": "uncertainties",
}


def parse_hole_text(
    raw: str, holes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Accept complete blocks only; omitted or malformed blocks remain repairable."""
    headers = list(_HEADER.finditer(raw))
    seen: set[int] = set()
    fills: list[dict[str, Any]] = []
    for index, header in enumerate(headers):
        ordinal = int(header.group(1))
        if ordinal in seen or ordinal > len(holes):
            raise ValueError(f"Unknown or repeated hole ordinal: {ordinal}")
        seen.add(ordinal)
        end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
        fields: dict[str, list[str]] = {}
        current = ""
        invalid = False
        for line in raw[header.end() : end].strip().splitlines():
            match = _FIELD.fullmatch(line.strip())
            if match:
                current, value = match.groups()
                if current in fields:
                    invalid = True
                fields[current] = [value] if value else []
            elif line.strip() and current:
                fields[current].append(line.strip())
            elif line.strip():
                invalid = True
        values: dict[str, Any] = {}
        for field, lines in fields.items():
            if field in _LIST_FIELDS:
                values[_KEYS[field]] = [
                    item
                    for line in lines
                    if (item := re.sub(r"^(?:[-*]|[0-9]+[.)])\s+", "", line).strip())
                    and item.casefold() not in {"none", "n/a", "[]"}
                ]
            else:
                values[_KEYS[field]] = " ".join(lines).strip()
        if invalid or not all(
            values.get(key)
            for key in (
                "implementation_decision",
                "local_steps",
                "verification_intent",
            )
        ):
            continue
        fills.append({"hole_id": holes[ordinal - 1]["hole_id"], **values})
    return fills
