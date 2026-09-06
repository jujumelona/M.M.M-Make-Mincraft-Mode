"""Host-numbered and multi-page text blocks avoid asking a small model to escape nested JSON."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_HEADER = re.compile(r"^### Hole ([1-9][0-9]*)\s*$", re.MULTILINE)
_FIELD = re.compile(
    r"^(Decision|Steps|Bindings|References|Verification|Uncertainties):[ \t]*(.*)$",
    re.IGNORECASE,
)
_LIST_FIELDS = {"steps", "bindings", "references", "uncertainties"}
_KEYS = {
    "decision": "implementation_decision",
    "steps": "local_steps",
    "bindings": "code_bindings",
    "references": "reference_uses",
    "verification": "verification_intent",
    "uncertainties": "uncertainties",
}

_PAGE_PATTERN = re.compile(
    r"^\s*(?:BEGIN\s+PAGE\s+([A-Za-z0-9_.:-]+)|###\s*Page\s+([A-Za-z0-9_.:-]+))\s*\n?(.*?)(?:^\s*END\s+PAGE(?:\s+(?:\1|\2))?\s*$|(?=^\s*(?:BEGIN\s+PAGE|###\s*Page))|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

_DELIMITED_HOLE_PATTERN = re.compile(
    r"^\s*(?:BEGIN\s+([A-Za-z0-9_.:-]+)|###\s*Hole\s+([A-Za-z0-9_.:-]+))\s*\n?(.*?)(?:^\s*END(?:\s+(?:\1|\2))?\s*$|(?=^\s*(?:BEGIN\s+[A-Za-z0-9_.:-]+|###\s*Hole|END\s+PAGE))|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def _parse_hole_fields(raw_block: str) -> dict[str, Any] | None:
    fields: dict[str, list[str]] = {}
    current = ""
    invalid = False
    for line in raw_block.strip().splitlines():
        match = _FIELD.fullmatch(line.strip())
        if match:
            current_raw, value = match.groups()
            current = current_raw.casefold()
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
        return None
    for list_key in ("code_bindings", "reference_uses", "uncertainties"):
        if list_key not in values:
            values[list_key] = []
    return values


def parse_hole_text(
    raw: str, holes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Accept complete blocks only; omitted or malformed blocks remain repairable."""
    if raw.strip().startswith("{"):
        import json

        try:
            packet = json.loads(raw)
            if isinstance(packet, Mapping):
                raw_fills = None
                if isinstance(packet.get("modules"), list) and packet["modules"]:
                    first_mod = packet["modules"][0]
                    if isinstance(first_mod, Mapping):
                        cfg = first_mod.get("config")
                        if isinstance(cfg, Mapping):
                            raw_fills = cfg.get("hole_fills")
                elif isinstance(packet.get("hole_fills"), list):
                    raw_fills = packet.get("hole_fills")
                if isinstance(raw_fills, list):
                    holes_by_id = {
                        str(h.get("hole_id") or ""): h for h in holes if h.get("hole_id")
                    }
                    validated_fills = []
                    for fill in raw_fills:
                        if (
                            isinstance(fill, Mapping)
                            and str(fill.get("hole_id") or "") in holes_by_id
                        ):
                            validated_fills.append(dict(fill))
                    if validated_fills:
                        return validated_fills
        except Exception:
            pass

    headers = list(_HEADER.finditer(raw))
    if not headers:
        # Check if delimited hole pattern is present
        delimited_matches = list(_DELIMITED_HOLE_PATTERN.finditer(raw))
        if delimited_matches:
            holes_by_id = {
                str(h.get("hole_id") or ""): h
                for h in holes
                if h.get("hole_id")
            }
            fills: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for m in delimited_matches:
                token = m.group(1) or m.group(2)
                target_id = None
                if token in holes_by_id:
                    target_id = token
                elif token.isdigit():
                    ord_val = int(token)
                    if 1 <= ord_val <= len(holes):
                        target_id = str(holes[ord_val - 1].get("hole_id") or "")
                if target_id and target_id in holes_by_id and target_id not in seen_ids:
                    parsed = _parse_hole_fields(m.group(3))
                    if parsed is not None:
                        seen_ids.add(target_id)
                        fills.append({"hole_id": target_id, **parsed})
            return fills
        return []

    seen: set[int] = set()
    fills = []
    for index, header in enumerate(headers):
        ordinal = int(header.group(1))
        if ordinal in seen or ordinal > len(holes):
            raise ValueError(f"Unknown or repeated hole ordinal: {ordinal}")
        seen.add(ordinal)
        end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
        parsed = _parse_hole_fields(raw[header.end() : end])
        if parsed is not None:
            fills.append({"hole_id": holes[ordinal - 1]["hole_id"], **parsed})
    return fills


def parse_multi_page_hole_text(
    raw: str,
    pages_holes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Parse multi-page bounded text blocks; enforce host-owned page & hole IDs."""
    if len(pages_holes) == 1 and not re.search(r"BEGIN\s+PAGE|###\s*Page", raw, re.IGNORECASE):
        single_page_id = next(iter(pages_holes))
        try:
            return {single_page_id: parse_hole_text(raw, pages_holes[single_page_id])}
        except Exception:
            return {single_page_id: []}

    if raw.strip().startswith("{"):
        import json

        try:
            packet = json.loads(raw)
            if isinstance(packet, Mapping):
                modules = packet.get("modules")
                if isinstance(modules, list):
                    json_by_page: dict[str, list[dict[str, Any]]] = {
                        page_id: [] for page_id in pages_holes
                    }
                    for mod in modules:
                        if not isinstance(mod, Mapping):
                            continue
                        mod_id = str(mod.get("module_id") or "")
                        matching_pages = [
                            pid
                            for pid in pages_holes
                            if pid == mod_id or pid.startswith(f"{mod_id}_p")
                        ]
                        cfg = mod.get("config")
                        raw_fills = (
                            cfg.get("hole_fills") if isinstance(cfg, Mapping) else None
                        )
                        if isinstance(raw_fills, list):
                            for pid in matching_pages:
                                expected = {
                                    str(h.get("hole_id") or "")
                                    for h in pages_holes[pid]
                                }
                                page_fills = [
                                    dict(f)
                                    for f in raw_fills
                                    if isinstance(f, Mapping)
                                    and str(f.get("hole_id") or "") in expected
                                ]
                                json_by_page[pid].extend(page_fills)
                    if any(json_by_page.values()):
                        return json_by_page
        except Exception:
            pass

    result: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in pages_holes}
    for match in _PAGE_PATTERN.finditer(raw):
        page_id = match.group(1) or match.group(2)
        if page_id not in pages_holes:
            continue
        page_body = match.group(3)
        expected_holes = list(pages_holes[page_id])
        holes_by_id = {
            str(h.get("hole_id") or ""): h
            for h in expected_holes
            if h.get("hole_id")
        }

        seen_holes: set[str] = set()
        for h_match in _DELIMITED_HOLE_PATTERN.finditer(page_body):
            hole_token = h_match.group(1) or h_match.group(2)
            hole_body = h_match.group(3)

            target_hole_id = None
            if hole_token in holes_by_id:
                target_hole_id = hole_token
            elif hole_token.isdigit():
                ordinal = int(hole_token)
                if 1 <= ordinal <= len(expected_holes):
                    target_hole_id = str(expected_holes[ordinal - 1].get("hole_id") or "")

            if not target_hole_id or target_hole_id not in holes_by_id or target_hole_id in seen_holes:
                continue

            parsed = _parse_hole_fields(hole_body)
            if parsed is not None:
                seen_holes.add(target_hole_id)
                result[page_id].append({"hole_id": target_hole_id, **parsed})

    return result


__all__ = ["parse_hole_text", "parse_multi_page_hole_text"]
