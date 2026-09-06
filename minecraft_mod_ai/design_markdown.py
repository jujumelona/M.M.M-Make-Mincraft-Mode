from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .model_meta_output_contract import assert_design_field_clean
from .spec import SpecValidationError

_LIST_FIELDS = frozenset({"core_loop", "progression", "acceptance_tests"})

_MAP_FIELDS = frozenset({"combat", "mod_context", "art_direction"})

_NONE_VALUES = frozenset({"none", "n/a", "없음"})


def _section_field_body(raw: Any, field: str, fields: Sequence[str]) -> str:
    expected = {_normalize_heading(value): value for value in fields}
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in str(raw or "").splitlines():
        match = re.match(r"^\s*(?:#\s+)?##\s+(.+?)\s*$", line)
        if match:
            current = expected.get(_normalize_heading(match.group(1)))
            if current is not None:
                bodies.setdefault(current, [])
            continue
        if current is not None:
            bodies[current].append(line)
    if field not in bodies:
        raise SpecValidationError(
            f"Planner prose omitted required Markdown heading: {field}"
        )
    return "\n".join(bodies[field]).strip()


def _strip_accidental_field_wrapper(raw: Any, field: str) -> str:
    body = str(raw or "").strip()
    if body.startswith("```") and body.endswith("```"):
        lines = body.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    lines = body.splitlines()
    if lines:
        heading = re.match(r"^\s*##\s+(.+?)\s*$", lines[0])
        if heading and _normalize_heading(heading.group(1)) == _normalize_heading(
            field
        ):
            body = "\n".join(lines[1:]).strip()
    return body


def _parse_field_output(raw: Any, field: str) -> Any:
    body = _strip_accidental_field_wrapper(raw, field)
    if field in {"title", "pitch"}:
        value = _plain_text(body)
        if not value:
            raise SpecValidationError(f"Planner left {field} empty")
        try:
            assert_design_field_clean(field, value)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc
        return value
    if field in _LIST_FIELDS:
        values = _markdown_list(body)
        if not values:
            raise SpecValidationError(f"Planner left {field} empty")
        if field == "core_loop":
            try:
                assert_design_field_clean(field, values)
            except ValueError as exc:
                raise SpecValidationError(str(exc)) from exc
        return values
    if field in _MAP_FIELDS:
        return _markdown_map(body)
    if field == "modules":
        return _module_rows(body)
    if field == "assets":
        return _asset_rows(body)
    raise SpecValidationError(f"Unsupported host design field: {field}")


def _normalize_heading(value: str) -> str:
    value = value.strip().strip("`").casefold()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _parse_markdown_section(raw: str, fields: Sequence[str]) -> dict[str, Any]:
    expected = {_normalize_heading(field): field for field in fields}
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in str(raw or "").splitlines():
        match = re.match(r"^\s*(?:#\s+)?##\s+(.+?)\s*$", line)
        if match:
            current = expected.get(_normalize_heading(match.group(1)))
            if current is not None:
                bodies.setdefault(current, [])
            continue
        if current is not None:
            bodies[current].append(line)
    missing = [field for field in fields if field not in bodies]
    if missing:
        raise SpecValidationError(
            "Planner prose omitted required Markdown heading(s): " + ", ".join(missing)
        )
    section: dict[str, Any] = {}
    for field in fields:
        body = "\n".join(bodies[field]).strip()
        if field in {"title", "pitch"}:
            value = _plain_text(body)
            if not value:
                raise SpecValidationError(f"Planner prose left ## {field} empty")
            section[field] = value
        elif field in _LIST_FIELDS:
            values = _markdown_list(body)
            if not values:
                raise SpecValidationError(f"Planner prose left ## {field} empty")
            section[field] = values
        elif field in _MAP_FIELDS:
            section[field] = _markdown_map(body)
        elif field == "modules":
            section[field] = _module_rows(body)
        elif field == "assets":
            section[field] = _asset_rows(body)
        else:
            raise SpecValidationError(f"Unsupported host design field: {field}")
    return section


def _plain_text(body: str) -> str:
    lines = [_strip_list_marker(line) for line in body.splitlines()]
    return " ".join(line for line in lines if line).strip()


def _strip_list_marker(line: str) -> str:
    value = line.strip()
    if not value:
        return ""
    return re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", value).strip()


def _markdown_list(body: str) -> list[str]:
    values: list[str] = []
    for line in body.splitlines():
        if re.match(r"^\s*#{1,6}\s", line):
            continue
        value = _strip_list_marker(line)
        if value:
            values.append(value)
    return values


def _markdown_map(body: str) -> dict[str, list[str]]:
    normalized = (
        " ".join(
            _strip_list_marker(line)
            for line in body.splitlines()
            if _strip_list_marker(line)
        )
        .strip()
        .casefold()
    )
    if normalized in _NONE_VALUES:
        return {}
    result: dict[str, list[str]] = {}
    current = "summary"
    for line in body.splitlines():
        heading = re.match(r"^\s*###\s+(.+?)\s*$", line)
        if heading:
            current = _normalize_heading(heading.group(1)) or "summary"
            result.setdefault(current, [])
            continue
        value = _strip_list_marker(line)
        if value:
            result.setdefault(current, []).append(value)
    return {key: values for key, values in result.items() if values}


def _split_csv(value: str) -> list[str]:
    text = value.strip().strip("[](){}")
    if text.casefold() in _NONE_VALUES:
        return []
    return list(
        dict.fromkeys(
            item.strip().strip("`'\"")
            for item in re.split(r"\s*[,;，；]\s*", text)
            if item.strip().strip("`'\"")
        )
    )


def _split_obligations(value: str) -> list[str]:
    text = value.strip().strip("[]")
    if text.casefold() in _NONE_VALUES:
        return []
    separators = r"\s*(?:;|；|<br\s*/?>)\s*"
    parts = [
        item.strip().strip("`'\"")
        for item in re.split(separators, text, flags=re.IGNORECASE)
    ]
    return list(dict.fromkeys(item for item in parts if item))


def _record_key_value(value: str) -> tuple[str, str] | None:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_ -]*)\s*:\s*(.*)$", value.strip())
    if not match:
        return None
    key = _normalize_heading(match.group(1))
    return key, match.group(2).strip()


def _pipe_parts(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [part.strip() for part in text.split("|")]


def _is_markdown_table_separator(parts: Sequence[str]) -> bool:
    return bool(parts) and all(
        re.fullmatch(r":?-{3,}:?", part.replace(" ", "")) for part in parts if part
    )


def _finalize_module_record(
    record: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    plugin_id = str(record.get("plugin_id") or "").strip()
    status = str(record.get("status") or "").strip()
    reason = str(record.get("reason") or "").strip()
    raw_refs = record.get("requirement_refs")
    refs = (
        list(raw_refs)
        if isinstance(raw_refs, list)
        else _split_csv(str(raw_refs or ""))
    )
    raw_obligations = record.get("implementation_obligations")
    obligations = (
        [str(item).strip() for item in raw_obligations if str(item).strip()]
        if isinstance(raw_obligations, list)
        else _split_obligations(str(raw_obligations or ""))
    )
    obligations = list(dict.fromkeys(obligations))
    if not plugin_id or not status or not reason or not obligations:
        raise SpecValidationError(
            "Each ## modules record requires plugin_id, status, reason, requirement_refs, "
            f"and concrete implementation_obligations; malformed record: {source}"
        )
    return {
        "plugin_id": plugin_id,
        "status": status,
        "reason": reason,
        "requirement_refs": refs,
        "implementation_obligations": obligations,
    }


def _module_rows(body: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_list_key: str | None = None

    def flush() -> None:
        nonlocal current, active_list_key
        if current is not None:
            rows.append(_finalize_module_record(current, source=str(current)))
        current = None
        active_list_key = None

    for raw_line in body.splitlines():
        heading = re.match(r"^\s*###\s+(.+?)\s*$", raw_line)
        if heading:
            flush()
            heading_value = heading.group(1).strip().strip("`")
            heading_value = re.sub(
                r"^(?:module|plugin_id)\s*:\s*", "", heading_value, flags=re.IGNORECASE
            )
            if heading_value.casefold() in _NONE_VALUES:
                continue
            current = {"plugin_id": heading_value}
            continue

        value = _strip_list_marker(raw_line)
        if not value:
            continue
        if value.casefold() in _NONE_VALUES and current is None:
            continue

        parts = _pipe_parts(value)
        if "|" in value:
            normalized_header = [_normalize_heading(part) for part in parts]
            if normalized_header[:2] == [
                "plugin_id",
                "status",
            ] or _is_markdown_table_separator(parts):
                continue
            if len(parts) >= 5 and all(parts[:2]) and parts[-2] and parts[-1]:
                flush()
                rows.append(
                    _finalize_module_record(
                        {
                            "plugin_id": parts[0],
                            "status": parts[1],
                            "reason": " | ".join(parts[2:-2]).strip(),
                            "requirement_refs": _split_csv(parts[-2]),
                            "implementation_obligations": _split_obligations(parts[-1]),
                        },
                        source=value,
                    )
                )
                continue

        key_value = _record_key_value(value)
        if key_value is not None:
            key, item_value = key_value
            if key == "plugin_id":
                if current is not None and current.get("plugin_id"):
                    flush()
                current = {"plugin_id": item_value}
                continue
            if current is None:
                current = {}
            if key in {"status", "reason"}:
                current[key] = item_value
                active_list_key = None
                continue
            if key == "requirement_refs":
                current[key] = _split_csv(item_value)
                active_list_key = "requirement_refs" if not item_value else None
                continue
            if key == "implementation_obligations":
                current[key] = _split_obligations(item_value)
                active_list_key = (
                    "implementation_obligations" if not item_value else None
                )
                continue

        if current is not None and active_list_key == "implementation_obligations":
            current.setdefault("implementation_obligations", []).append(value)
            continue
        if current is not None and active_list_key == "requirement_refs":
            current.setdefault("requirement_refs", []).extend(_split_csv(value))
            continue
        if current is not None and current.get("reason"):
            current["reason"] = f"{current['reason']} {value}".strip()
            continue
        raise SpecValidationError(
            "Could not parse a ## modules record. Use labeled Markdown records or the supported legacy pipe record."
        )

    flush()
    return rows


def _finalize_asset_record(record: Mapping[str, Any], *, source: str) -> dict[str, str]:
    asset_id = str(record.get("id") or "").strip()
    kind = str(record.get("kind") or "").strip()
    brief = str(record.get("brief") or "").strip()
    if not asset_id or not kind or not brief:
        raise SpecValidationError(
            f"Each ## assets record requires id, kind, and brief; malformed record: {source}"
        )
    return {"id": asset_id, "kind": kind, "brief": brief}


def _asset_rows(body: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            rows.append(_finalize_asset_record(current, source=str(current)))
        current = None

    for raw_line in body.splitlines():
        heading = re.match(r"^\s*###\s+(.+?)\s*$", raw_line)
        if heading:
            flush()
            heading_value = heading.group(1).strip().strip("`")
            heading_value = re.sub(
                r"^(?:asset|id)\s*:\s*", "", heading_value, flags=re.IGNORECASE
            )
            if heading_value.casefold() in _NONE_VALUES:
                continue
            current = {"id": heading_value}
            continue

        value = _strip_list_marker(raw_line)
        if not value:
            continue
        if value.casefold() in _NONE_VALUES and current is None:
            continue
        parts = _pipe_parts(value)
        if "|" in value:
            normalized_header = [_normalize_heading(part) for part in parts]
            if normalized_header[:2] == ["id", "kind"] or _is_markdown_table_separator(
                parts
            ):
                continue
            if len(parts) >= 3 and all(parts[:2]):
                flush()
                rows.append(
                    _finalize_asset_record(
                        {
                            "id": parts[0],
                            "kind": parts[1],
                            "brief": " | ".join(parts[2:]).strip(),
                        },
                        source=value,
                    )
                )
                continue

        key_value = _record_key_value(value)
        if key_value is not None:
            key, item_value = key_value
            if key == "id":
                if current is not None and current.get("id"):
                    flush()
                current = {"id": item_value}
                continue
            if current is None:
                current = {}
            if key in {"kind", "brief"}:
                current[key] = item_value
                continue
        if current is not None and current.get("brief"):
            current["brief"] = f"{current['brief']} {value}".strip()
            continue
        raise SpecValidationError(
            "Could not parse a ## assets record. Use labeled Markdown records or id | kind | brief."
        )

    flush()
    return rows
