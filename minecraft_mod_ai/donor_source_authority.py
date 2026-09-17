from __future__ import annotations

"""Low-level donor-read authority helpers shared by planning and tool-loop code."""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

_DONOR_TOOL = "read_reuse_source"
_HOST_ROLES = frozenset({"system", "tool", "developer"})
_HEX_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")
_HEX_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_DONOR_ROOT_FRAGMENT = ".minecraft_ai/reuse/donors/"


def structured_payload(content: Any) -> Any | None:
    if isinstance(content, (Mapping, list, tuple)):
        return content
    if isinstance(content, str):
        raw = content.strip()
        if not raw.startswith(("{", "[")):
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return None


def _walk_scalar_fields(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_scalar_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_scalar_fields(item)


def _donor_scalar_flags(key: str, item: Any) -> tuple[bool, bool, bool, bool]:
    if not isinstance(item, str):
        return False, False, False, False
    key_cf = key.casefold()
    text = item.replace("\\", "/").strip()
    donor_path = _DONOR_ROOT_FRAGMENT in text and ".." not in PurePosixPath(text).parts
    commit = "commit" in key_cf and bool(_HEX_COMMIT.fullmatch(text))
    digest = ("sha256" in key_cf or "hash" in key_cf) and bool(_HEX_SHA256.fullmatch(text))
    license_id = key_cf in {"license", "license_id"} and bool(text)
    return donor_path, commit, digest, license_id


def _donor_receipt_ready(payload: Any) -> bool:
    flags = [False, False, False, False]
    for key, item in _walk_scalar_fields(payload):
        current = _donor_scalar_flags(key, item)
        flags = [left or right for left, right in zip(flags, current)]
        if all(flags):
            return True
    return False


def approved_donor_authority(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Require one host-role immutable donor receipt before donor reads are exposed."""

    for message in messages:
        role = str(message.get("role") or "").strip().casefold()
        if role not in _HOST_ROLES:
            continue
        payload = structured_payload(message.get("content"))
        if payload is not None and _donor_receipt_ready(payload):
            return True
    return False


def tool_name(schema: Any) -> str:
    if isinstance(schema, Mapping):
        fn = schema.get("function")
        return str(fn.get("name") or "").strip() if isinstance(fn, Mapping) else ""
    return str(getattr(schema, "name", "") or "").strip()


def filter_donor_tool_schemas(schemas: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(schema for schema in schemas if tool_name(schema) != _DONOR_TOOL)


__all__ = [
    "approved_donor_authority",
    "filter_donor_tool_schemas",
    "structured_payload",
    "tool_name",
]
