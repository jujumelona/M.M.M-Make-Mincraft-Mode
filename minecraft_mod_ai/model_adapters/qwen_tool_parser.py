"""Strict Qwen native tool-markup parser with schema-guided argument recovery.

Qwen family models sometimes wrap a function's argument object in a synthetic
parameter (for example ``apply``/``arguments``), use conventional aliases such as
``file`` for a schema's ``path``, emit the canonical permission identity instead of a
narrow model-facing alias, or format a string enum harmlessly differently. Recovery is
bounded by the currently exposed schema: unknown tools and keys remain errors and
``additionalProperties`` stays authoritative.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping, Sequence

from ..model_tool_aliases import resolve_exposed_model_tool
from .base import ToolCall

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_FUNCTION_OPEN = "<function="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_OPEN = "<parameter="
_PARAMETER_CLOSE = "</parameter>"
_STRUCTURAL_MARKERS = (
    _TOOL_CALL_OPEN,
    _TOOL_CALL_CLOSE,
    _FUNCTION_OPEN,
    _FUNCTION_CLOSE,
    _PARAMETER_OPEN,
    _PARAMETER_CLOSE,
)
_ARGUMENT_CONTAINER_KEYS = frozenset(
    {"apply", "arguments", "args", "parameters", "params", "input"}
)
_HOST_OWNED_ARGUMENT_KEYS = frozenset({"workspace_root"})
_APPLY_SOURCE_EDIT_ALIASES = {
    "file": "path",
    "file_path": "path",
    "target": "path",
    "target_path": "path",
    "target_file": "path",
    "action": "operation",
    "apply": "operation",
    "op": "operation",
    "mode": "operation",
    "new_text": "new",
    "new_content": "new",
    "replacement": "new",
    "old_text": "old",
    "code": "content",
    "source": "content",
    "body": "content",
}
_MAX_CONTAINER_DEPTH = 3


def parse_qwen_tool_markup(
    text: str,
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[str, tuple[ToolCall, ...]]:
    """Parse native Qwen function tags without weakening the exposed tool schema."""
    if not text:
        return "", ()
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        wrapped_at = text.find(_TOOL_CALL_OPEN, cursor)
        direct_at = text.find(_FUNCTION_OPEN, cursor)
        starts = [value for value in (wrapped_at, direct_at) if value >= 0]
        if not starts:
            break
        start = min(starts)
        wrapped = wrapped_at == start
        function_at = start + len(_TOOL_CALL_OPEN) if wrapped else start
        function_at = _skip_space(text, function_at)
        if not text.startswith(_FUNCTION_OPEN, function_at):
            if wrapped:
                raise RuntimeError("Qwen tool_call block does not begin with a function")
            cursor = start + 1
            continue
        call, end = _parse_qwen_function(text, function_at, schemas, call_index=len(calls))
        if wrapped:
            close_at = _skip_space(text, end)
            if not text.startswith(_TOOL_CALL_CLOSE, close_at):
                raise RuntimeError("Qwen tool_call block is missing </tool_call>")
            end = close_at + len(_TOOL_CALL_CLOSE)
        calls.append(call)
        spans.append((start, end))
        cursor = end
    for marker in _STRUCTURAL_MARKERS:
        pos = text.find(marker)
        if pos >= 0 and not any(begin <= pos < end for begin, end in spans):
            raise RuntimeError(f"unparsed Qwen tool markup begins at {marker!r}")
    if not spans:
        return text, ()
    visible: list[str] = []
    previous = 0
    for begin, end in spans:
        visible.append(text[previous:begin])
        previous = end
    visible.append(text[previous:])
    return "".join(visible), tuple(calls)


def _parse_qwen_function(
    text: str,
    start: int,
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    call_index: int,
) -> tuple[ToolCall, int]:
    name_start = start + len(_FUNCTION_OPEN)
    name_end = text.find(">", name_start)
    if name_end < 0:
        raise RuntimeError("Qwen function tag is missing '>'")
    emitted_name = text[name_start:name_end].strip()
    if not emitted_name:
        raise RuntimeError("Qwen function tag has an empty tool name")
    name = resolve_exposed_model_tool(emitted_name, schemas.keys())
    if name is None:
        name = emitted_name
        schema = {}
    else:
        schema = schemas[name]
    properties_value = schema.get("properties", {})
    properties = properties_value if isinstance(properties_value, Mapping) else {}
    required_value = schema.get("required", ())
    required: set[str] = set()
    if isinstance(required_value, Sequence) and not isinstance(required_value, (str, bytes)):
        required = {str(value) for value in required_value}
    additional = schema.get("additionalProperties", True)
    arguments: dict[str, Any] = {}
    argument_sources: dict[str, str] = {}
    pos = name_end + 1
    while True:
        pos = _skip_space(text, pos)
        if text.startswith(_FUNCTION_CLOSE, pos):
            end = pos + len(_FUNCTION_CLOSE)
            break
        if not text.startswith(_PARAMETER_OPEN, pos):
            snippet = " ".join(text[pos : pos + 120].split())
            raise RuntimeError(
                f"Qwen tool {name!r} emitted invalid parameter structure near {snippet!r}"
            )
        key_start = pos + len(_PARAMETER_OPEN)
        key_end = text.find(">", key_start)
        if key_end < 0:
            raise RuntimeError(f"Qwen tool {name!r} parameter tag is missing '>'")
        emitted_key = text[key_start:key_end].strip()
        if not emitted_key:
            raise RuntimeError(f"Qwen tool {name!r} emitted an empty parameter name")
        value_start = key_end + 1
        close_at = _find_parameter_close(text, value_start)
        if close_at < 0:
            raise RuntimeError(
                f"Qwen tool {name!r} parameter {emitted_key!r} is missing a structural "
                "</parameter> terminator"
            )
        raw = _unwrap_parameter_text(text[value_start:close_at])
        if emitted_key not in properties and emitted_key in _ARGUMENT_CONTAINER_KEYS:
            container = _decode_argument_container(raw)
            if container is not None:
                _merge_argument_container(
                    name,
                    emitted_key,
                    container,
                    properties,
                    additional,
                    arguments,
                    argument_sources,
                    depth=1,
                )
                pos = close_at + len(_PARAMETER_CLOSE)
                continue
        if _is_host_owned_argument(emitted_key, properties):
            pos = close_at + len(_PARAMETER_CLOSE)
            continue
        key = _canonical_key(name, emitted_key, properties)
        if key not in properties and additional is False:
            raise _unknown_parameter_error(name, emitted_key, properties, required)
        value_schema = properties.get(key, {})
        if not isinstance(value_schema, Mapping):
            value_schema = {}
        value = _decode_parameter_value(name, key, raw, value_schema)
        _insert_argument(name, key, value, emitted_key, arguments, argument_sources)
        pos = close_at + len(_PARAMETER_CLOSE)
    missing = sorted(required - arguments.keys())
    if missing:
        if "minecraft_version" in missing:
            env_ver = os.environ.get("MMM_MINECRAFT_VERSION", "").strip()
            if env_ver:
                arguments["minecraft_version"] = env_ver
            missing.remove("minecraft_version")
        if missing:
            allowed = ", ".join(sorted(str(key) for key in properties)) or "<none>"
            raise RuntimeError(
                f"Qwen tool {name!r} omitted required parameters: {', '.join(missing)}; "
                f"allowed parameters: {allowed}"
            )
    raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{call_index}\0{name}\0{raw_arguments}".encode("utf-8")
    ).hexdigest()[:16]
    return ToolCall(
        id=f"call_{digest}",
        name=name,
        arguments=arguments,
        raw_arguments=raw_arguments,
    ), end


def _is_host_owned_argument(emitted_key: str, properties: Mapping[str, Any]) -> bool:
    """Ignore host-bound execution metadata unless a tool explicitly declares it."""
    return emitted_key in _HOST_OWNED_ARGUMENT_KEYS and emitted_key not in properties


def _canonical_key(tool_name: str, emitted_key: str, properties: Mapping[str, Any]) -> str:
    if emitted_key in properties:
        return emitted_key
    if tool_name == "apply_source_edit":
        canonical = _APPLY_SOURCE_EDIT_ALIASES.get(emitted_key)
        if canonical and canonical in properties and emitted_key not in properties:
            return canonical
    return emitted_key


def _decode_argument_container(raw: str) -> Mapping[str, Any] | None:
    compact = raw.strip()
    if not compact.startswith("{"):
        return None
    try:
        value = json.loads(compact)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _merge_argument_container(
    tool_name: str,
    container_name: str,
    container: Mapping[str, Any],
    properties: Mapping[str, Any],
    additional: Any,
    arguments: dict[str, Any],
    argument_sources: dict[str, str],
    *,
    depth: int,
) -> None:
    if depth > _MAX_CONTAINER_DEPTH:
        raise RuntimeError(f"Qwen tool {tool_name!r} nested argument containers too deeply")
    for raw_key, raw_value in container.items():
        emitted_key = str(raw_key).strip()
        if not emitted_key:
            raise RuntimeError(f"Qwen tool {tool_name!r} emitted an empty nested parameter")
        if (
            emitted_key not in properties
            and emitted_key in _ARGUMENT_CONTAINER_KEYS
            and isinstance(raw_value, Mapping)
        ):
            _merge_argument_container(
                tool_name,
                emitted_key,
                raw_value,
                properties,
                additional,
                arguments,
                argument_sources,
                depth=depth + 1,
            )
            continue
        if _is_host_owned_argument(emitted_key, properties):
            continue
        key = _canonical_key(tool_name, emitted_key, properties)
        if key not in properties and additional is False:
            raise _unknown_parameter_error(tool_name, emitted_key, properties, ())
        value_schema = properties.get(key, {})
        if not isinstance(value_schema, Mapping):
            value_schema = {}
        value = _validate_decoded_value(tool_name, key, raw_value, value_schema)
        _insert_argument(
            tool_name,
            key,
            value,
            f"{container_name}.{emitted_key}",
            arguments,
            argument_sources,
        )


def _insert_argument(
    tool_name: str,
    key: str,
    value: Any,
    source: str,
    arguments: dict[str, Any],
    argument_sources: dict[str, str],
) -> None:
    if key in arguments:
        previous = argument_sources[key]
        raise RuntimeError(
            f"Qwen tool {tool_name!r} emitted conflicting sources for canonical "
            f"parameter {key!r}: {previous!r} and {source!r}"
        )
    arguments[key] = value
    argument_sources[key] = source


def _unknown_parameter_error(
    tool_name: str,
    emitted_key: str,
    properties: Mapping[str, Any],
    required: Sequence[str] | set[str],
) -> RuntimeError:
    allowed = sorted(str(key) for key in properties)
    required_names = sorted(str(key) for key in required)
    aliases: list[str] = []
    if tool_name == "apply_source_edit":
        aliases = sorted(
            alias
            for alias, canonical in _APPLY_SOURCE_EDIT_ALIASES.items()
            if canonical in properties and alias not in properties
        )
    return RuntimeError(
        f"Qwen tool {tool_name!r} emitted unknown parameter {emitted_key!r}; "
        f"allowed={allowed!r}; required={required_names!r}; accepted_aliases={aliases!r}; "
        f"object_containers={sorted(_ARGUMENT_CONTAINER_KEYS)!r}"
    )


def _find_parameter_close(text: str, start: int) -> int:
    search = start
    while True:
        candidate = text.find(_PARAMETER_CLOSE, search)
        if candidate < 0:
            return -1
        after = _skip_space(text, candidate + len(_PARAMETER_CLOSE))
        if (
            text.startswith(_PARAMETER_OPEN, after)
            or text.startswith(_FUNCTION_CLOSE, after)
            or text.startswith(_TOOL_CALL_CLOSE, after)
        ):
            return candidate
        search = candidate + len(_PARAMETER_CLOSE)


def _unwrap_parameter_text(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def _decode_parameter_value(
    tool_name: str,
    key: str,
    raw: str,
    schema: Mapping[str, Any],
) -> Any:
    expected = _schema_value_type(schema)
    compact = raw.strip()
    try:
        if expected == "string":
            value: Any = raw
        elif expected == "integer":
            if not compact or any(ch in compact.lower() for ch in (".", "e")):
                raise ValueError("not an integer")
            value = int(compact)
        elif expected == "number":
            value = float(compact)
        elif expected == "boolean":
            lowered = compact.lower()
            if lowered not in {"true", "false"}:
                raise ValueError("not a boolean")
            value = lowered == "true"
        elif expected == "null":
            if compact.lower() != "null":
                raise ValueError("not null")
            value = None
        elif expected in {"object", "array"}:
            value = json.loads(compact)
            if expected == "object" and not isinstance(value, Mapping):
                raise ValueError("not an object")
            if expected == "array" and not isinstance(value, list):
                raise ValueError("not an array")
        else:
            if compact.startswith(("{", "[", '"')) or compact in {"true", "false", "null"}:
                value = json.loads(compact)
            else:
                value = raw
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Qwen tool {tool_name!r} emitted invalid {expected or 'schema'} value "
            f"for parameter {key!r}"
        ) from exc
    return _validate_decoded_value(tool_name, key, value, schema)


def _validate_decoded_value(
    tool_name: str,
    key: str,
    value: Any,
    schema: Mapping[str, Any],
) -> Any:
    expected = _schema_value_type(schema)
    valid = True
    if expected == "string":
        valid = isinstance(value, str)
    elif expected == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "null":
        valid = value is None
    elif expected == "object":
        valid = isinstance(value, Mapping)
    elif expected == "array":
        valid = isinstance(value, list)
    if not valid:
        raise RuntimeError(
            f"Qwen tool {tool_name!r} emitted invalid {expected or 'schema'} value "
            f"for parameter {key!r}"
        )
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        if isinstance(value, str) and all(isinstance(item, str) for item in enum):
            canonical = _canonical_string_enum(value, enum)
            if canonical is None:
                raise RuntimeError(
                    f"Qwen tool {tool_name!r} emitted value outside enum for parameter {key!r}"
                )
            return canonical
        if value not in enum:
            raise RuntimeError(
                f"Qwen tool {tool_name!r} emitted value outside enum for parameter {key!r}"
            )
    return value


def _enum_key(value: str) -> str:
    compact = value.strip()
    compact = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", compact)
    compact = re.sub(r"[\s-]+", "_", compact.casefold())
    return re.sub(r"_+", "_", compact)


def _string_enum_candidates(value: str) -> tuple[str, ...]:
    candidates = [value, value.strip()]
    compact = value.strip()
    if len(compact) >= 2 and compact[0] == compact[-1] == '"':
        try:
            decoded = json.loads(compact)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, str):
            candidates.append(decoded)
    return tuple(dict.fromkeys(candidates))


def _canonical_string_enum(value: str, allowed: Sequence[str]) -> str | None:
    for candidate in _string_enum_candidates(value):
        if candidate in allowed:
            return candidate
        key = _enum_key(candidate)
        matches = tuple(item for item in allowed if _enum_key(item) == key)
        if len(matches) == 1:
            return matches[0]
    return None


def _schema_value_type(schema: Mapping[str, Any]) -> str:
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        return raw_type
    if isinstance(raw_type, list):
        non_null = [str(value) for value in raw_type if str(value) != "null"]
        if len(non_null) == 1:
            return non_null[0]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        kinds = {_json_type(value) for value in enum if value is not None}
        if len(kinds) == 1:
            return next(iter(kinds))
    for keyword in ("oneOf", "anyOf"):
        choices = schema.get(keyword)
        if isinstance(choices, list):
            kinds = {
                _schema_value_type(choice)
                for choice in choices
                if isinstance(choice, Mapping)
            }
            kinds.discard("")
            kinds.discard("null")
            if len(kinds) == 1:
                return next(iter(kinds))
    return ""


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return ""


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position
