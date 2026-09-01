from __future__ import annotations

"""Execute host-selected actions without re-asking the model for an action name.

The causal frontier owns action selection. Remote adapters keep the argument-only
fallback for source mutations. Local llama.cpp/Qwen uses one host-narrowed ``required``
tool decode for mutations so the already-validated native Qwen tool parser owns its
wire format; protocol failures still fail over to host-validated argument generation.
Other exact local actions use native ``required`` decoding after a live capability probe.
"""

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

from .source_mutation_contract import SOURCE_MUTATION_NAMES as _SOURCE_MUTATION_TOOLS
from .structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)

_MARKER = "_mmm_forced_tool_execution"
_DETERMINISTIC_READ_TOOLS = frozenset({"search_code_rag", "search_project_rag"})
_MAX_FALLBACK_QUERY_CHARS = 4096
_MAX_FALLBACK_ERROR_CHARS = 768
_MAX_ARGUMENT_ERROR_CHARS = 1600
_NATIVE_PROBE_TOOL = "mmm_required_tool_probe"
_NATIVE_PROBE_LOCK = threading.RLock()
_NATIVE_PROBE_CACHE: dict[tuple[str, str], bool] = {}
_NATIVE_PROBE_NEGATIVE_AT: dict[tuple[str, str], float] = {}
_NATIVE_PROBE_TRANSIENT_AT: dict[tuple[str, str], float] = {}
_NATIVE_PROBE_KEY_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_DEFAULT_NATIVE_NEGATIVE_TTL_SECONDS = 60.0
_DEFAULT_NATIVE_TRANSIENT_COOLDOWN_SECONDS = 5.0


def _positive_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _native_probe_key_lock(key: tuple[str, str]) -> threading.Lock:
    with _NATIVE_PROBE_LOCK:
        lock = _NATIVE_PROBE_KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _NATIVE_PROBE_KEY_LOCKS[key] = lock
        return lock


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def forced_tool_name(tool_choice: Any) -> str:
    if not isinstance(tool_choice, Mapping):
        return ""
    if str(tool_choice.get("type", "")).strip() != "function":
        return ""
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _selected_schema(request: Any, name: str) -> Mapping[str, Any]:
    from .model_adapters import ModelConfigurationError

    selected = tuple(
        schema
        for schema in tuple(getattr(request, "tools", ()) or ())
        if isinstance(schema, Mapping) and _tool_name(schema) == name
    )
    if len(selected) != 1:
        raise ModelConfigurationError(
            f"Host-selected tool {name!r} does not resolve to exactly one exposed schema."
        )
    return selected[0]


def _parameters(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    from .model_adapters import ModelConfigurationError

    function = schema.get("function")
    if not isinstance(function, Mapping):
        raise ModelConfigurationError("Host-selected tool schema is missing function metadata.")
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ModelConfigurationError("Host-selected tool schema is missing JSON parameters.")
    return parameters


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.lstrip().startswith("{"):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _redacted_text(value: Any, *, limit: int) -> str:
    from .agent_tool_runtime import _redact_text

    return " ".join(_redact_text(str(value or "")).split())[:limit]


def _structured_task(content: str) -> str:
    payload = _json_mapping(content)
    if payload is None:
        return ""
    parts: list[str] = []
    task = _redacted_text(payload.get("task", ""), limit=2048)
    if task:
        parts.append(task)
    module = payload.get("module")
    if isinstance(module, Mapping):
        for key in ("module_id", "kind"):
            value = _redacted_text(module.get(key, ""), limit=256)
            if value:
                parts.append(f"{key}={value}")
    return " ".join(parts)


def _latest_failed_mutation_context(messages: Sequence[Mapping[str, Any]]) -> str:
    selected_index = -1
    selected: Mapping[str, Any] | None = None
    tool_name = ""
    call_id = ""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role", "")).strip().casefold() != "tool":
            continue
        name = str(message.get("name", "")).strip()
        if name not in _SOURCE_MUTATION_TOOLS:
            continue
        payload = _json_mapping(message.get("content"))
        if payload is None or payload.get("ok") is True:
            continue
        selected_index = index
        selected = payload
        tool_name = name
        call_id = str(message.get("tool_call_id", "")).strip()
        break
    if selected is None:
        return ""

    operation = ""
    path = ""
    for message in reversed(messages[:selected_index]):
        if str(message.get("role", "")).strip().casefold() != "assistant":
            continue
        for raw_call in reversed(tuple(message.get("tool_calls", ()) or ())):
            if not isinstance(raw_call, Mapping):
                continue
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                continue
            if call_id and str(raw_call.get("id", "")).strip() != call_id:
                continue
            if str(function.get("name", "")).strip() != tool_name:
                continue
            arguments = _json_mapping(function.get("arguments"))
            if arguments is not None:
                operation = _redacted_text(arguments.get("operation", ""), limit=64)
                path = _redacted_text(arguments.get("path", ""), limit=512)
            break
        if operation or path:
            break

    fields = [f"tool={tool_name}"]
    if operation:
        fields.append(f"operation={operation}")
    if path:
        fields.append(f"path={path}")
    error = _redacted_text(selected.get("error", ""), limit=_MAX_FALLBACK_ERROR_CHARS)
    if error:
        fields.append(f"error={error}")
    return "failed mutation " + " ".join(fields)


def _bounded_task_query(request: Any) -> str:
    base = ""
    for raw in (getattr(request, "task", ""), getattr(request, "prompt", "")):
        value = " ".join(str(raw or "").split())
        if value:
            base = _redacted_text(value, limit=_MAX_FALLBACK_QUERY_CHARS)
            break
    messages = tuple(
        message
        for message in tuple(getattr(request, "messages", ()) or ())
        if isinstance(message, Mapping)
    )
    if not base:
        for message in reversed(messages):
            if str(message.get("role", "")).strip().casefold() != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            value = _structured_task(content) or _redacted_text(
                content,
                limit=_MAX_FALLBACK_QUERY_CHARS,
            )
            if value:
                base = value
                break
    failure = _latest_failed_mutation_context(messages)
    return " ".join(value for value in (base, failure) if value)[:_MAX_FALLBACK_QUERY_CHARS]


def _metadata_value(request: Any, key: str) -> str:
    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, Mapping):
        direct = str(metadata.get(key, "") or "").strip()
        if direct:
            return direct
        for container_name in ("platform", "platform_lock", "target"):
            nested = metadata.get(container_name)
            if isinstance(nested, Mapping):
                value = str(nested.get(key, "") or "").strip()
                if value:
                    return value
    for message in reversed(tuple(getattr(request, "messages", ()) or ())):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")).strip().casefold() != "user":
            continue
        payload = _json_mapping(message.get("content"))
        if payload is None:
            continue
        for container_name in ("platform", "platform_lock", "target"):
            nested = payload.get(container_name)
            if isinstance(nested, Mapping):
                value = str(nested.get(key, "") or "").strip()
                if value:
                    return value
    return ""


def _schema_formats_supported(schema: Mapping[str, Any], checker: Any) -> bool:
    available = getattr(checker, "checkers", {})
    if "format" in schema:
        name = schema.get("format")
        if not isinstance(name, str) or name not in available:
            return False
    for keyword in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas"):
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            for child in children.values():
                if isinstance(child, Mapping) and not _schema_formats_supported(child, checker):
                    return False
    for keyword in (
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedProperties",
    ):
        child = schema.get(keyword)
        if isinstance(child, Mapping) and not _schema_formats_supported(child, checker):
            return False
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(keyword)
        if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
            for child in children:
                if isinstance(child, Mapping) and not _schema_formats_supported(child, checker):
                    return False
    return True


def _arguments_match_schema(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> bool:
    try:
        from jsonschema import validators
        from jsonschema.exceptions import SchemaError

        schema_dict = dict(schema)
        validator_cls = validators.validator_for(schema_dict)
        validator_cls.check_schema(schema_dict)
        checker = validator_cls.FORMAT_CHECKER
        if not _schema_formats_supported(schema_dict, checker):
            return False
        validator = validator_cls(schema_dict, format_checker=checker)
        return not any(validator.iter_errors(dict(arguments)))
    except (SchemaError, TypeError, ValueError, KeyError, RecursionError):
        return False


def _deterministic_read_arguments(request: Any, name: str) -> dict[str, Any] | None:
    if name not in _DETERMINISTIC_READ_TOOLS:
        return None
    raw_parameters = _parameters(_selected_schema(request, name))
    raw_properties = raw_parameters.get("properties", {})
    properties = raw_properties if isinstance(raw_properties, Mapping) else {}
    raw_required = raw_parameters.get("required", ())
    required = (
        tuple(str(value) for value in raw_required)
        if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes))
        else ()
    )

    known: dict[str, Any] = {}
    query = _bounded_task_query(request)
    if query:
        known["query"] = query
    minecraft_version = _metadata_value(request, "minecraft_version")
    if minecraft_version:
        known["minecraft_version"] = minecraft_version
    if "query" not in properties or not query:
        return None
    if name == "search_project_rag" and (
        "minecraft_version" not in properties or not minecraft_version
    ):
        return None

    arguments: dict[str, Any] = {}
    for key, schema_value in properties.items():
        property_name = str(key)
        if property_name in known:
            if not isinstance(schema_value, Mapping):
                return None
            arguments[property_name] = known[property_name]
    if any(key not in arguments for key in required):
        return None
    if any(key not in properties for key in required):
        return None
    if not _arguments_match_schema(arguments, raw_parameters):
        return None
    return arguments


def _response_for_call(name: str, arguments: Mapping[str, Any], *, prefix: str) -> Any:
    from .model_adapters.base import GenerationResponse, ToolCall

    raw_arguments = json.dumps(
        dict(arguments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{name}\0{raw_arguments}".encode()).hexdigest()[:16]
    return GenerationResponse(
        tool_calls=(
            ToolCall(
                id=f"{prefix}_{digest}",
                name=name,
                arguments=dict(arguments),
                raw_arguments=raw_arguments,
            ),
        )
    )


def deterministic_forced_read_turn(request: Any, name: str) -> Any | None:
    arguments = _deterministic_read_arguments(request, name)
    if arguments is None:
        return None
    return _response_for_call(name, arguments, prefix="host_read")


def _focused_argument_messages(
    request: Any,
    name: str,
    *,
    repair_error: str = "",
) -> tuple[dict[str, Any], ...]:
    messages = [
        dict(raw)
        for raw in tuple(getattr(request, "messages", ()) or ())
        if isinstance(raw, Mapping)
    ]
    if name == "apply_source_edit":
        instruction = (
            f"HOST ACTION IS FIXED: {name}. Do not emit a tool/function tag. "
            "Return exactly one JSON object containing only the arguments for that action. "
            "The 'operation' field specifies the edit type (one of: 'create_file', 'replace_exact', 'insert_before', 'insert_after', 'create_java_type', 'add_java_import', 'insert_java_member', 'delete_file'). "
            "Use 'path' for target file, 'old' for span to replace, and 'new' (or 'content') for replacement text."
        )
    else:
        instruction = (
            f"HOST ACTION IS FIXED: {name}. Do not choose or emit a tool/function name. "
            "Return exactly one JSON object containing only the arguments for that host action. "
            "Use the supplied JSON schema exactly. The host will execute the action after validation."
        )
    if repair_error:
        instruction += (
            " The previous argument object was invalid. Repair the arguments only; do not repeat the "
            f"same invalid object. Validation: {repair_error[:_MAX_ARGUMENT_ERROR_CHARS]}"
        )
    # Qwen3.5 permits a system message only at the beginning of a chat template. This
    # is a current-turn instruction, so append it as user input rather than creating an
    # illegal trailing system role after the accumulated tool conversation.
    messages.append({"role": "user", "content": instruction})
    return tuple(messages)


def _argument_page_request(
    request: Any,
    name: str,
    parameters: Mapping[str, Any],
    *,
    repair_error: str = "",
) -> Any:
    return replace(
        request,
        messages=_focused_argument_messages(request, name, repair_error=repair_error),
        tools=(),
        tool_validation_schemas=(),
        tool_choice=None,
        parallel_tool_calls=False,
        response_format="json",
        response_schema=dict(parameters),
    )


def _argument_failure(
    turn: Any,
    parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str, str]:
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    content = str(getattr(turn, "content", "") or "").strip()
    if calls:
        names = ",".join(str(getattr(call, "name", "")).strip() for call in calls)
        reason = f"argument-only page emitted tool calls instead of JSON arguments: {names or '<unknown>'}"
        fingerprint = hashlib.sha256(reason.encode("utf-8")).hexdigest()
        return None, reason, fingerprint
    try:
        validate_structured_output(
            content,
            response_format="json",
            response_schema=parameters,
        )
        decoded = json.loads(content)
        if not isinstance(decoded, Mapping):
            raise TypeError("host action arguments must be a JSON object")
        raw = json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return dict(decoded), "", hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except StructuredOutputValidationError as exc:
        reason = "; ".join(exc.errors)[:_MAX_ARGUMENT_ERROR_CHARS]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        reason = f"{type(exc).__name__}: {exc}"[:_MAX_ARGUMENT_ERROR_CHARS]
    fingerprint = hashlib.sha256(f"{content}\0{reason}".encode()).hexdigest()
    return None, reason, fingerprint


def _structured_exception_failure(
    exc: BaseException,
) -> tuple[Mapping[str, Any] | None, str, str] | None:
    candidate: Any = exc
    cause = getattr(exc, "cause", None)
    if isinstance(cause, StructuredOutputValidationError):
        candidate = cause
    if not isinstance(candidate, StructuredOutputValidationError):
        return None
    reason = "; ".join(candidate.errors)[:_MAX_ARGUMENT_ERROR_CHARS]
    fingerprint = hashlib.sha256(
        f"{candidate.output}\0{reason}".encode()
    ).hexdigest()
    return None, reason, fingerprint


def _argument_attempt(
    current: Any,
    adapter: Any,
    page_request: Any,
    parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str, str]:
    try:
        turn = current(adapter, page_request)
    except BaseException as exc:
        structured = _structured_exception_failure(exc)
        if structured is None:
            raise
        return structured
    return _argument_failure(turn, parameters)


def host_selected_argument_turn(
    current: Any,
    adapter: Any,
    request: Any,
    name: str,
    *,
    prefix: str = "host_action",
) -> Any:
    """Generate only arguments for one already-selected action with one repair page."""

    from .model_adapters import ModelConfigurationError

    parameters = _parameters(_selected_schema(request, name))
    first = _argument_page_request(request, name, parameters)
    arguments, error, first_fingerprint = _argument_attempt(
        current, adapter, first, parameters
    )
    if arguments is not None:
        return _response_for_call(name, arguments, prefix=prefix)

    repair = _argument_page_request(
        request,
        name,
        parameters,
        repair_error=error,
    )
    repaired, repair_error, second_fingerprint = _argument_attempt(
        current, adapter, repair, parameters
    )
    if repaired is not None:
        return _response_for_call(name, repaired, prefix=prefix)

    fixed_point = first_fingerprint == second_fingerprint
    suffix = (
        "repeated-invalid-argument fixed point"
        if fixed_point
        else "bounded argument repair exhausted"
    )
    raise ModelConfigurationError(
        f"Host-selected action {name!r} {suffix}; first={first_fingerprint[:12]} "
        f"retry={second_fingerprint[:12]} error={repair_error or error}."
    )


def host_selected_mutation_turn(current: Any, adapter: Any, request: Any, name: str) -> Any:
    return host_selected_argument_turn(
        current,
        adapter,
        request,
        name,
        prefix="host_mutation",
    )


def _single_tool_request(request: Any, name: str) -> Any:
    selected = (_selected_schema(request, name),)
    return replace(
        request,
        tools=selected,
        tool_validation_schemas=selected,
        tool_choice="required",
        parallel_tool_calls=False,
    )


def _contains_exact_call(turn: Any, name: str) -> bool:
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    return len(calls) == 1 and str(getattr(calls[0], "name", "")).strip() == name


def _call_names(turn: Any) -> str:
    return ",".join(
        str(getattr(call, "name", "")).strip()
        for call in tuple(getattr(turn, "tool_calls", ()) or ())
    ) or "<prose>"


def _native_probe_request(request: Any) -> Any:
    schema = {
        "type": "function",
        "function": {
            "name": _NATIVE_PROBE_TOOL,
            "description": "MMM startup/preflight required-tool capability probe",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"nonce": {"type": "string", "const": "mmm"}},
                "required": ["nonce"],
            },
        },
    }
    return replace(
        request,
        messages=(
            {
                "role": "system",
                "content": "Capability probe. Call the only available function exactly once.",
            },
            {
                "role": "user",
                "content": "Call the available function with nonce set to mmm.",
            },
        ),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice="required",
        parallel_tool_calls=False,
        response_format="text",
        response_schema=None,
        media_paths=(),
    )


def _native_probe_key(adapter: Any, request: Any) -> tuple[str, str] | None:
    try:
        endpoint = str(adapter._server_url(request)).strip().rstrip("/")
    except Exception:  # noqa: BLE001 - optional adapter endpoint boundary
        return None
    model_id = str(getattr(getattr(adapter, "config", None), "model_id", "local"))
    return (endpoint, model_id) if endpoint else None


def _native_required_supported(current: Any, adapter: Any, request: Any) -> bool:
    key = _native_probe_key(adapter, request)
    if key is None:
        return False
    negative_ttl = _positive_seconds(
        "MMM_LLAMA_NATIVE_TOOL_NEGATIVE_TTL_SECONDS",
        _DEFAULT_NATIVE_NEGATIVE_TTL_SECONDS,
    )
    transient_cooldown = _positive_seconds(
        "MMM_LLAMA_NATIVE_TOOL_TRANSIENT_COOLDOWN_SECONDS",
        _DEFAULT_NATIVE_TRANSIENT_COOLDOWN_SECONDS,
    )

    # Serialize probes only per endpoint/model. Different local models remain concurrent,
    # while duplicate simultaneous requests do not each launch the same capability decode.
    with _native_probe_key_lock(key):
        now = time.monotonic()
        with _NATIVE_PROBE_LOCK:
            cached = _NATIVE_PROBE_CACHE.get(key)
            negative_at = _NATIVE_PROBE_NEGATIVE_AT.get(key)
            transient_at = _NATIVE_PROBE_TRANSIENT_AT.get(key)
            if cached is True:
                return True
            if cached is False and negative_at is not None:
                if now - negative_at < negative_ttl:
                    return False
                _NATIVE_PROBE_CACHE.pop(key, None)
                _NATIVE_PROBE_NEGATIVE_AT.pop(key, None)
            elif cached is False:
                # Reprobe legacy unbounded negative entries instead of inheriting a
                # permanent false capability state.
                _NATIVE_PROBE_CACHE.pop(key, None)
            if transient_at is not None and now - transient_at < transient_cooldown:
                return False

        supported = False
        try:
            turn = current(adapter, _native_probe_request(request))
            if _contains_exact_call(turn, _NATIVE_PROBE_TOOL):
                call = next(iter(getattr(turn, "tool_calls", ()) or ()))
                arguments = getattr(call, "arguments", {})
                supported = (
                    isinstance(arguments, Mapping)
                    and arguments.get("nonce") == "mmm"
                )
        except Exception as exc:  # noqa: BLE001 - capability transport/protocol boundary
            with _NATIVE_PROBE_LOCK:
                if _native_protocol_failure(exc):
                    _NATIVE_PROBE_CACHE[key] = False
                    _NATIVE_PROBE_NEGATIVE_AT[key] = now
                    _NATIVE_PROBE_TRANSIENT_AT.pop(key, None)
                    reason = "protocol"
                    retry_after = negative_ttl
                else:
                    _NATIVE_PROBE_CACHE.pop(key, None)
                    _NATIVE_PROBE_TRANSIENT_AT[key] = now
                    reason = "transient"
                    retry_after = transient_cooldown
            print(
                "llama native forced-tool preflight:",
                " supported=unknown" if reason == "transient" else " supported=no",
                f" reason={reason}",
                f" model={key[1]}",
                f" retry_after={retry_after:.0f}s",
                flush=True,
            )
            return False

        with _NATIVE_PROBE_LOCK:
            _NATIVE_PROBE_TRANSIENT_AT.pop(key, None)
            _NATIVE_PROBE_CACHE[key] = supported
            if supported:
                _NATIVE_PROBE_NEGATIVE_AT.pop(key, None)
            else:
                _NATIVE_PROBE_NEGATIVE_AT[key] = now
        print(
            "llama native forced-tool preflight:",
            f" supported={'yes' if supported else 'no'}",
            f" model={key[1]}",
            "" if supported else f" retry_after={negative_ttl:.0f}s",
            sep="",
            flush=True,
        )
        return supported


def _native_probe_cache_key(adapter: Any, request: Any) -> tuple[str, str] | None:
    return _native_probe_key(adapter, request)


def _mark_native_unsupported(adapter: Any, request: Any) -> None:
    key = _native_probe_cache_key(adapter, request)
    if key is not None:
        with _NATIVE_PROBE_LOCK:
            _NATIVE_PROBE_CACHE[key] = False
            _NATIVE_PROBE_NEGATIVE_AT[key] = time.monotonic()
            _NATIVE_PROBE_TRANSIENT_AT.pop(key, None)


def _native_protocol_failure(exc: BaseException) -> bool:
    cause = getattr(exc, "cause", exc)
    if not isinstance(cause, RuntimeError):
        return False
    # A native forced-tool probe can succeed with a tiny scalar schema and still fail
    # on real arguments. Treat strict parser failures as transport/protocol failures so
    # the existing bounded argument-only path gets one chance with the action already
    # selected by the host. This does not weaken schema validation or retry indefinitely.
    if type(cause).__name__ == "ToolCallValidationError":
        return True
    text = str(cause).casefold()
    markers = (
        "did not emit a tool call",
        "violated named tool_choice",
        "no semantic action",
        "tool continuation without a semantic action",
        "unexpected empty grammar stack",
    )
    return any(marker in text for marker in markers)


def _install_adapter_class(
    cls: Any,
    *,
    transport_name: str,
    deterministic_stale_read: bool,
    probe_native_required: bool = False,
) -> None:
    current = cls.generate_turn
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate_turn(self: Any, request: Any) -> Any:
        from .model_adapters import ModelConfigurationError

        name = forced_tool_name(getattr(request, "tool_choice", None))
        if not name:
            return current(self, request)

        local_native_mutation = probe_native_required and name in _SOURCE_MUTATION_TOOLS
        if name in _SOURCE_MUTATION_TOOLS and not local_native_mutation:
            return host_selected_mutation_turn(current, self, request, name)

        if deterministic_stale_read:
            deterministic = deterministic_forced_read_turn(request, name)
            if deterministic is not None:
                return deterministic

        if (
            probe_native_required
            and not local_native_mutation
            and not _native_required_supported(current, self, request)
        ):
            return host_selected_argument_turn(current, self, request, name)

        try:
            turn = current(self, _single_tool_request(request, name))
        except BaseException as exc:
            if probe_native_required and _native_protocol_failure(exc):
                if not local_native_mutation:
                    _mark_native_unsupported(self, request)
                return host_selected_argument_turn(current, self, request, name)
            raise
        if _contains_exact_call(turn, name):
            return turn
        if probe_native_required:
            if not local_native_mutation:
                _mark_native_unsupported(self, request)
            return host_selected_argument_turn(current, self, request, name)
        raise ModelConfigurationError(
            f"{transport_name} failed the host-selected action {name!r}; received "
            f"{_call_names(turn)}. The host will not repeat the same tool-name selection."
        )

    setattr(generate_turn, _MARKER, True)
    generate_turn._mmm_forced_tool_execution_v2 = True
    generate_turn._mmm_forced_tool_single_surface = True
    generate_turn._mmm_forced_tool_single_context = True
    generate_turn._mmm_forced_tool_required_transport = True
    generate_turn._mmm_host_selected_mutation_arguments = True
    generate_turn._mmm_local_native_mutation_transport = probe_native_required
    generate_turn._mmm_native_forced_tool_preflight = probe_native_required
    cls.generate_turn = generate_turn


def install(*, openai_compatible_module: Any, llama_cpp_module: Any | None = None) -> None:
    _install_adapter_class(
        openai_compatible_module.OpenAICompatibleAdapter,
        transport_name="Remote model",
        deterministic_stale_read=False,
    )
    if llama_cpp_module is not None:
        _install_adapter_class(
            llama_cpp_module.LlamaCppAdapter,
            transport_name="Local llama model",
            deterministic_stale_read=True,
            probe_native_required=True,
        )


__all__ = [
    "deterministic_forced_read_turn",
    "forced_tool_name",
    "host_selected_argument_turn",
    "host_selected_mutation_turn",
    "install",
]
