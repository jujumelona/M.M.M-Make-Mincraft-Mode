from __future__ import annotations

"""Enforce exact host-selected tool turns with one transport contract.

When the causal planner selects one exact function, every adapter receives the same
request shape: one visible schema, ``tool_choice='required'`` and serial execution.
There is no local-only auto/metadata/prompt detour. A malformed result may receive one
protocol-correction retry; causal stale-tool recovery remains the owner of stale calls.
"""

import hashlib
import json
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_forced_tool_execution_v2"
_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
_DETERMINISTIC_READ_TOOLS = frozenset({"search_code_rag", "search_project_rag"})
_MAX_FALLBACK_QUERY_CHARS = 4096
_MAX_FALLBACK_ERROR_CHARS = 768
_SOURCE_MUTATION_TOOLS = frozenset(
    {"apply_source_edit", "apply_java_operations", "repair_project"}
)
_RETRY_INSTRUCTION = (
    "The previous assistant turn did not satisfy the required function call. "
    "Call the only available function exactly once with schema-valid arguments."
)


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


def _selected_schema(request: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    from .model_adapters import ModelConfigurationError

    selected = tuple(
        schema
        for schema in request.tools
        if isinstance(schema, Mapping) and _tool_name(schema) == name
    )
    if len(selected) != 1:
        raise ModelConfigurationError(
            f"Host-forced tool {name!r} does not resolve to exactly one exposed schema."
        )
    return selected


def _schema_names(schemas: Any) -> frozenset[str]:
    try:
        candidates = tuple(schemas or ())
    except TypeError:
        return frozenset()
    return frozenset(
        name
        for schema in candidates
        if isinstance(schema, Mapping) and (name := _tool_name(schema))
    )


def _validation_only_tool_names(request: Any) -> frozenset[str]:
    visible = _schema_names(getattr(request, "tools", ()))
    validation = _schema_names(getattr(request, "tool_validation_schemas", ()))
    return validation - visible


def _is_validation_only_stale_turn(request: Any, turn: Any) -> bool:
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    if not calls:
        return False
    stale = _validation_only_tool_names(request)
    return bool(
        stale
        and all(str(getattr(call, "name", "")).strip() in stale for call in calls)
    )


def _narrow_capability_context(
    messages: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    from .agent_capability_context import build_agent_capability_context

    narrowed: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if (
            str(copied.get("role", "")).strip() == "system"
            and isinstance(content, str)
            and content.startswith(_CAPABILITY_PREFIX)
        ):
            try:
                payload = json.loads(content[len(_CAPABILITY_PREFIX) :])
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, Mapping):
                stage = str(payload.get("stage", "")).strip()
                model_role = str(
                    payload.get("execution_model_role", payload.get("model_role", ""))
                ).strip()
                if stage:
                    copied["content"] = build_agent_capability_context(
                        stage,
                        selected,
                        model_role=model_role,
                    )
        narrowed.append(copied)
    return tuple(narrowed)


def _single_tool_request(request: Any, name: str, *, retry: bool) -> Any:
    """Narrow one exact action without changing its required semantics."""

    selected = _selected_schema(request, name)
    messages: Sequence[Mapping[str, Any]] = _narrow_capability_context(
        request.messages,
        selected,
    )
    if retry:
        messages = (*tuple(messages), {"role": "system", "content": _RETRY_INSTRUCTION})
    return replace(
        request,
        messages=messages,
        tools=selected,
        tool_choice="required",
        parallel_tool_calls=False,
    )


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
    except Exception:
        return False


def _deterministic_read_arguments(request: Any, name: str) -> dict[str, Any] | None:
    if name not in _DETERMINISTIC_READ_TOOLS:
        return None
    selected = _selected_schema(request, name)
    function = selected[0].get("function")
    if not isinstance(function, Mapping):
        return None
    raw_parameters = function.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        return None
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


def deterministic_forced_read_turn(request: Any, name: str) -> Any | None:
    arguments = _deterministic_read_arguments(request, name)
    if arguments is None:
        return None
    from .model_adapters.base import GenerationResponse, ToolCall

    raw_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{name}\0{raw_arguments}".encode("utf-8")).hexdigest()[:16]
    return GenerationResponse(
        tool_calls=(
            ToolCall(
                id=f"host_read_{digest}",
                name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
            ),
        )
    )


def _contains_exact_call(turn: Any, name: str) -> bool:
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    return len(calls) == 1 and str(getattr(calls[0], "name", "")).strip() == name


def _call_names(turn: Any) -> str:
    return ",".join(
        str(getattr(call, "name", "")).strip()
        for call in tuple(getattr(turn, "tool_calls", ()) or ())
    ) or "<prose>"


def _install_adapter_class(
    cls: Any,
    *,
    transport_name: str,
    deterministic_stale_read: bool,
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

        first = current(self, _single_tool_request(request, name, retry=False))
        if _contains_exact_call(first, name):
            return first
        if _is_validation_only_stale_turn(request, first):
            if deterministic_stale_read:
                deterministic = deterministic_forced_read_turn(request, name)
                if deterministic is not None:
                    return deterministic
            return first

        second = current(self, _single_tool_request(request, name, retry=True))
        if _contains_exact_call(second, name):
            return second
        raise ModelConfigurationError(
            f"{transport_name} violated the required single-tool contract after one "
            f"protocol correction for {name!r}; first={_call_names(first)}, "
            f"retry={_call_names(second)}."
        )

    setattr(generate_turn, _MARKER, True)
    generate_turn._mmm_forced_tool_single_surface = True
    generate_turn._mmm_forced_tool_single_context = True
    generate_turn._mmm_forced_tool_required_transport = True
    generate_turn._mmm_forced_tool_protocol_retry = True
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
        )


__all__ = ["deterministic_forced_read_turn", "forced_tool_name", "install"]
