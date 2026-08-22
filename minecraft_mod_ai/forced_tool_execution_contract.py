from __future__ import annotations

"""Make host-forced tool choices executable, not advisory.

Several production paths can require one exact function call: writable source mutation,
mandatory RAG retrieval, and structured decisions. Every transport must therefore make
that requirement visible to the model before validating it after generation.

Remote OpenAI-compatible endpoints can use their native ``required`` transport after
narrowing the visible surface to one function. Local llama.cpp also narrows to one
schema and records the selected name in host metadata; its hardware transport renders
wire ``required`` while managed pure-content parsing returns raw Qwen markup for host
validation. Causal stale-tool recovery owns validation-only historical tools, so this
layer returns those calls unchanged instead of multiplying full model decodes.
"""

import hashlib
import json
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_forced_tool_execution_v1"
_LOCAL_MARKER = "_mmm_local_forced_tool_execution_v1"
_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
_HOST_FORCED_TOOL_METADATA_KEY = "_mmm_host_forced_tool"
_DETERMINISTIC_READ_TOOLS = frozenset({"search_code_rag", "search_project_rag"})
_MAX_FALLBACK_QUERY_CHARS = 4096
_MAX_FALLBACK_ERROR_CHARS = 768
_SOURCE_MUTATION_TOOLS = frozenset(
    {"apply_source_edit", "apply_source_patch", "apply_java_operations", "repair_project"}
)
_FIRST_LOCAL_INSTRUCTION = (
    "The host requires the only available function for this turn. Call it exactly once "
    "with schema-valid arguments. Do not answer in prose instead of the required call."
)
_RETRY_INSTRUCTION = (
    "The previous assistant turn did not satisfy the host-required function call. "
    "Call the only available function exactly once now. Do not answer in prose."
)


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def forced_tool_name(tool_choice: Any) -> str:
    """Return the exact host-required function name, or an empty string for auto/none."""

    if not isinstance(tool_choice, Mapping):
        return ""
    if str(tool_choice.get("type", "")).strip() != "function":
        return ""
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def host_forced_tool_name(request: Any) -> str:
    """Return the local prompt-forced tool retained in host-only request metadata."""

    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get(_HOST_FORCED_TOOL_METADATA_KEY, "")).strip()


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
    """Return parseable historical tools that are not executable this turn."""

    visible = _schema_names(getattr(request, "tools", ()))
    validation = _schema_names(getattr(request, "tool_validation_schemas", ()))
    return validation - visible


def _is_validation_only_stale_turn(request: Any, turn: Any) -> bool:
    """Whether the turn belongs to the outer causal stale-tool recovery owner."""

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
    """Keep textual routing guidance aligned with the one executable forced schema."""

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


def _single_tool_request(
    request: Any,
    name: str,
    *,
    retry: bool,
    native_required: bool,
) -> Any:
    """Reduce one forced turn to one schema without dropping request metadata."""

    selected = _selected_schema(request, name)
    messages: Sequence[Mapping[str, Any]] = _narrow_capability_context(
        request.messages,
        selected,
    )
    instruction = _RETRY_INSTRUCTION if retry else _FIRST_LOCAL_INSTRUCTION
    if retry or not native_required:
        messages = (*tuple(messages), {"role": "system", "content": instruction})

    # Remote endpoints retain required directly. Local llama.cpp keeps this host-facing
    # request as auto plus selected-name metadata; its hardware layer converts the one
    # schema to wire required while the raw Qwen result remains host-validated.
    changes: dict[str, Any] = {
        "messages": messages,
        "tools": selected,
        "tool_choice": "required" if native_required else "auto",
        "parallel_tool_calls": False,
    }
    if not native_required and hasattr(request, "metadata"):
        raw_metadata = getattr(request, "metadata", None)
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata[_HOST_FORCED_TOOL_METADATA_KEY] = name
        changes["metadata"] = metadata
    return replace(request, **changes)


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
    """Return safe repair anchors, never source/replacement bodies."""

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
    """Derive one deterministic repair query from host-owned request context."""

    base = ""
    candidates = (
        getattr(request, "task", ""),
        getattr(request, "prompt", ""),
    )
    for raw in candidates:
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
    return " ".join(value for value in (base, failure) if value)[
        :_MAX_FALLBACK_QUERY_CHARS
    ]


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
    """Reject unknown formats before jsonschema treats them as annotations."""

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
    """Validate a complete host-synthesized call against its declared JSON Schema."""

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
        # Resolution errors and unsupported schema extensions must not authorize a
        # fabricated host call. Model decoding remains the fail-closed fallback.
        return False


def _deterministic_read_arguments(request: Any, name: str) -> dict[str, Any] | None:
    """Build only schema-proven, read-only arguments; unknown requirements fail closed."""

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
    # Project RAG is exact-version retrieval. Even if a transport schema exposes a
    # host-side default, never synthesize this call without the structured approved
    # target carried by the request.
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
    # A malformed schema cannot authorize a fabricated call. In particular, do not
    # manufacture arguments when it declares required names outside properties.
    if any(key not in properties for key in required):
        return None
    if not _arguments_match_schema(arguments, raw_parameters):
        return None
    return arguments


def deterministic_forced_read_turn(request: Any, name: str) -> Any | None:
    """Return a host-derived read-only tool call, or ``None`` when not provable."""

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


def _install_remote_adapter_class(cls: Any) -> None:
    current = cls.generate_turn
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate_turn(self: Any, request: Any) -> Any:
        from .model_adapters import ModelConfigurationError

        name = forced_tool_name(getattr(request, "tool_choice", None))
        if not name:
            return current(self, request)

        first = current(
            self,
            _single_tool_request(
                request,
                name,
                retry=False,
                native_required=True,
            ),
        )
        if _contains_exact_call(first, name):
            return first
        # Causal stale recovery deliberately keeps previously authorized tools
        # parseable through tool_validation_schemas while exposing only the current
        # frontier in request.tools. It owns discard/re-sync for those calls. Retrying
        # here would hide the stale result from that owner and duplicate full decodes.
        if _is_validation_only_stale_turn(request, first):
            return first

        second = current(
            self,
            _single_tool_request(
                request,
                name,
                retry=True,
                native_required=True,
            ),
        )
        if _contains_exact_call(second, name):
            return second

        raise ModelConfigurationError(
            "Remote model violated the host-forced single-tool contract after the "
            f"bounded retry for {name!r}; first={_call_names(first)}, "
            f"retry={_call_names(second)}."
        )

    setattr(generate_turn, _MARKER, True)
    generate_turn._mmm_forced_tool_single_surface = True
    generate_turn._mmm_forced_tool_single_context = True
    generate_turn._mmm_forced_tool_required_transport = True
    generate_turn._mmm_forced_tool_bounded_retry = True
    cls.generate_turn = generate_turn


def _install_local_adapter_class(cls: Any) -> None:
    current = cls.generate_turn
    if getattr(current, _LOCAL_MARKER, False):
        return

    @wraps(current)
    def generate_turn(self: Any, request: Any) -> Any:
        from .model_adapters import ModelConfigurationError

        name = forced_tool_name(getattr(request, "tool_choice", None))
        if not name:
            return current(self, request)

        first = current(
            self,
            _single_tool_request(
                request,
                name,
                retry=False,
                native_required=False,
            ),
        )
        if _contains_exact_call(first, name):
            return first
        if _is_validation_only_stale_turn(request, first):
            deterministic = deterministic_forced_read_turn(request, name)
            if deterministic is not None:
                return deterministic
            return first

        second = current(
            self,
            _single_tool_request(
                request,
                name,
                retry=True,
                native_required=False,
            ),
        )
        if _contains_exact_call(second, name):
            return second

        raise ModelConfigurationError(
            "Local llama model violated the host-forced single-tool contract after the "
            f"bounded prompt-enforced retry for {name!r}; first={_call_names(first)}, "
            f"retry={_call_names(second)}."
        )

    setattr(generate_turn, _LOCAL_MARKER, True)
    generate_turn._mmm_forced_tool_single_surface = True
    generate_turn._mmm_forced_tool_single_context = True
    generate_turn._mmm_forced_tool_prompt_transport = True
    generate_turn._mmm_forced_tool_bounded_retry = True
    cls.generate_turn = generate_turn


def install(*, openai_compatible_module: Any, llama_cpp_module: Any | None = None) -> None:
    """Install one exact-tool forcing policy across remote and local transports."""

    _install_remote_adapter_class(openai_compatible_module.OpenAICompatibleAdapter)
    if llama_cpp_module is not None:
        _install_local_adapter_class(llama_cpp_module.LlamaCppAdapter)


__all__ = [
    "deterministic_forced_read_turn",
    "forced_tool_name",
    "host_forced_tool_name",
    "install",
]
