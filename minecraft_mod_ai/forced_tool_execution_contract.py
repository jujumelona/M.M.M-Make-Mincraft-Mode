from __future__ import annotations

"""Make host-forced tool choices executable, not advisory.

Several production paths can require one exact function call: writable source mutation,
mandatory RAG retrieval, and structured decisions. Every transport must therefore make
that requirement visible to the model before validating it after generation.

Remote OpenAI-compatible endpoints can use their native ``required`` transport after
narrowing the visible surface to one function. Local llama.cpp intentionally keeps PEG
server parsing disabled, so its transport-level ``tool_choice`` remains ``none``; the
same semantic requirement is conveyed through one visible schema plus an explicit
system instruction, and this host wrapper validates the exact call with one bounded
retry. Causal stale-tool recovery owns retries for validation-only historical tools, so
this layer returns those calls unchanged instead of multiplying full model decodes.
"""

import json
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_forced_tool_execution_v1"
_LOCAL_MARKER = "_mmm_local_forced_tool_execution_v1"
_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
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

    # Remote endpoints can enforce required natively. Local llama.cpp deliberately
    # leaves PEG parsing disabled, so semantic forcing is prompt+surface owned here and
    # the inner Qwen parser must remain on auto rather than validating a force that the
    # server transport never received.
    return replace(
        request,
        messages=messages,
        tools=selected,
        tool_choice="required" if native_required else "auto",
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


__all__ = ["forced_tool_name", "install"]
