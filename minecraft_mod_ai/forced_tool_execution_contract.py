from __future__ import annotations

"""Make host-forced tool choices executable, not advisory.

Several production paths can require one exact function call: writable source mutation,
mandatory RAG retrieval, and structured decisions. Historically each caller set
``tool_choice`` and then checked the model response independently while still exposing
the rest of the current tool frontier. A model could therefore emit prose or another
action and the caller would only discover the violation after a long generation round.

The local llama.cpp adapter owns and validates its host-side tool transport directly.
This late contract only handles remote OpenAI-compatible adapters, where exact host
choices are reduced to one visible function and native ``required`` forcing. Keeping
those ownership boundaries separate avoids coupling runtime finalization to private
implementation helpers inside the local adapter.
"""

import json
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_forced_tool_execution_v1"
_CAPABILITY_PREFIX = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
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


def _single_tool_request(request: Any, name: str, *, retry: bool) -> Any:
    """Reduce a remote forced turn to one schema without dropping request metadata."""

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

    messages: Sequence[Mapping[str, Any]] = _narrow_capability_context(
        request.messages,
        selected,
    )
    if retry:
        messages = (*tuple(messages), {"role": "system", "content": _RETRY_INSTRUCTION})

    # Remote OpenAI-compatible endpoints use their native required-tool transport.
    # Exact identity is structural because only the selected schema remains visible.
    return replace(
        request,
        messages=messages,
        tools=selected,
        tool_choice="required",
        parallel_tool_calls=False,
    )


def _contains_exact_call(turn: Any, name: str) -> bool:
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    return len(calls) == 1 and str(getattr(calls[0], "name", "")).strip() == name


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

        constrained = _single_tool_request(request, name, retry=False)
        first = current(self, constrained)
        if _contains_exact_call(first, name):
            return first

        retry_request = _single_tool_request(request, name, retry=True)
        second = current(self, retry_request)
        if _contains_exact_call(second, name):
            return second

        first_calls = ",".join(
            str(getattr(call, "name", "")).strip()
            for call in tuple(getattr(first, "tool_calls", ()) or ())
        ) or "<prose>"
        second_calls = ",".join(
            str(getattr(call, "name", "")).strip()
            for call in tuple(getattr(second, "tool_calls", ()) or ())
        ) or "<prose>"
        raise ModelConfigurationError(
            "Remote model violated the host-forced single-tool contract after the "
            f"bounded retry for {name!r}; first={first_calls}, retry={second_calls}."
        )

    setattr(generate_turn, _MARKER, True)
    generate_turn._mmm_forced_tool_single_surface = True
    generate_turn._mmm_forced_tool_single_context = True
    generate_turn._mmm_forced_tool_required_transport = True
    generate_turn._mmm_forced_tool_bounded_retry = True
    cls.generate_turn = generate_turn


def install(*, openai_compatible_module: Any) -> None:
    """Install exact-tool forcing only where transport ownership is still remote."""

    _install_remote_adapter_class(openai_compatible_module.OpenAICompatibleAdapter)


__all__ = ["forced_tool_name", "install"]
