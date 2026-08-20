from __future__ import annotations

"""Make host-forced native tool choices executable, not advisory.

Several production paths can require one exact native function call: writable source
mutation, mandatory RAG retrieval, and structured decisions.  Historically each
caller set ``tool_choice`` and then checked the model response independently while
still exposing the rest of the current tool frontier.  A local model could therefore
emit prose or another action and the caller would only discover the violation after a
long generation round.

This contract sits at the model-adapter boundary.  Whenever the host names one exact
function, the adapter receives only that schema.  A prose-only response gets one
bounded protocol retry with the single-tool surface kept intact.  Callers retain
their semantic checks, but they no longer have to reinvent transport enforcement.
"""

from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_forced_tool_execution_v1"
_RETRY_INSTRUCTION = (
    "The previous assistant turn did not satisfy the host-required native function "
    "call. Call the only available function exactly once now. Do not answer in prose."
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


def _single_tool_request(request: Any, name: str, *, retry: bool) -> Any:
    """Preserve the canonical request while reducing a forced turn to one schema."""

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

    messages: Sequence[Mapping[str, Any]] = request.messages
    tool_choice: Any = {
        "type": "function",
        "function": {"name": name},
    }
    if retry:
        messages = (*tuple(request.messages), {"role": "system", "content": _RETRY_INSTRUCTION})
        # Some OpenAI-compatible native servers honor `required` more reliably than
        # the named-choice object.  With exactly one schema exposed both forms have
        # identical semantics, so the retry deliberately exercises the alternate
        # protocol representation instead of repeating the same failed request.
        tool_choice = "required"

    return replace(
        request,
        messages=messages,
        tools=selected,
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )


def _contains_exact_call(turn: Any, name: str) -> bool:
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    return len(calls) == 1 and str(getattr(calls[0], "name", "")).strip() == name


def _install_adapter_class(cls: Any) -> None:
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
            "Native model violated the host-forced tool contract after both exact "
            f"single-tool protocol forms for {name!r}; first={first_calls}, "
            f"retry={second_calls}."
        )

    setattr(generate_turn, _MARKER, True)
    generate_turn._mmm_forced_tool_single_surface = True
    generate_turn._mmm_forced_tool_bounded_retry = True
    cls.generate_turn = generate_turn


def install(*, llama_cpp_module: Any, openai_compatible_module: Any) -> None:
    """Install one forced-tool transport contract on every native chat adapter."""

    _install_adapter_class(llama_cpp_module.LlamaCppAdapter)
    _install_adapter_class(openai_compatible_module.OpenAICompatibleAdapter)


__all__ = ["forced_tool_name", "install"]
