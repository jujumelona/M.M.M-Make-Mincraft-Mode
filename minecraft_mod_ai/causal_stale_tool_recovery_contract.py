from __future__ import annotations

"""Recover stale or malformed model tool actions without ever executing them.

The causal frontier is the only executable tool surface. A broader frozen authorized
surface may remain parseable so stale model actions can be recognized and discarded,
but parser recognition never grants execution authority.

This module is the single owner of model-action re-synchronization. It handles both a
successfully parsed stale tool call and Qwen tool markup/argument failures that occur
before a ``GenerationResponse`` can be produced. In either case the semantic turn gets
at most one deterministic retry against one current legal action. Transport/backend
failures are never retried here, and unknown tools fail closed.

The recovery hook is installed on the canonical adapter class *in place*. Late runtime
contracts import that class before finalization, so rebinding one module-level class
name would leave those pre-bound aliases on the unrecovered implementation.
"""

import re
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

from . import causal_frontier_adapter as causal_frontier_adapter_module
from .causal_frontier_adapter import (
    _with_capability_context,
    current_frontier_names,
)

_MARKER = "_mmm_stale_tool_recovery_v2"
_MAX_RESYNC_ATTEMPTS = 1
_RUNTIME_CONTRACT_EPOCH = "causal-resync-v3"
_PROTOCOL_PREFIXES = (
    "Qwen ",
    "unparsed Qwen ",
    "model emitted ",
    "model did not emit ",
    "model violated ",
)
_QWEN_TOOL_PATTERNS = (
    re.compile(r"Qwen tool '([^']+)'"),
    re.compile(r"Qwen requested an unexposed tool '([^']+)'"),
    re.compile(r"named tool_choice '([^']+)'"),
)
_runtime_marker_printed = False


def _name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return str(function.get("name", "")).strip() if isinstance(function, Mapping) else ""


def _forced_tool_name(tool_choice: Any) -> str:
    if not isinstance(tool_choice, Mapping):
        return ""
    if str(tool_choice.get("type", "")).strip() != "function":
        return ""
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _stale_names(
    turn: Any,
    *,
    authorized_names: frozenset[str],
    visible_names: frozenset[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(call.name)
                for call in getattr(turn, "tool_calls", ())
                if str(call.name) in authorized_names
                and str(call.name) not in visible_names
            }
        )
    )


def _tool_name_from_protocol_error(message: str) -> str:
    for pattern in _QWEN_TOOL_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1).strip()
    return ""


def _model_tool_protocol_failure(exc: BaseException) -> tuple[str, str] | None:
    """Classify model-produced Qwen/tool-choice failures, never backend failures.

    ``LlamaCppAdapter`` historically wrapped every host parser ``RuntimeError`` in a
    ``ModelBackendError``. Until that transport boundary can be narrowed without
    changing backend-failure semantics, inspect only its explicit ``cause`` and only
    the finite parser/tool-choice prefixes emitted by the Qwen host parser. HTTP,
    timeout, server payload, hardware and process failures do not match this contract.
    """

    from .model_adapters import ModelBackendError

    if not isinstance(exc, ModelBackendError):
        return None
    cause = getattr(exc, "cause", None)
    if not isinstance(cause, RuntimeError):
        return None
    message = str(cause).strip()
    if not message.startswith(_PROTOCOL_PREFIXES):
        return None
    return message, _tool_name_from_protocol_error(message)


def _candidate_surfaces(
    self: Any,
    request: Any,
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...], dict[str, Mapping[str, Any]]]:
    candidates: Sequence[Mapping[str, Any]] = (
        self.authorized_surface or tuple(request.tools)
    )
    by_name = {_name(schema): schema for schema in candidates if _name(schema)}
    visible = tuple(current_frontier_names() or ())
    if not visible:
        visible = tuple(
            name
            for schema in tuple(request.tools)
            if (name := _name(schema)) and name in by_name
        )
    return visible, tuple(candidates), by_name


def _select_resync_tool(
    request: Any,
    *,
    visible: tuple[str, ...],
    authorized: frozenset[str],
    failed_tool: str,
) -> str:
    """Select one deterministic current action or fail closed.

    A malformed current tool retries that same tool. A known stale tool retries the
    current host-selected action. A tool that was never authorized is never converted
    into a legal request. Nameless malformed markup is recoverable only when the host
    had already selected one exact current action.
    """

    from .model_adapters import ModelConfigurationError

    visible_set = frozenset(visible)
    if failed_tool:
        if failed_tool not in authorized:
            raise ModelConfigurationError(
                f"Model emitted malformed call for unauthorized tool {failed_tool!r}."
            )
        if failed_tool in visible_set:
            return failed_tool

    requested_forced = _forced_tool_name(getattr(request, "tool_choice", None))
    if requested_forced and requested_forced in visible_set:
        return requested_forced
    if failed_tool and failed_tool in authorized and failed_tool not in visible_set:
        if visible:
            return visible[0]
        raise ModelConfigurationError(
            "Model emitted a stale authorized tool after the causal frontier became empty: "
            + failed_tool
        )
    if len(visible) == 1:
        return visible[0]
    raise ModelConfigurationError(
        "Malformed model tool output could not be deterministically mapped to one current "
        "causal-frontier action."
    )


def _resync_once(
    self: Any,
    request: Any,
    *,
    visible: tuple[str, ...],
    candidates: tuple[Mapping[str, Any], ...],
    by_name: Mapping[str, Mapping[str, Any]],
    forced_name: str,
    rejected: str,
    protocol_detail: str = "",
) -> Any:
    from .model_adapters import ModelConfigurationError

    forced_schema = by_name.get(forced_name)
    if forced_schema is None:
        raise ModelConfigurationError(
            f"Causal re-synchronization selected {forced_name!r} without an authorized schema."
        )

    self._reset_stale_guard()
    forced_tools = (forced_schema,)
    detail = " ".join(protocol_detail.split())[:240]
    feedback_text = (
        "The previous model tool action was discarded without execution. "
        f"Rejected: {rejected}. Call exactly {forced_name!r} once with schema-valid "
        "arguments; do not emit any other tool name."
    )
    if detail:
        feedback_text += f" Validation error: {detail}"
    feedback = {"role": "system", "content": feedback_text}
    retry_messages = _with_capability_context(
        (*tuple(request.messages), feedback),
        stage=self.stage,
        role=self.role,
        tools=forced_tools,
    )
    retry_request = replace(
        request,
        messages=retry_messages,
        tools=forced_tools,
        tool_validation_schemas=tuple(candidates),
        tool_choice={
            "type": "function",
            "function": {"name": forced_name},
        },
        parallel_tool_calls=False,
    )
    self._publish_frontier((forced_name,))
    print(
        "causal tool resync:",
        f"attempt=1/{_MAX_RESYNC_ATTEMPTS}",
        f"rejected={rejected}",
        f"forced={forced_name}",
        flush=True,
    )
    try:
        corrected = self.inner.generate_turn(retry_request)
    except BaseException as exc:
        protocol_failure = _model_tool_protocol_failure(exc)
        if protocol_failure is None:
            raise
        message, retry_tool = protocol_failure
        raise ModelConfigurationError(
            "Model failed the single causal-frontier re-synchronization attempt with "
            f"malformed tool output; forced={forced_name!r}, emitted={retry_tool or '<unknown>'!r}, "
            f"error={' '.join(message.split())[:240]}"
        ) from exc

    calls = tuple(getattr(corrected, "tool_calls", ()) or ())
    names = tuple(str(call.name).strip() for call in calls)
    if len(calls) == 1 and names == (forced_name,):
        self._reset_stale_guard()
        return corrected

    rejected_names = tuple(sorted(set(names))) or ("<missing-tool-call>",)
    raise ModelConfigurationError(
        "Model failed the single causal-frontier re-synchronization attempt; "
        f"forced={forced_name!r} rejected={','.join(rejected_names)} "
        f"visible={','.join(visible)}"
    )


def _install_generate_turn(base: type[Any]) -> None:
    current = base.generate_turn
    if bool(getattr(current, _MARKER, False)):
        setattr(base, _MARKER, True)
        return

    @wraps(current)
    def generate_turn(self: Any, request: Any) -> Any:
        from .model_adapters import ModelConfigurationError

        visible, candidates, by_name = _candidate_surfaces(self, request)
        authorized = frozenset(by_name)
        visible_set = frozenset(visible)
        try:
            turn = current(self, request)
        except BaseException as exc:
            protocol_failure = _model_tool_protocol_failure(exc)
            if protocol_failure is None:
                raise
            message, failed_tool = protocol_failure
            forced_name = _select_resync_tool(
                request,
                visible=visible,
                authorized=authorized,
                failed_tool=failed_tool,
            )
            return _resync_once(
                self,
                request,
                visible=visible,
                candidates=candidates,
                by_name=by_name,
                forced_name=forced_name,
                rejected=failed_tool or "<malformed-tool-markup>",
                protocol_detail=message,
            )

        stale = _stale_names(
            turn,
            authorized_names=authorized,
            visible_names=visible_set,
        )
        if not stale:
            return turn

        if not visible:
            raise ModelConfigurationError(
                "Model emitted a stale authorized tool call after the causal frontier "
                "became empty: stale=" + ",".join(stale)
            )
        forced_name = _select_resync_tool(
            request,
            visible=visible,
            authorized=authorized,
            failed_tool=stale[0],
        )
        return _resync_once(
            self,
            request,
            visible=visible,
            candidates=candidates,
            by_name=by_name,
            forced_name=forced_name,
            rejected=",".join(stale),
        )

    setattr(generate_turn, _MARKER, True)
    generate_turn.__wrapped__ = current  # type: ignore[attr-defined]
    base.generate_turn = generate_turn
    setattr(base, _MARKER, True)


def install(causal_frontier_contract_module: Any) -> None:
    global _runtime_marker_printed

    canonical = causal_frontier_adapter_module.CausalFrontierAdapter
    _install_generate_turn(canonical)
    causal_frontier_contract_module.CausalFrontierAdapter = canonical
    if not _runtime_marker_printed:
        print(
            "MMM runtime contract:",
            f"epoch={_RUNTIME_CONTRACT_EPOCH}",
            "stale_resync=1",
            "malformed_tool_resync=1",
            f"source={__file__}",
            flush=True,
        )
        _runtime_marker_printed = True


__all__ = ["install"]