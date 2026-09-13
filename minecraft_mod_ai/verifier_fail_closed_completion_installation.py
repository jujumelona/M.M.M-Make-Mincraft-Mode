from __future__ import annotations

"""Fail closed when a coder tries to complete with unresolved verifier failure.

The progress-aware loop owns mutation/verification state, but completion generation is
centralized in ``_generate_turn_with_context_recovery``. Guarding that boundary keeps
normal verifier retry and mutation turns intact while preventing invalid exits and
unproductive repair cycles:

* prose/final summaries emitted while the latest verifier result is still FAIL;
* no-tool fixed-point finalization after a failed verifier result;
* recreating a path that the retained mutation history proves was already created;
* repair turns that omit the verifier diagnostics that must drive the next edit.

A later verifier PASS supersedes the earlier failure. No retry-count heuristic is used.
"""

import json
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from .source_mutation_contract import mutation_payload_applied
from .validation_diagnostic_contract import diagnostic_errors

_CREATE_OPERATIONS = frozenset(
    {
        "create",
        "create_file",
        "create_java_type",
        "create_class",
        "create_type",
        "write",
        "write_file",
    }
)
_REPAIR_GUIDANCE_MARKER = "MMM_VERIFIER_REPAIR_CONTEXT_V1"


def _message_payload(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    content = message.get("content")
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _latest_verifier_observation(
    loop_module: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, Mapping[str, Any] | None]:
    for message in reversed(tuple(messages)):
        if str(message.get("role") or "").strip().casefold() != "tool":
            continue
        name = str(message.get("name") or "").strip()
        if name not in loop_module._VERIFY_TOOLS:
            continue
        payload = _message_payload(message)
        if payload is None:
            continue
        outcome = str(loop_module._verification_outcome(name, payload) or "").strip().upper()
        return outcome or None, name, payload
    return None, None, None


def latest_verifier_outcome(loop_module: Any, messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the newest verifier semantic outcome represented in tool observations."""
    outcome, _name, _payload = _latest_verifier_observation(loop_module, messages)
    return outcome


def _request_tool_names(loop_module: Any, request: Any) -> frozenset[str]:
    tools = getattr(request, "tools", ())
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return frozenset()
    return frozenset(
        name
        for schema in tools
        if isinstance(schema, Mapping)
        for name in (str(loop_module._tool_name(schema) or "").strip(),)
        if name
    )


def _tool_call_arguments(call: Mapping[str, Any]) -> Mapping[str, Any] | None:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None
    raw = function.get("arguments")
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _mutation_call_arguments(
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    calls: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for message in messages:
        if str(message.get("role") or "").strip().casefold() != "assistant":
            continue
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes, bytearray)):
            continue
        for call in raw_calls:
            if not isinstance(call, Mapping):
                continue
            call_id = str(call.get("id") or "").strip()
            function = call.get("function")
            name = str(function.get("name") or "").strip() if isinstance(function, Mapping) else ""
            arguments = _tool_call_arguments(call)
            if call_id and name and arguments is not None:
                calls[call_id] = (name, arguments)
    return calls


def _normalized_path(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _applied_created_paths(messages: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Return paths whose retained create action has a receipt-proven byte diff."""
    calls = _mutation_call_arguments(messages)
    created: set[str] = set()
    for message in messages:
        if str(message.get("role") or "").strip().casefold() != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "").strip()
        call = calls.get(call_id)
        if call is None:
            continue
        name, arguments = call
        operation = str(arguments.get("operation") or "").strip().casefold()
        if operation not in _CREATE_OPERATIONS:
            continue
        payload = _message_payload(message)
        if payload is None or not mutation_payload_applied(name, payload):
            continue
        path = _normalized_path(
            arguments.get("path")
            or arguments.get("file")
            or arguments.get("target_path")
            or arguments.get("target_file")
        )
        if path:
            created.add(path)
    return frozenset(created)


def _diagnostic_receipt(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    result = payload.get("result")
    return result if isinstance(result, Mapping) else payload


def _compact_diagnostic(item: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("uri", "path", "file", "severity", "code", "source", "message", "range", "line"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    return compact


def _repair_guidance(
    verifier_name: str | None,
    verifier_payload: Mapping[str, Any] | None,
    applied_created_paths: frozenset[str],
) -> str:
    receipt = _diagnostic_receipt(verifier_payload)
    errors = diagnostic_errors(receipt) if receipt is not None else []
    diagnostics = [_compact_diagnostic(item) for item in errors]
    evidence = {
        "verifier": verifier_name,
        "diagnostics": diagnostics,
        "paths_already_created_in_this_run": sorted(applied_created_paths),
    }
    return (
        f"{_REPAIR_GUIDANCE_MARKER}\n"
        "The latest trustworthy verifier result is FAIL. This is a repair turn, not a fresh-generation turn. "
        "Use the verifier diagnostics below as the direct repair target. Any path listed under "
        "paths_already_created_in_this_run has an APPLIED byte-diff receipt and therefore exists now: "
        "do not call create_file/create/write for that path again. Apply an existing-file edit that materially "
        "changes source bytes and addresses the diagnostics, then let the host run verification again. "
        "Do not claim completion while any listed verifier error remains unresolved.\n"
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)
    )


def _guidance_already_present(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in reversed(tuple(messages)):
        if str(message.get("role") or "").strip().casefold() != "system":
            continue
        content = message.get("content")
        return isinstance(content, str) and content.startswith(_REPAIR_GUIDANCE_MARKER)
    return False


def _turn_create_conflict(turn: Any, applied_created_paths: frozenset[str]) -> str | None:
    if not applied_created_paths:
        return None
    for call in tuple(getattr(turn, "tool_calls", ()) or ()):
        if str(getattr(call, "name", "") or "").strip() != "apply_source_edit":
            continue
        arguments = getattr(call, "arguments", None)
        if not isinstance(arguments, Mapping):
            raw = getattr(call, "raw_arguments", None)
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    parsed = None
                arguments = parsed if isinstance(parsed, Mapping) else None
        if not isinstance(arguments, Mapping):
            continue
        operation = str(arguments.get("operation") or "").strip().casefold()
        if operation not in _CREATE_OPERATIONS:
            continue
        path = _normalized_path(
            arguments.get("path")
            or arguments.get("file")
            or arguments.get("target_path")
            or arguments.get("target_file")
        )
        if path and path in applied_created_paths:
            return path
    return None


def install(loop_module: Any) -> None:
    current = loop_module._generate_turn_with_context_recovery
    if getattr(current, "_mmm_verifier_fail_closed_completion", False):
        return

    @wraps(current)
    def guarded_generate_turn(
        router: Any,
        *,
        config: Any,
        adapter: Any,
        request: Any,
        messages: list[dict[str, Any]],
        media_paths: tuple[Any, ...],
        tool_choice: Any,
        parallel_tool_calls: bool,
    ) -> Any:
        unresolved_before, verifier_name, verifier_payload = _latest_verifier_observation(
            loop_module,
            messages,
        )
        tool_names = _request_tool_names(loop_module, request)
        corrective_tools = tool_names & (
            frozenset(loop_module._MUTATION_ACT_TOOLS) | frozenset(loop_module._VERIFY_TOOLS)
        )

        if unresolved_before == "FAIL" and not corrective_tools:
            raise loop_module.ModelConfigurationError(
                "VERIFICATION_FAILED_FIXED_POINT: the latest trustworthy verifier result is FAIL; "
                "the host refuses no-tool/final-summary completion until a mutation is applied and "
                "a verifier subsequently reports PASS."
            )

        applied_created_paths = _applied_created_paths(messages)
        if (
            unresolved_before == "FAIL"
            and tool_names & frozenset(loop_module._MUTATION_ACT_TOOLS)
            and not _guidance_already_present(messages)
        ):
            messages.append(
                {
                    "role": "system",
                    "content": _repair_guidance(
                        verifier_name,
                        verifier_payload,
                        applied_created_paths,
                    ),
                }
            )

        turn = current(
            router,
            config=config,
            adapter=adapter,
            request=request,
            messages=messages,
            media_paths=media_paths,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        unresolved_after = latest_verifier_outcome(loop_module, messages) or unresolved_before
        if unresolved_after == "FAIL":
            recreate_path = _turn_create_conflict(turn, applied_created_paths)
            if recreate_path is not None:
                raise loop_module.ModelConfigurationError(
                    "VERIFICATION_REPAIR_CREATE_CONFLICT: verifier-driven repair attempted to recreate "
                    f"the already APPLIED path {recreate_path!r}; repair must edit the existing file and "
                    "materially change bytes before verification can run again."
                )
            if not getattr(turn, "tool_calls", ()):
                raise loop_module.ModelConfigurationError(
                    "VERIFICATION_FAILED_PROSE_REJECTED: source verification is still FAIL; "
                    "a prose/final-summary response is not execution progress and cannot complete the coder loop."
                )
        return turn

    guarded_generate_turn._mmm_verifier_fail_closed_completion = True  # type: ignore[attr-defined]
    guarded_generate_turn.__wrapped__ = current
    loop_module._generate_turn_with_context_recovery = guarded_generate_turn


__all__ = ["install", "latest_verifier_outcome"]
