from __future__ import annotations

"""Fail closed when a coder tries to complete with unresolved verifier failure.

The progress-aware loop owns mutation/verification state, but completion generation is
centralized in ``_generate_turn_with_context_recovery``.  Guarding that boundary keeps
normal verifier retry and mutation turns intact while preventing two invalid exits:

* prose/final summaries emitted while the latest verifier result is still FAIL;
* no-tool fixed-point finalization after a failed verifier result.

A later verifier PASS supersedes the earlier failure.  No retry-count heuristic is used.
"""

import json
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any


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


def latest_verifier_outcome(loop_module: Any, messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the newest verifier semantic outcome represented in tool observations."""
    for message in reversed(tuple(messages)):
        if str(message.get("role") or "").strip().casefold() != "tool":
            continue
        name = str(message.get("name") or "").strip()
        if name not in loop_module._VERIFY_TOOLS:
            continue
        payload = _message_payload(message)
        if payload is None:
            continue
        return str(loop_module._verification_outcome(name, payload) or "").strip().upper() or None
    return None


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
        unresolved_before = latest_verifier_outcome(loop_module, messages)
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
        if unresolved_after == "FAIL" and not getattr(turn, "tool_calls", ()):
            raise loop_module.ModelConfigurationError(
                "VERIFICATION_FAILED_PROSE_REJECTED: source verification is still FAIL; "
                "a prose/final-summary response is not execution progress and cannot complete the coder loop."
            )
        return turn

    guarded_generate_turn._mmm_verifier_fail_closed_completion = True  # type: ignore[attr-defined]
    guarded_generate_turn.__wrapped__ = current
    loop_module._generate_turn_with_context_recovery = guarded_generate_turn


__all__ = ["install", "latest_verifier_outcome"]
