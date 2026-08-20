from __future__ import annotations

"""Bounded recovery for llama.cpp ``finish_reason='length'`` responses.

OpenAI-compatible ``length`` is ambiguous: it can mean the requested completion-token
budget was exhausted, or that prompt+completion reached the server/model context
boundary.  The base adapter historically treated both as an unrecoverable context
failure.  This contract performs exactly one recovery attempt: compact large tool
observations and increase the completion allowance within a fixed ceiling.
"""

import json
import sys
from functools import wraps
from typing import Any, Mapping

from .model_context_budget import emergency_fit_messages

_MARKER = "_mmm_bounded_length_recovery_v1"
_LENGTH_ERROR_FRAGMENT = "reached its model/server context boundary before the assistant turn completed"
_MAX_RECOVERY_TOKENS = 16_384


def _payload_bytes(messages: Any) -> int:
    try:
        return len(
            json.dumps(
                list(messages or ()),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
    except Exception:
        return 0


def _recovery_max_tokens(payload: Mapping[str, Any]) -> int:
    try:
        current = max(1, int(payload.get("max_tokens", 0) or 0))
    except (TypeError, ValueError):
        current = 1
    return min(_MAX_RECOVERY_TOKENS, max(current + 2048, current * 2))


def install(llama_cpp_module: Any) -> None:
    """Install one non-recursive retry around the final completion-message owner."""

    current = llama_cpp_module._completion_message
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def completion_with_length_recovery(
        server_url: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            return current(server_url, payload)
        except RuntimeError as exc:
            if _LENGTH_ERROR_FRAGMENT not in str(exc):
                raise

            retry_payload = dict(payload)
            original_messages = tuple(payload.get("messages", ()) or ())
            fitted_messages = emergency_fit_messages(
                original_messages,
                budget_bytes=40 * 1024,
            )
            original_tokens = int(payload.get("max_tokens", 0) or 0)
            recovery_tokens = _recovery_max_tokens(payload)
            retry_payload["messages"] = [dict(message) for message in fitted_messages]
            retry_payload["max_tokens"] = recovery_tokens

            before_bytes = _payload_bytes(original_messages)
            after_bytes = _payload_bytes(fitted_messages)
            if after_bytes >= before_bytes and recovery_tokens <= original_tokens:
                raise

            # stderr is mandatory: MCP stdio reserves stdout for JSON-RPC frames.
            print(
                "llama length recovery: retrying once",
                f" message_bytes={before_bytes}->{after_bytes}",
                f" max_tokens={original_tokens}->{recovery_tokens}",
                file=sys.stderr,
                flush=True,
            )
            return current(server_url, retry_payload)

    setattr(completion_with_length_recovery, _MARKER, True)
    llama_cpp_module._completion_message = completion_with_length_recovery


__all__ = ["install"]
