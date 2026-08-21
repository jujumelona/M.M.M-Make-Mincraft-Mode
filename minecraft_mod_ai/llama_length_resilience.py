from __future__ import annotations

"""Bounded recovery for llama.cpp ``finish_reason='length'`` responses.

A ``length`` stop can mean prompt/context pressure or exhaustion of the bounded output
allowance. Recovery is exactly one retry after compacting large tool observations,
while preserving the request's authoritative positive output-token policy. The retry
message budget is deliberately tighter than the normal first-pass budget so a 32K T4
slot still has room for the full coder tool decode.
"""

import json
import sys
from functools import wraps
from typing import Any, Mapping

from .model_context_budget import emergency_fit_messages

_MARKER = "_mmm_bounded_length_recovery_v2"
_LENGTH_ERROR_FRAGMENT = "reached its model/server context boundary before the assistant turn completed"
_LENGTH_RETRY_MESSAGE_BYTES = 32 * 1024


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


def install(llama_cpp_module: Any) -> None:
    """Install one non-recursive context-pressure retry around completion handling."""

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

            original_messages = tuple(payload.get("messages", ()) or ())
            fitted_messages = emergency_fit_messages(
                original_messages,
                budget_bytes=_LENGTH_RETRY_MESSAGE_BYTES,
            )
            before_bytes = _payload_bytes(original_messages)
            after_bytes = _payload_bytes(fitted_messages)
            if after_bytes >= before_bytes:
                raise

            retry_payload = dict(payload)
            retry_payload["messages"] = [dict(message) for message in fitted_messages]
            # Preserve the authoritative tool/page bound. Input fitting now reserves
            # this exact decode allowance against the live server context, so recovery
            # only needs to reclaim prompt space rather than mutate output policy.

            # stderr is mandatory: MCP stdio reserves stdout for JSON-RPC frames.
            print(
                "llama length recovery: retrying once",
                f" message_bytes={before_bytes}->{after_bytes}",
                f" max_tokens={retry_payload.get('max_tokens', 'model-default')}",
                file=sys.stderr,
                flush=True,
            )
            return current(server_url, retry_payload)

    setattr(completion_with_length_recovery, _MARKER, True)
    llama_cpp_module._completion_message = completion_with_length_recovery


__all__ = ["install"]
