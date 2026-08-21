from __future__ import annotations

"""Bounded recovery for llama.cpp ``finish_reason='length'`` responses.

A ``length`` stop is treated as prompt/context pressure. Recovery is exactly one retry
after compacting large tool observations, while preserving the request's authoritative
output-token policy. In particular, bounded tool/section turns must never be widened to
``max_tokens=-1`` during recovery.
"""

import json
import sys
from functools import wraps
from typing import Any, Mapping

from .model_context_budget import emergency_fit_messages

_MARKER = "_mmm_bounded_length_recovery_v2"
_LENGTH_ERROR_FRAGMENT = "reached its model/server context boundary before the assistant turn completed"


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
                budget_bytes=40 * 1024,
            )
            before_bytes = _payload_bytes(original_messages)
            after_bytes = _payload_bytes(fitted_messages)
            if after_bytes >= before_bytes:
                raise

            retry_payload = dict(payload)
            retry_payload["messages"] = [dict(message) for message in fitted_messages]
            # Preserve the authoritative max_tokens value from the original request.
            # Replacing a bounded tool/page decode with -1 can consume the remaining
            # context before the assistant finishes its semantic action.

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
