from __future__ import annotations

"""Bounded recovery for llama.cpp context-pressure completion stops.

The finish-reason contract owns classification. This module owns exactly one recovery:
compact large observations and retry a genuine context-pressure turn while preserving
the authoritative positive output-token policy. Output exhaustion is deliberately not
reinterpreted here as context pressure.
"""

import json
import sys
from functools import wraps
from typing import Any, Mapping

from .llama_finish_reason_contract import CONTEXT_PRESSURE, completion_boundary_kind
from .model_context_budget import emergency_fit_messages

_MARKER = "_mmm_bounded_length_recovery_v4"
_LENGTH_RETRY_MESSAGE_BYTES = 32 * 1024


def length_recovery_installed(completion_message: Any) -> bool:
    """Return whether the canonical context-pressure recovery owns this call path."""

    return bool(getattr(completion_message, _MARKER, False))


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
    if length_recovery_installed(current):
        return

    @wraps(current)
    def completion_with_length_recovery(
        server_url: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            return current(server_url, payload)
        except RuntimeError as exc:
            if completion_boundary_kind(exc) != CONTEXT_PRESSURE:
                raise

            original_messages = tuple(payload.get("messages", ()) or ())
            # A trailing assistant message is llama.cpp's assistant-prefill contract.
            # Compacting it would discard or rewrite the exact unfinished output and
            # could make a partial tool action appear complete. Let the typed context
            # boundary reach the producer so it can split the outstanding obligation.
            if (
                original_messages
                and isinstance(original_messages[-1], Mapping)
                and original_messages[-1].get("role") == "assistant"
            ):
                raise
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
            # Preserve the authoritative tool/page bound. Input fitting reserves this
            # exact decode allowance against the live server context; recovery reclaims
            # prompt space and never mutates output ownership.
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


__all__ = ["install", "length_recovery_installed"]
