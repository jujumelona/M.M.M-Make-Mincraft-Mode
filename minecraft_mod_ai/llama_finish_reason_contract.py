from __future__ import annotations

"""Classify llama.cpp ``finish_reason=length`` using returned token usage.

llama.cpp uses the same finish reason when a bounded decode exhausts ``max_tokens``
and when the server cannot complete within its context window. Those are different
agent failures: shrinking observations can recover context pressure, while a bounded
output stop must be continued by the owner that can preserve tool/workspace state.

The canonical inner agent loop gets the first chance to shrink an oversized action.
If that atomic retry still exhausts output, the typed boundary remains recoverable by
the outer custom-module checkpoint owner instead of being hidden behind a terminal
configuration error. This gives the pipeline one state-preserving continuation layer
rather than aborting the entire generation node.
"""

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

_MARKER = "_mmm_llama_finish_reason_classifier"
CONTEXT_PRESSURE = "context_pressure"
OUTPUT_EXHAUSTED = "output_exhausted"
_CONTEXT_ERROR = (
    "native llama-server reached its model/server context boundary before the "
    "assistant turn completed"
)
_OUTPUT_ERROR = (
    "native llama-server exhausted the bounded output allowance before the "
    "assistant action completed"
)
_HTTP_CONTEXT_MARKERS = (
    "exceeds the available context size",
    '"type":"exceed_context_size"',
    '"type": "exceed_context_size"',
)
_CONTEXT_RECOVERY_EXHAUSTED_ATTR = "_mmm_context_recovery_exhausted"


class LlamaCompletionBoundaryError(RuntimeError):
    """Typed llama completion stop that is safe to classify through wrapper chains."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        partial_message: Mapping[str, Any] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        max_tokens: int = 0,
    ) -> None:
        if kind not in {CONTEXT_PRESSURE, OUTPUT_EXHAUSTED}:
            raise ValueError(f"unsupported llama completion boundary kind: {kind!r}")
        self.kind = kind
        self.partial_message = _copy_partial_message(partial_message)
        self.prompt_tokens = _int_value(prompt_tokens)
        self.completion_tokens = _int_value(completion_tokens)
        self.max_tokens = _int_value(max_tokens)
        self.partial_bytes, self.partial_sha256 = partial_message_receipt(
            self.partial_message
        )
        super().__init__(message)


def completion_boundary_error(
    exc: BaseException,
) -> LlamaCompletionBoundaryError | None:
    """Return the typed boundary through backend/wrapper exception chains."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, LlamaCompletionBoundaryError):
            return current
        wrapped = getattr(current, "cause", None)
        if isinstance(wrapped, BaseException) and id(wrapped) not in seen:
            current = wrapped
            continue
        current = current.__cause__ or current.__context__
    return None


def mark_context_recovery_exhausted(exc: BaseException) -> None:
    """Mark one context boundary as already recovered by the canonical tool loop.

    The boundary object itself is preserved so callers still receive the original
    backend exception and its partial-message receipt. Only outer recovery-policy
    classification is disabled, preventing legacy layers from starting a second,
    tool-disabled recovery path after deterministic compaction already failed.
    """

    boundary = completion_boundary_error(exc)
    if boundary is not None and boundary.kind == CONTEXT_PRESSURE:
        setattr(boundary, _CONTEXT_RECOVERY_EXHAUSTED_ATTR, True)


def context_recovery_exhausted(exc: BaseException) -> bool:
    """Return whether canonical deterministic context recovery already ran."""

    boundary = completion_boundary_error(exc)
    return bool(
        boundary is not None
        and getattr(boundary, _CONTEXT_RECOVERY_EXHAUSTED_ATTR, False)
    )


def completion_boundary_kind(exc: BaseException) -> str:
    """Return a recoverable completion-boundary kind through wrapper chains.

    Context recovery has a single canonical owner and therefore becomes terminal once
    that owner marks it exhausted. Output exhaustion is different: after the inner
    atomic retry, the outer custom-module generator still owns a stronger recovery
    mechanism because it can persist staged edits and restart from a compact checkpoint.
    Therefore an ``ATOMIC_ACTION_OUTPUT_STALLED`` wrapper does not erase the underlying
    typed OUTPUT_EXHAUSTED boundary.
    """

    boundary = completion_boundary_error(exc)
    if boundary is None:
        return ""
    if getattr(boundary, _CONTEXT_RECOVERY_EXHAUSTED_ATTR, False):
        return ""
    return boundary.kind


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _copy_partial_message(
    message: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        return {}
    # Keep continuation data structural and out of the exception string. The server
    # already bounds it by max_tokens; a deep copy prevents later response mutation.
    return copy.deepcopy(dict(message))


def partial_message_receipt(message: Mapping[str, Any] | None) -> tuple[int, str]:
    """Return a non-secret progress receipt for one partial assistant message."""

    partial = _copy_partial_message(message)
    progress = {
        key: partial[key]
        for key in ("reasoning_content", "reasoning", "content", "tool_calls")
        if key in partial and partial[key] not in (None, "", (), [], {})
    }
    if not progress:
        return 0, ""
    encoded = json.dumps(
        progress,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _usage(data: Mapping[str, Any]) -> tuple[int, int]:
    raw = data.get("usage")
    if not isinstance(raw, Mapping):
        return 0, 0
    return _int_value(raw.get("prompt_tokens")), _int_value(raw.get("completion_tokens"))


def _http_context_pressure(status_code: Any, body: str) -> bool:
    """Recognize only llama.cpp's explicit prompt/context HTTP 400 contract."""

    if _int_value(status_code) != 400:
        return False
    normalized = " ".join(str(body or "").casefold().split())
    return any(marker in normalized for marker in _HTTP_CONTEXT_MARKERS)


def _length_error(
    data: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> LlamaCompletionBoundaryError:
    prompt_tokens, completion_tokens = _usage(data)
    max_tokens = _int_value(payload.get("max_tokens"))
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    partial_message = (
        choice.get("message")
        if isinstance(choice, Mapping) and isinstance(choice.get("message"), Mapping)
        else None
    )
    details = (
        f" prompt_tokens={prompt_tokens}"
        f" completion_tokens={completion_tokens}"
        f" max_tokens={max_tokens or 'model-default'}"
    )

    # Some llama.cpp builds stop one or two tokens shy of the requested bound because
    # of parser/control tokens. Treat only a near-saturated positive decode allowance
    # as output exhaustion; otherwise preserve the context-pressure recovery path.
    if max_tokens > 0 and completion_tokens >= max(1, max_tokens - 2):
        return LlamaCompletionBoundaryError(
            _OUTPUT_ERROR + "; split the action into smaller tool edits;" + details,
            kind=OUTPUT_EXHAUSTED,
            partial_message=partial_message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_tokens=max_tokens,
        )
    return LlamaCompletionBoundaryError(
        _CONTEXT_ERROR + "; compact/retrieve less evidence for this turn;" + details,
        kind=CONTEXT_PRESSURE,
        partial_message=partial_message,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        max_tokens=max_tokens,
    )


def install(llama_cpp_module: Any) -> None:
    """Own completion-boundary decoding without mutating prefill policy."""
    if bool(getattr(llama_cpp_module, _MARKER, False)):
        return

    def completion_message(
        server_url: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        response = llama_cpp_module._post_completion(server_url, payload)
        if response.status_code >= 400:
            body = llama_cpp_module._bounded_response_body(response)
            if _http_context_pressure(response.status_code, body):
                raise LlamaCompletionBoundaryError(
                    _CONTEXT_ERROR
                    + "; llama-server rejected the prompt because it exceeds the "
                    "available context size;"
                    + f" max_tokens={_int_value(payload.get('max_tokens')) or 'model-default'}",
                    kind=CONTEXT_PRESSURE,
                    max_tokens=_int_value(payload.get("max_tokens")),
                )
            raise RuntimeError(
                f"llama server returned HTTP {response.status_code}"
                + (f": {body}" if body else "")
            )
        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("native llama-server returned no completion choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise TypeError("native llama-server returned an invalid completion choice")
        finish_reason = str(choice.get("finish_reason", "") or "").strip().lower()
        if finish_reason == "length":
            raise _length_error(data, payload)
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise TypeError("native llama-server returned no assistant message")
        return message

    llama_cpp_module._completion_message = completion_message
    setattr(llama_cpp_module, _MARKER, True)


__all__ = [
    "CONTEXT_PRESSURE",
    "OUTPUT_EXHAUSTED",
    "_CONTEXT_ERROR",
    "_OUTPUT_ERROR",
    "LlamaCompletionBoundaryError",
    "_http_context_pressure",
    "_length_error",
    "completion_boundary_error",
    "completion_boundary_kind",
    "context_recovery_exhausted",
    "install",
    "mark_context_recovery_exhausted",
    "partial_message_receipt",
]
