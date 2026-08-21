from __future__ import annotations

"""Classify llama.cpp ``finish_reason=length`` using returned token usage.

llama.cpp uses the same finish reason when a bounded decode exhausts ``max_tokens``
and when the server cannot complete within its context window. Those are different
agent failures: shrinking observations can recover context pressure, but it cannot
repair an action that is itself too large. The latter must be split into another
bounded agent action.
"""

from typing import Any, Mapping

_MARKER = "_mmm_llama_finish_reason_classifier_v1"
_CONTEXT_ERROR = (
    "native llama-server reached its model/server context boundary before the "
    "assistant turn completed"
)
_OUTPUT_ERROR = (
    "native llama-server exhausted the bounded output allowance before the "
    "assistant action completed"
)


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage(data: Mapping[str, Any]) -> tuple[int, int]:
    raw = data.get("usage")
    if not isinstance(raw, Mapping):
        return 0, 0
    return _int_value(raw.get("prompt_tokens")), _int_value(raw.get("completion_tokens"))


def _length_error(data: Mapping[str, Any], payload: Mapping[str, Any]) -> RuntimeError:
    prompt_tokens, completion_tokens = _usage(data)
    max_tokens = _int_value(payload.get("max_tokens"))
    details = (
        f" prompt_tokens={prompt_tokens}"
        f" completion_tokens={completion_tokens}"
        f" max_tokens={max_tokens or 'model-default'}"
    )

    # Some llama.cpp builds stop one or two tokens shy of the requested bound because
    # of parser/control tokens. Treat only a near-saturated positive decode allowance
    # as output exhaustion; otherwise preserve the context-pressure recovery path.
    if max_tokens > 0 and completion_tokens >= max(1, max_tokens - 2):
        return RuntimeError(_OUTPUT_ERROR + "; split the action into smaller tool edits;" + details)
    return RuntimeError(_CONTEXT_ERROR + "; compact/retrieve less evidence for this turn;" + details)


def install(llama_cpp_module: Any) -> None:
    """Own completion decoding directly, without adding another wrapper chain layer."""

    if bool(getattr(llama_cpp_module, _MARKER, False)):
        return

    def completion_message(server_url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = llama_cpp_module._post_completion(server_url, payload)
        if response.status_code >= 400:
            body = llama_cpp_module._bounded_response_body(response)
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
            raise RuntimeError("native llama-server returned an invalid completion choice")
        finish_reason = str(choice.get("finish_reason", "") or "").strip().lower()
        if finish_reason == "length":
            raise _length_error(data, payload)
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("native llama-server returned no assistant message")
        return message

    llama_cpp_module._completion_message = completion_message
    setattr(llama_cpp_module, _MARKER, True)


__all__ = ["_CONTEXT_ERROR", "_OUTPUT_ERROR", "_length_error", "install"]
