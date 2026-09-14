from __future__ import annotations

"""Transactional retry plus per-completion observability for native llama.cpp turns.

A semantic turn may legitimately need a continuation (for example after an output
boundary or a reasoning-only response), but every additional model completion must be
observable and stale/out-of-phase tools must be rejected against the *current* tool
frontier before they can escape the adapter.
"""

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any, Callable

import httpx

from ..model_concurrency import (
    ModelExecutionDeadlineExceeded,
    remaining_model_execution_seconds,
)
from .base import ModelBackendError

_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
_MAX_RETRY_ATTEMPTS = 5
_TURN_CONTEXT = threading.local()
_PATCH_LOCK = threading.Lock()


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0.0 else default


def _retry_attempts() -> int:
    return _bounded_env_int(
        "MMM_LLAMA_TURN_RETRY_ATTEMPTS",
        _DEFAULT_RETRY_ATTEMPTS,
        minimum=1,
        maximum=_MAX_RETRY_ATTEMPTS,
    )


def _retry_backoff_seconds() -> float:
    return _positive_env_float(
        "MMM_LLAMA_TURN_RETRY_BACKOFF_SECONDS",
        _DEFAULT_RETRY_BACKOFF_SECONDS,
    )


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if isinstance(current, ModelBackendError) and isinstance(current.cause, BaseException):
            current = current.cause
            continue
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_retriable_turn_failure(exc: BaseException) -> bool:
    """Retry transport/liveness failures, never semantic/schema/deadline failures."""

    chain = _exception_chain(exc)
    if any(isinstance(item, ModelExecutionDeadlineExceeded) for item in chain):
        return False
    for item in chain:
        if isinstance(item, (httpx.TransportError, TimeoutError)):
            return True
        message = str(item).strip().lower()
        if "stream ended before the [done] marker" in message:
            return True
        if "llama server returned http 5" in message:
            return True
    return False


def _discarded_partial_hint(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        message = str(item).lower()
        if "stream" in message or "sse" in message or "transport" in message:
            return True
    return False


def _install_marker(owner: type[Any]) -> str:
    return f"{owner.__module__}.{owner.__qualname__}"


def _sleep_before_retry(*, backoff: float, attempt: int, cause: BaseException) -> None:
    """Back off only while the caller's absolute model deadline still has budget."""

    delay = max(0.0, float(backoff) * max(1, int(attempt)))
    remaining = remaining_model_execution_seconds()
    if remaining is not None:
        if remaining <= 0.0 or delay >= remaining:
            raise ModelExecutionDeadlineExceeded(
                "model execution deadline exhausted before llama turn retry"
            ) from cause
    if delay:
        time.sleep(delay)
    remaining_after_sleep = remaining_model_execution_seconds()
    if remaining_after_sleep is not None and remaining_after_sleep <= 0.0:
        raise ModelExecutionDeadlineExceeded(
            "model execution deadline exhausted during llama turn retry backoff"
        ) from cause


def _context() -> dict[str, Any] | None:
    value = getattr(_TURN_CONTEXT, "value", None)
    return value if isinstance(value, dict) else None


def _emit(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(
        "llama adapter: " + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def _payload_tool_names(payload: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    raw_tools = payload.get("tools", ())
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        return names
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        function = raw_tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", "") or "").strip()
        if name:
            names.append(name)
    return names


def _message_tool_names(message: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    raw_calls = message.get("tool_calls", ())
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        return names
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function")
        if isinstance(function, Mapping):
            name = str(function.get("name", "") or "").strip()
        else:
            name = str(raw_call.get("name", "") or "").strip()
        if name:
            names.append(name)
    return names


def _message_lengths(message: Mapping[str, Any]) -> tuple[int, int]:
    content = message.get("content")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    return (
        len(content) if isinstance(content, str) else 0,
        len(reasoning) if isinstance(reasoning, str) else 0,
    )


def _response_metadata(response: Any) -> tuple[str, str]:
    finish_reason = ""
    request_id = ""
    try:
        data = response.json()
        choices = data.get("choices") if isinstance(data, Mapping) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        if isinstance(choice, Mapping):
            finish_reason = str(choice.get("finish_reason", "") or "").strip()
    except Exception:
        pass
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        for key in ("x-request-id", "request-id", "x-llama-request-id"):
            value = str(headers.get(key, "") or "").strip()
            if value:
                request_id = value
                break
    if not request_id:
        for attr in ("request_id", "_request_id"):
            value = str(getattr(response, attr, "") or "").strip()
            if value:
                request_id = value
                break
    return finish_reason, request_id


def _completion_fields(
    ctx: dict[str, Any],
    *,
    message: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_tool_names = _message_tool_names(message or {})
    content_length, reasoning_length = _message_lengths(message or {})
    return {
        "parent_turn_id": ctx.get("parent_turn_id", ""),
        "attempt": int(ctx.get("completion_attempt", 0) or 0),
        "completion_id": ctx.get("completion_id", ""),
        "llama_request_id": ctx.get("llama_request_id", ""),
        "finish_reason": ctx.get("finish_reason", ""),
        "content_length": content_length,
        "reasoning_length": reasoning_length,
        "tool_calls_count": len(parsed_tool_names),
        "parsed_tool_names": parsed_tool_names,
        "allowed_tool_names": list(ctx.get("allowed_tool_names", ())),
    }


def _boundary_retry_reason(exc: BaseException, payload: Mapping[str, Any]) -> tuple[str, str]:
    try:
        from ..llama_finish_reason_contract import OUTPUT_EXHAUSTED, completion_boundary_error

        boundary = completion_boundary_error(exc)
    except Exception:
        boundary = None
        OUTPUT_EXHAUSTED = "output_exhausted"
    if boundary is None:
        return "", ""
    kind = str(getattr(boundary, "kind", "") or "")
    template_kwargs = payload.get("chat_template_kwargs")
    nonthinking = (
        isinstance(template_kwargs, Mapping)
        and template_kwargs.get("enable_thinking") is False
    )
    if kind == OUTPUT_EXHAUSTED and nonthinking:
        return kind, "assistant_prefill_continuation"
    return kind, ""


def _ensure_adapter_guards(adapter_type: type[Any]) -> None:
    """Install guards on the live adapter module, including wrappers installed later."""

    module = sys.modules.get(adapter_type.__module__)
    if module is None:
        return
    if not all(
        hasattr(module, name)
        for name in (
            "_post_completion",
            "_completion_message",
            "_validate_tool_calls_against_host_schema",
            "_qwen_tool_generation_response",
        )
    ):
        return

    with _PATCH_LOCK:
        current_post = module._post_completion
        if not getattr(current_post, "_mmm_completion_observability", False):
            original_post = current_post

            @wraps(original_post)
            def post_completion(server_url: str, payload: Mapping[str, Any]) -> Any:
                ctx = _context()
                if ctx is not None:
                    next_attempt = int(ctx.get("completion_attempt", 0) or 0) + 1
                    ctx["completion_attempt"] = next_attempt
                    ctx["completion_id"] = (
                        f"{ctx.get('parent_turn_id', 'llama-turn')}/completion-{next_attempt}"
                    )
                    ctx["allowed_tool_names"] = tuple(_payload_tool_names(payload))
                    ctx["finish_reason"] = ""
                    ctx["llama_request_id"] = ""
                    _emit(
                        "llama_completion_start",
                        parent_turn_id=ctx.get("parent_turn_id", ""),
                        attempt=next_attempt,
                        completion_id=ctx["completion_id"],
                        allowed_tool_names=list(ctx["allowed_tool_names"]),
                        outcome="started",
                    )
                response = original_post(server_url, payload)
                if ctx is not None:
                    finish_reason, request_id = _response_metadata(response)
                    ctx["finish_reason"] = finish_reason
                    ctx["llama_request_id"] = request_id
                return response

            post_completion._mmm_completion_observability = True  # type: ignore[attr-defined]
            module._post_completion = post_completion

        current_completion = module._completion_message
        if not getattr(current_completion, "_mmm_completion_observability", False):
            original_completion = current_completion

            @wraps(original_completion)
            def completion_message(
                server_url: str,
                payload: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                ctx = _context()
                try:
                    message = original_completion(server_url, payload)
                except Exception as exc:
                    if ctx is not None:
                        boundary_kind, retry_reason = _boundary_retry_reason(exc, payload)
                        fields = _completion_fields(ctx)
                        _emit(
                            "llama_completion_outcome",
                            **fields,
                            parse_status="exception",
                            reject_reason=boundary_kind or type(exc).__name__,
                            retry_reason=retry_reason,
                            outcome="retrying" if retry_reason else "failed",
                        )
                    raise
                if ctx is not None:
                    fields = _completion_fields(ctx, message=message)
                    has_visible = bool(fields["content_length"] or fields["tool_calls_count"])
                    if not has_visible and fields["reasoning_length"]:
                        _emit(
                            "llama_completion_outcome",
                            **fields,
                            parse_status="reasoning_only",
                            reject_reason="reasoning_only_no_semantic_output",
                            retry_reason="semantic_reasoning_continuation",
                            outcome="retrying",
                        )
                    else:
                        _emit(
                            "llama_completion_outcome",
                            **fields,
                            parse_status="message_received",
                            reject_reason="",
                            retry_reason="",
                            outcome="candidate",
                        )
                return message

            completion_message._mmm_completion_observability = True  # type: ignore[attr-defined]
            module._completion_message = completion_message

        current_validator = module._validate_tool_calls_against_host_schema
        if not getattr(current_validator, "_mmm_current_tool_frontier_guard", False):
            original_validator = current_validator

            @wraps(original_validator)
            def validate_tool_calls(calls: Sequence[Any], schemas: Mapping[str, Any]) -> None:
                allowed = tuple(str(name) for name in schemas)
                for call in calls:
                    name = str(getattr(call, "name", "") or "").strip()
                    if name and name not in schemas:
                        ctx = _context()
                        fields = _completion_fields(ctx or {})
                        if ctx is not None:
                            ctx["validation_reject_attempt"] = fields["attempt"]
                        _emit(
                            "llama_completion_outcome",
                            **fields,
                            parse_status="tool_name_not_allowed",
                            parsed_tool_names=[name],
                            tool_calls_count=1,
                            allowed_tool_names=list(allowed),
                            reject_reason="tool_name_not_allowed",
                            retry_reason="",
                            outcome="discarded",
                        )
                        raise module.ToolCallValidationError(
                            f"Qwen tool {name!r} is not exposed in the current tool frontier"
                        )
                original_validator(calls, schemas)

            validate_tool_calls._mmm_current_tool_frontier_guard = True  # type: ignore[attr-defined]
            module._validate_tool_calls_against_host_schema = validate_tool_calls

        current_qwen = module._qwen_tool_generation_response
        if not getattr(current_qwen, "_mmm_completion_observability", False):
            original_qwen = current_qwen

            @wraps(original_qwen)
            def qwen_tool_generation_response(message: Mapping[str, Any], request: Any) -> Any:
                ctx = _context()
                try:
                    turn = original_qwen(message, request)
                except Exception as exc:
                    if ctx is not None and ctx.get("validation_reject_attempt") != ctx.get(
                        "completion_attempt"
                    ):
                        fields = _completion_fields(ctx, message=message)
                        _emit(
                            "llama_completion_outcome",
                            **fields,
                            parse_status="tool_parse_or_validation_failed",
                            reject_reason=type(exc).__name__,
                            retry_reason="",
                            outcome="discarded",
                        )
                    raise
                calls = tuple(getattr(turn, "tool_calls", ()) or ())
                content = str(getattr(turn, "content", "") or "")
                if ctx is not None and (calls or content):
                    names = [str(getattr(call, "name", "") or "") for call in calls]
                    fields = _completion_fields(ctx, message=message)
                    _emit(
                        "llama_completion_outcome",
                        **fields,
                        parse_status="host_schema_validated",
                        parsed_tool_names=names,
                        tool_calls_count=len(names),
                        reject_reason="",
                        retry_reason="",
                        outcome="accepted",
                    )
                return turn

            qwen_tool_generation_response._mmm_completion_observability = True  # type: ignore[attr-defined]
            module._qwen_tool_generation_response = qwen_tool_generation_response


def install_llama_turn_retry(adapter_type: type[Any]) -> None:
    """Install idempotent transport retry and live per-completion guards."""

    _ensure_adapter_guards(adapter_type)
    current = adapter_type.generate_turn
    if getattr(current, "_mmm_transactional_turn_retry", False):
        return

    original: Callable[..., Any] = current

    @wraps(original)
    def generate_turn_with_retry(self: Any, request: Any) -> Any:
        _ensure_adapter_guards(adapter_type)
        attempts = _retry_attempts()
        backoff = _retry_backoff_seconds()
        started = time.monotonic()
        previous_context = getattr(_TURN_CONTEXT, "value", None)
        ctx: dict[str, Any] = {
            "parent_turn_id": f"llama-turn-{uuid.uuid4().hex[:16]}",
            "completion_attempt": 0,
            "completion_id": "",
            "finish_reason": "",
            "llama_request_id": "",
            "allowed_tool_names": (),
            "validation_reject_attempt": 0,
        }
        _TURN_CONTEXT.value = ctx
        try:
            for attempt in range(1, attempts + 1):
                try:
                    return original(self, request)
                except Exception as exc:
                    retriable = _is_retriable_turn_failure(exc)
                    will_retry = retriable and attempt < attempts
                    elapsed = time.monotonic() - started
                    _emit(
                        "llama_turn_failure",
                        parent_turn_id=ctx["parent_turn_id"],
                        transaction_attempt=attempt,
                        transaction_attempts=attempts,
                        error=type(exc).__name__,
                        elapsed_seconds=round(elapsed, 3),
                        partial_discarded=_discarded_partial_hint(exc),
                        retry_reason="transport_or_liveness_failure" if will_retry else "",
                        outcome="retrying" if will_retry else "failed",
                    )
                    if not will_retry:
                        raise
                    _sleep_before_retry(backoff=backoff, attempt=attempt, cause=exc)
            raise AssertionError("unreachable llama turn retry state")
        finally:
            _TURN_CONTEXT.value = previous_context

    generate_turn_with_retry._mmm_transactional_turn_retry = True  # type: ignore[attr-defined]
    generate_turn_with_retry._mmm_retry_owner = _install_marker(adapter_type)  # type: ignore[attr-defined]
    adapter_type.generate_turn = generate_turn_with_retry


__all__ = [
    "_ensure_adapter_guards",
    "_is_retriable_turn_failure",
    "install_llama_turn_retry",
]
