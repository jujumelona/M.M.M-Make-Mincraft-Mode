from __future__ import annotations

"""Qwen-family agent policy for native llama.cpp tool loops.

The MCP/tool surface is intentionally shared across model families. This module is a
late, transparent decorator over the already-composed llama payload stack: every
non-Qwen3.6 request is returned exactly as produced by the existing runtime policy.
Only Qwen3.6 autonomous agent continuations opt into thinking preservation and
historical ``reasoning_content`` restoration.
"""

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

from .model_adapters.base import GenerationRequest, GenerationResponse
from .qwen_model_profiles import QWEN36_PRECISE_CODING, qwen_family

_MAX_REASONING_TRACES = 64
_REASONING_TRACE_LOCK = threading.RLock()
_INSTALLED = False


def _model_family(model_id: str) -> str:
    return qwen_family(model_id) or "other"


def _forced_tool_choice(tool_choice: Any) -> bool:
    return isinstance(tool_choice, Mapping) or str(tool_choice or "").casefold() == "required"


def _request_messages(request: Any) -> Sequence[Mapping[str, Any]]:
    raw = getattr(request, "messages", ()) or ()
    return tuple(message for message in raw if isinstance(message, Mapping))


def _assistant_has_agent_history(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        message.get("role") == "assistant"
        and (
            bool(message.get("tool_calls"))
            or bool(str(message.get("reasoning_content") or "").strip())
        )
        for message in messages
    )


def _qwen36_agent_request(request: Any) -> bool:
    tool_choice = getattr(request, "tool_choice", None)
    if _forced_tool_choice(tool_choice):
        return False
    tools = getattr(request, "tools", ()) or ()
    if tools and tool_choice in (None, "auto"):
        return True
    return _assistant_has_agent_history(_request_messages(request))


def _apply_family_payload_policy(
    payload: dict[str, Any],
    *,
    model_id: str,
    request: Any,
) -> dict[str, Any]:
    """Mutate only autonomous Qwen3.6 agent turns; all other payloads pass through."""

    if _model_family(model_id) != "qwen3.6" or not _qwen36_agent_request(request):
        return payload

    payload.pop("reasoning_effort", None)
    payload["chat_template_kwargs"] = {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    payload.pop("repetition_penalty", None)
    payload.update(QWEN36_PRECISE_CODING)
    return payload


def _canonical_arguments(value: Any) -> str:
    if isinstance(value, str):
        raw = value.strip() or "{}"
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return json.dumps(
            decoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    if isinstance(value, Mapping):
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    return str(value)


def _message_signature(message: Mapping[str, Any]) -> str:
    if message.get("role") != "assistant":
        return ""
    calls: list[dict[str, str]] = []
    for raw in message.get("tool_calls") or ():
        if not isinstance(raw, Mapping):
            continue
        function = raw.get("function")
        if not isinstance(function, Mapping):
            continue
        calls.append(
            {
                "id": str(raw.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": _canonical_arguments(function.get("arguments", "{}")),
            }
        )
    content = message.get("content")
    normalized_content = content if isinstance(content, str) else ""
    if not normalized_content and not calls:
        return ""
    encoded = json.dumps(
        {"content": normalized_content, "tool_calls": calls},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_signature(response: GenerationResponse) -> str:
    calls = [
        {
            "id": call.id,
            "name": call.name,
            "arguments": _canonical_arguments(
                call.raw_arguments or dict(call.arguments)
            ),
        }
        for call in response.tool_calls
    ]
    if not response.content and not calls:
        return ""
    encoded = json.dumps(
        {"content": response.content, "tool_calls": calls},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trace_store(adapter: Any) -> OrderedDict[str, str]:
    current = getattr(adapter, "_mmm_qwen36_reasoning_traces", None)
    if isinstance(current, OrderedDict):
        return current
    store: OrderedDict[str, str] = OrderedDict()
    setattr(adapter, "_mmm_qwen36_reasoning_traces", store)
    return store


def _inject_reasoning_history(
    adapter: Any,
    request: GenerationRequest,
) -> GenerationRequest:
    """Restore only traces whose exact assistant exchange is present in history."""

    messages = _request_messages(request)
    signed_messages = tuple(
        (message, _message_signature(message)) for message in messages
    )
    signatures = [signature for _, signature in signed_messages if signature]
    tools = getattr(request, "tools", ()) or ()
    with _REASONING_TRACE_LOCK:
        store = _trace_store(adapter)
        if tools and getattr(request, "tool_choice", None) in (None, "auto") and not any(
            signature in store for signature in signatures
        ):
            # A fresh request cannot inherit an unrelated trace. Keep the bounded store
            # intact because the same adapter may serve concurrent independent loops.
            return request

        changed = False
        rewritten: list[Mapping[str, Any]] = []
        for raw, signature in signed_messages:
            message = dict(raw)
            reasoning = store.get(signature) if signature else None
            if (
                reasoning
                and message.get("role") == "assistant"
                and not str(message.get("reasoning_content") or "").strip()
            ):
                message["reasoning_content"] = reasoning
                changed = True
            rewritten.append(message)
    if not changed:
        return request
    return replace(request, messages=tuple(rewritten))


def _remember_reasoning(adapter: Any, response: GenerationResponse) -> None:
    reasoning = response.reasoning_content.strip()
    signature = _response_signature(response)
    if not reasoning or not signature:
        return
    with _REASONING_TRACE_LOCK:
        store = _trace_store(adapter)
        store[signature] = reasoning
        store.move_to_end(signature)
        while len(store) > _MAX_REASONING_TRACES:
            store.popitem(last=False)


def install() -> None:
    """Install the Qwen3.6 incremental backend policy exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import llama_server_hardware_policy
    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    current_payload = llama_server_hardware_policy._server_payload
    if not getattr(current_payload, "_mmm_qwen_family_agent_policy", False):

        @wraps(current_payload)
        def server_payload(adapter: Any, request: Any) -> dict[str, Any]:
            payload = current_payload(adapter, request)
            config = getattr(adapter, "config", None)
            model_id = str(getattr(config, "model_id", ""))
            return _apply_family_payload_policy(
                payload,
                model_id=model_id,
                request=request,
            )

        server_payload._mmm_qwen_family_agent_policy = True  # type: ignore[attr-defined]
        llama_server_hardware_policy._server_payload = server_payload

    current_generate_turn = LlamaCppAdapter.generate_turn
    if not getattr(current_generate_turn, "_mmm_qwen36_reasoning_history", False):

        @wraps(current_generate_turn)
        def generate_turn(
            self: Any,
            request: GenerationRequest,
        ) -> GenerationResponse:
            config = getattr(self, "config", None)
            family = _model_family(str(getattr(config, "model_id", "")))
            qwen36_agent = family == "qwen3.6" and _qwen36_agent_request(request)
            prepared = _inject_reasoning_history(self, request) if qwen36_agent else request
            response = current_generate_turn(self, prepared)
            if qwen36_agent:
                _remember_reasoning(self, response)
            return response

        generate_turn._mmm_qwen36_reasoning_history = True  # type: ignore[attr-defined]
        LlamaCppAdapter.generate_turn = generate_turn

    _INSTALLED = True


__all__ = ["install"]
