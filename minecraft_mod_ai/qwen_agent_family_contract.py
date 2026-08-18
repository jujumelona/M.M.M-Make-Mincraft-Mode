from __future__ import annotations

"""Qwen-family agent policy for native llama.cpp tool loops.

The MCP/tool surface is intentionally shared across model families.  This module only
adapts model-native reasoning behavior: Qwen3.5 keeps the proven non-thinking tool
path, while Qwen3.6 uses thinking preservation and restores historical
``reasoning_content`` before each continuation turn.
"""

import hashlib
import json
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .model_adapters.base import GenerationRequest, GenerationResponse

_MAX_REASONING_TRACES = 64
_INSTALLED = False


def _normalized_model_id(model_id: str) -> str:
    return "".join(ch for ch in str(model_id).casefold() if ch.isalnum())


def _model_family(model_id: str) -> str:
    normalized = _normalized_model_id(model_id)
    if "qwen36" in normalized:
        return "qwen3.6"
    if "qwen35" in normalized:
        return "qwen3.5"
    return "other"


def _forced_tool_choice(tool_choice: Any) -> bool:
    return isinstance(tool_choice, Mapping)


def _assistant_has_agent_history(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        message.get("role") == "assistant"
        and (
            bool(message.get("tool_calls"))
            or bool(str(message.get("reasoning_content") or "").strip())
        )
        for message in messages
    )


def _qwen36_agent_request(request: GenerationRequest) -> bool:
    if _forced_tool_choice(request.tool_choice):
        return False
    if request.tools and request.tool_choice == "auto":
        return True
    return _assistant_has_agent_history(request.messages)


def _apply_family_payload_policy(
    payload: dict[str, Any],
    *,
    model_id: str,
    request: GenerationRequest,
) -> dict[str, Any]:
    """Apply only model-family-specific sampling/thinking behavior."""

    family = _model_family(model_id)
    tools = bool(request.tools)

    if family == "qwen3.6" and _qwen36_agent_request(request):
        payload.pop("reasoning_effort", None)
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "preserve_thinking": True,
        }
        payload["temperature"] = 0.6
        payload["top_p"] = 0.95
        payload["top_k"] = 20
        payload["min_p"] = 0.0
        payload["presence_penalty"] = 0.0
        payload["repetition_penalty"] = 1.0
        return payload

    if family == "qwen3.5" and tools:
        # Preserve the established Qwen3.5 non-thinking tool path.  Explicitly
        # include the remaining Qwen-recommended neutral sampler controls.
        payload["reasoning_effort"] = "none"
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["temperature"] = 0.7
        payload["top_p"] = 0.8
        payload["top_k"] = 20
        payload["min_p"] = 0.0
        payload["presence_penalty"] = 1.5
        payload["repetition_penalty"] = 1.0
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

    store = _trace_store(adapter)
    signatures = [
        signature
        for message in request.messages
        if (signature := _message_signature(message))
    ]
    if request.tools and request.tool_choice == "auto" and not any(
        signature in store for signature in signatures
    ):
        # Exact signatures already prevent a fresh conversation from inheriting
        # another request's trace; keep the bounded store intact for concurrent loops.
        return request

    changed = False
    rewritten: list[Mapping[str, Any]] = []
    for raw in request.messages:
        message = dict(raw)
        signature = _message_signature(message)
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
    store = _trace_store(adapter)
    store[signature] = reasoning
    store.move_to_end(signature)
    while len(store) > _MAX_REASONING_TRACES:
        store.popitem(last=False)


def install() -> None:
    """Install the Qwen3.5/Qwen3.6 backend split exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import llama_server_hardware_policy
    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    current_payload = llama_server_hardware_policy._server_payload
    if not getattr(current_payload, "_mmm_qwen_family_agent_policy", False):

        def server_payload(adapter: Any, request: GenerationRequest) -> dict[str, Any]:
            payload = current_payload(adapter, request)
            return _apply_family_payload_policy(
                payload,
                model_id=str(getattr(adapter.config, "model_id", "")),
                request=request,
            )

        server_payload._mmm_qwen_family_agent_policy = True
        llama_server_hardware_policy._server_payload = server_payload

    current_generate_turn = LlamaCppAdapter.generate_turn
    if not getattr(current_generate_turn, "_mmm_qwen36_reasoning_history", False):

        def generate_turn(
            self: Any,
            request: GenerationRequest,
        ) -> GenerationResponse:
            family = _model_family(str(getattr(self.config, "model_id", "")))
            qwen36_agent = family == "qwen3.6" and _qwen36_agent_request(request)
            prepared = _inject_reasoning_history(self, request) if qwen36_agent else request
            response = current_generate_turn(self, prepared)
            if qwen36_agent:
                _remember_reasoning(self, response)
            return response

        generate_turn._mmm_qwen36_reasoning_history = True
        LlamaCppAdapter.generate_turn = generate_turn

    _INSTALLED = True


__all__ = ["install"]
