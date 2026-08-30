from __future__ import annotations

"""Registry-declared request and agent policy for llama.cpp tool loops.

The MCP/tool surface is shared across model families. Runtime-specific behavior is
selected exclusively from ``AdapterConfig.extra`` values loaded from model_registry;
this module never infers a model version, size, repository id, context length, or
sampling profile from a model name.
"""

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

from .model_adapters.base import GenerationRequest, GenerationResponse
from .qwen_family_capabilities import qwen_family_capabilities

_MAX_REASONING_TRACES = 64
_REASONING_TRACE_LOCK = threading.RLock()
_TOOL_RUNTIME_LOCK = threading.RLock()
_TOOL_SAFE_RUNTIME_ACTIVE = False
_TOOL_SAFE_RUNTIME_KEY: str | None = None
_INSTALLED = False
_SAMPLING_MODES = frozenset({"general_thinking", "precise_coding", "non_thinking"})


def _config_extra(config: Any) -> Mapping[str, Any]:
    extra = getattr(config, "extra", {})
    return extra if isinstance(extra, Mapping) else {}


def _uses_runtime_contract(config: Any) -> bool:
    return qwen_family_capabilities(config, required=False) is not None


def _agent_thinking_enabled(config: Any) -> bool:
    return _uses_runtime_contract(config) and bool(_config_extra(config).get("agent_thinking", False))


def _sampling_profile(config: Any, mode: str) -> dict[str, Any] | None:
    if mode not in _SAMPLING_MODES:
        raise ValueError(f"unsupported sampling mode: {mode!r}")
    profiles = _config_extra(config).get("sampling_profiles")
    if not isinstance(profiles, Mapping):
        return None
    raw = profiles.get(mode)
    if not isinstance(raw, Mapping):
        return None
    return {str(key): value for key, value in raw.items()}


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


def _qwen_agent_request(request: Any) -> bool:
    tool_choice = getattr(request, "tool_choice", None)
    if _forced_tool_choice(tool_choice):
        return False
    tools = getattr(request, "tools", ()) or ()
    if tools:
        return False
    return _assistant_has_agent_history(_request_messages(request))


def _qwen_sampling_mode(role: object, request: Any) -> str | None:
    """Map MMM request semantics onto registry-declared generation modes."""

    tools = getattr(request, "tools", ()) or ()
    if _forced_tool_choice(getattr(request, "tool_choice", None)):
        return None
    if tools:
        return "non_thinking"
    if getattr(request, "response_format", None) == "json" and not tools:
        return "non_thinking"
    normalized_role = str(role or "").strip().casefold()
    if normalized_role in {"coder", "coder_safe"}:
        return "precise_coding"
    return "general_thinking"


def _apply_family_payload_policy(
    payload: dict[str, Any],
    *,
    config: Any,
    role: object = "",
    request: Any,
) -> dict[str, Any]:
    """Apply registry-declared hybrid-thinking/tool-loop semantics."""

    capabilities = qwen_family_capabilities(config, required=False)
    if capabilities is None:
        return payload

    mode = _qwen_sampling_mode(role, request)
    agent_request = _qwen_agent_request(request)
    tools = getattr(request, "tools", ()) or ()
    json_page = getattr(request, "response_format", None) == "json" and not tools
    action_page = bool(tools) or json_page
    if not action_page and not _agent_thinking_enabled(config):
        return payload
    if mode is None and not agent_request and not action_page:
        return payload

    extra = _config_extra(config)
    if action_page:
        if json_page:
            payload["reasoning_effort"] = "none"
        else:
            payload.pop("reasoning_effort", None)
        payload["chat_template_kwargs"] = capabilities.action_template_kwargs()
    else:
        payload.pop("reasoning_effort", None)
        template_kwargs: dict[str, Any] = {"enable_thinking": True}
        if capabilities.preserve_thinking:
            template_kwargs["preserve_thinking"] = True
        if capabilities.reasoning_effort:
            reasoning_effort = str(
                extra.get("thinking_reasoning_effort", "xhigh")
            ).strip().casefold()
            if reasoning_effort not in {"xhigh", "medium", "low"}:
                raise ValueError(
                    "Qwen3.8 thinking_reasoning_effort must be xhigh, medium, or low"
                )
            template_kwargs["reasoning_effort"] = reasoning_effort
        payload["chat_template_kwargs"] = template_kwargs

    if mode is not None:
        sampling = _sampling_profile(config, mode)
        if sampling is not None:
            payload.pop("repetition_penalty", None)
            payload.update(sampling)
    return payload


def _strip_reasoning_history(request: GenerationRequest) -> GenerationRequest:
    changed = False
    messages: list[Mapping[str, Any]] = []
    for raw in _request_messages(request):
        message = dict(raw)
        if message.get("role") == "assistant":
            for key in ("reasoning_content", "reasoning"):
                if key in message:
                    message.pop(key, None)
                    changed = True
        messages.append(message)
    return replace(request, messages=tuple(messages)) if changed else request


def _canonical_arguments(value: Any) -> str:
    if isinstance(value, str):
        raw = value.strip() or "{}"
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
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
        calls.append({
            "id": str(raw.get("id") or ""),
            "name": str(function.get("name") or ""),
            "arguments": _canonical_arguments(function.get("arguments", "{}")),
        })
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
            "arguments": _canonical_arguments(call.raw_arguments or dict(call.arguments)),
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
    current = getattr(adapter, "_mmm_qwen_reasoning_traces", None)
    if isinstance(current, OrderedDict):
        return current
    store: OrderedDict[str, str] = OrderedDict()
    adapter._mmm_qwen_reasoning_traces = store
    return store


def _inject_reasoning_history(adapter: Any, request: GenerationRequest) -> GenerationRequest:
    messages = _request_messages(request)
    signed_messages = tuple((message, _message_signature(message)) for message in messages)
    signatures = [signature for _, signature in signed_messages if signature]
    tools = getattr(request, "tools", ()) or ()
    with _REASONING_TRACE_LOCK:
        store = _trace_store(adapter)
        if tools and getattr(request, "tool_choice", None) in (None, "auto") and not any(
            signature in store for signature in signatures
        ):
            return request

        changed = False
        rewritten: list[Mapping[str, Any]] = []
        for raw, signature in signed_messages:
            message = dict(raw)
            reasoning = store.get(signature) if signature else None
            if reasoning and message.get("role") == "assistant" and not str(
                message.get("reasoning_content") or ""
            ).strip():
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


def _runtime_receipt(autotune: Any) -> Mapping[str, Any]:
    receipt = getattr(autotune, "_MMM_LLAMA_RUNTIME_RECEIPT", None)
    if isinstance(receipt, Mapping):
        return receipt
    try:
        decoded = json.loads(os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", ""))
    except Exception:
        decoded = None
    return decoded if isinstance(decoded, Mapping) else {}


def _managed_runtime_alive(autotune: Any) -> bool:
    process = getattr(autotune, "_MANAGED_PROCESS", None)
    url = str(getattr(autotune, "_MANAGED_URL", "") or "").strip()
    return process is not None and process.poll() is None and bool(url)


def _clear_managed_runtime(autotune: Any) -> None:
    process = getattr(autotune, "_MANAGED_PROCESS", None)
    url = str(getattr(autotune, "_MANAGED_URL", "") or "").strip().rstrip("/")
    autotune._stop_server(process)
    autotune._MANAGED_PROCESS = None
    autotune._MANAGED_URL = None
    configured = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
    if url and configured == url:
        os.environ.pop("LLAMA_SERVER_URL", None)


def _tool_safe_variant(autotune: Any, receipt: Mapping[str, Any]) -> Any:
    return autotune.ServerVariant(
        name="tool-safe",
        spec_type="none",
        draft_n_max=0,
        ubatch=max(0, int(receipt.get("ubatch", 0) or 0)),
        parallel=max(1, int(receipt.get("slots", 1) or 1)),
        cache_reuse=max(0, int(receipt.get("cache_reuse", 0) or 0)),
        draft_p_min=0.0,
    )


def _ensure_tool_safe_runtime(config: Any, request: GenerationRequest) -> None:
    """Suspend speculative decoding for the complete contiguous Qwen tool phase."""

    global _TOOL_SAFE_RUNTIME_ACTIVE, _TOOL_SAFE_RUNTIME_KEY

    if not (getattr(request, "tools", ()) or ()):
        return
    if qwen_family_capabilities(config, required=False) is None:
        return

    from . import llama_server_autotune as autotune

    with _TOOL_RUNTIME_LOCK:
        lock = getattr(autotune, "_AUTOTUNE_LOCK", None)
        if lock is None:
            raise RuntimeError("Qwen tool-safe runtime requires the managed llama-server lock")
        with lock:
            if not _managed_runtime_alive(autotune):
                autotune.ensure_tuned_server(config, request)
            if not _managed_runtime_alive(autotune):
                return

            receipt = _runtime_receipt(autotune)
            spec_type = str(
                receipt.get(
                    "spec_type",
                    os.environ.get("MMM_LLAMA_ACTIVE_SPEC_TYPE", "none"),
                )
                or "none"
            ).strip().casefold()
            if spec_type == "none":
                return
            if _TOOL_SAFE_RUNTIME_ACTIVE:
                return

            binary = autotune._server_binary()
            if not binary:
                raise RuntimeError("native llama-server binary disappeared before tool phase")
            model_path = autotune._resolve_model_path(config)
            managed_key = getattr(autotune, "_MANAGED_KEY", None)
            safe_variant = _tool_safe_variant(autotune, receipt)

            _clear_managed_runtime(autotune)
            try:
                autotune._launch_selected(binary, model_path, config, safe_variant)
            except Exception:
                autotune._MANAGED_KEY = None
                if managed_key:
                    autotune._ATTEMPTED_KEYS.discard(managed_key)
                _TOOL_SAFE_RUNTIME_ACTIVE = False
                _TOOL_SAFE_RUNTIME_KEY = None
                raise

            autotune._MANAGED_KEY = managed_key
            _TOOL_SAFE_RUNTIME_ACTIVE = True
            _TOOL_SAFE_RUNTIME_KEY = managed_key
            print(
                "native llama-server: tool phase speculation suspended",
                f"spec={spec_type}->none",
                flush=True,
            )


def _restore_tool_runtime(config: Any, request: GenerationRequest) -> None:
    """Restore the cached autotuned runtime once the tool phase is actually over."""

    global _TOOL_SAFE_RUNTIME_ACTIVE, _TOOL_SAFE_RUNTIME_KEY

    if not _TOOL_SAFE_RUNTIME_ACTIVE:
        return

    from . import llama_server_autotune as autotune

    with _TOOL_RUNTIME_LOCK:
        lock = getattr(autotune, "_AUTOTUNE_LOCK", None)
        if lock is None:
            raise RuntimeError("Qwen tool-safe runtime requires the managed llama-server lock")
        with lock:
            if not _TOOL_SAFE_RUNTIME_ACTIVE:
                return
            managed_key = _TOOL_SAFE_RUNTIME_KEY or getattr(autotune, "_MANAGED_KEY", None)
            _clear_managed_runtime(autotune)
            autotune._MANAGED_KEY = None
            if managed_key:
                autotune._ATTEMPTED_KEYS.discard(managed_key)
            os.environ.pop("MMM_LLAMA_RUNTIME_RECEIPT", None)
            autotune._MMM_LLAMA_RUNTIME_RECEIPT = None
            _TOOL_SAFE_RUNTIME_ACTIVE = False
            _TOOL_SAFE_RUNTIME_KEY = None
            print(
                "native llama-server: tool phase ended; restoring autotuned speculation",
                flush=True,
            )
            autotune.ensure_tuned_server(config, request)


def install() -> None:
    """Install registry-declared hybrid-thinking agent policy exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import llama_server_hardware_policy
    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    current_payload = llama_server_hardware_policy._server_payload
    if not getattr(current_payload, "_mmm_qwen_family_agent_policy_v2", False):

        @wraps(current_payload)
        def server_payload(adapter: Any, request: Any) -> dict[str, Any]:
            payload = current_payload(adapter, request)
            config = getattr(adapter, "config", None)
            return _apply_family_payload_policy(
                payload,
                config=config,
                role=getattr(config, "role", ""),
                request=request,
            )

        server_payload._mmm_qwen_family_agent_policy_v2 = True  # type: ignore[attr-defined]
        server_payload._mmm_qwen_family_agent_policy = True  # type: ignore[attr-defined]
        llama_server_hardware_policy._server_payload = server_payload

    current_generate_turn = LlamaCppAdapter.generate_turn
    if not getattr(current_generate_turn, "_mmm_qwen_reasoning_history_v2", False):

        @wraps(current_generate_turn)
        def generate_turn(self: Any, request: GenerationRequest) -> GenerationResponse:
            config = getattr(self, "config", None)
            qwen_agent = _agent_thinking_enabled(config) and _qwen_agent_request(request)
            tool_page = bool(getattr(request, "tools", ()) or ())
            action_page = tool_page or getattr(request, "response_format", None) == "json"
            capabilities = qwen_family_capabilities(config, required=False)

            if capabilities is not None:
                if tool_page:
                    _ensure_tool_safe_runtime(config, request)
                elif _TOOL_SAFE_RUNTIME_ACTIVE:
                    _restore_tool_runtime(config, request)

            if capabilities is not None and action_page:
                prepared = _strip_reasoning_history(request)
            else:
                prepared = _inject_reasoning_history(self, request) if qwen_agent else request
            response = current_generate_turn(self, prepared)
            if qwen_agent:
                _remember_reasoning(self, response)

            if capabilities is not None and tool_page and not response.tool_calls:
                restore_request = replace(
                    prepared,
                    tools=(),
                    tool_validation_schemas=(),
                    tool_choice=None,
                    parallel_tool_calls=False,
                )
                _restore_tool_runtime(config, restore_request)
            return response

        generate_turn._mmm_qwen_reasoning_history_v2 = True  # type: ignore[attr-defined]
        generate_turn._mmm_qwen_reasoning_history = True  # type: ignore[attr-defined]
        generate_turn._mmm_qwen_tool_safe_speculation = True  # type: ignore[attr-defined]
        LlamaCppAdapter.generate_turn = generate_turn

    _INSTALLED = True


_qwen36_agent_request = _qwen_agent_request
_qwen36_sampling_mode = _qwen_sampling_mode

__all__ = [
    "_ensure_tool_safe_runtime",
    "_restore_tool_runtime",
    "_strip_reasoning_history",
    "install",
]
