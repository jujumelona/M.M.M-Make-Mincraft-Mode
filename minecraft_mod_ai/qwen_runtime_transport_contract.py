from __future__ import annotations

"""Validate Qwen speculative decoding across the native tool-call boundary.

MMM exposes first-party MCP tools, reviewed external MCP tools, and temporary Skill
execution through the same OpenAI/Jinja function-tool transport. The staged
llama.cpp tuner already keeps each baseline/MTP candidate server alive for a warm-up
and an exact-output decode probe. This contract reuses that live server for one tiny,
deterministic function call before a candidate may participate in selection.

No second model load is performed, and later ubatch/parallel/cache tuning keeps its
existing exact-output contract unchanged. A speculative candidate that cannot emit
the canonical tool call is simply treated as a failed tuning variant.
"""

import hashlib
import json
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping

from .qwen_model_profiles import qwen_family

_TOOL_NAME = "mmm_transport_probe"
_TOOL_VALUE = 7
_RUN_VARIANT_MARKER = "_mmm_qwen_tool_calibration_context_v2"
_PROBE_MARKER = "_mmm_qwen_tool_calibration_probe_v2"
_ACTIVE_CALIBRATION: ContextVar[tuple[Any, Any] | None] = ContextVar(
    "mmm_qwen_transport_calibration",
    default=None,
)


def _config_identity(config: Any) -> tuple[str, str]:
    model_id = str(getattr(config, "model_id", ""))
    extra = getattr(config, "extra", {})
    filename = str(extra.get("gguf_filename", "")) if isinstance(extra, Mapping) else ""
    return model_id, filename


def _family(config: Any) -> str | None:
    model_id, filename = _config_identity(config)
    return qwen_family(model_id, filename)


def _canonical_arguments(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return value
    raw = value.strip() or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _tool_call_signature(response: Mapping[str, Any]) -> str:
    """Return a stable signature only for the exact expected semantic tool call."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return ""
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) != 1:
        return ""
    call = raw_calls[0]
    if not isinstance(call, Mapping):
        return ""
    function = call.get("function")
    if not isinstance(function, Mapping):
        return ""

    name = str(function.get("name", "")).strip()
    arguments = _canonical_arguments(function.get("arguments", "{}"))
    if name != _TOOL_NAME or arguments != {"value": _TOOL_VALUE}:
        return ""

    canonical = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tool_probe(base_url: str, autotune: Any) -> tuple[bool, str]:
    """Exercise llama.cpp's native function-tool parser on the already-live server."""

    import httpx

    payload = {
        "model": "local",
        "messages": [
            {
                "role": "system",
                "content": "Use the requested function exactly once. Do not answer in text.",
            },
            {
                "role": "user",
                "content": f"Call {_TOOL_NAME} with value {_TOOL_VALUE}.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": "Deterministic transport correctness probe.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 48,
        "temperature": 0.0,
        "seed": 1234,
        "cache_prompt": False,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=autotune._env_int("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", 300),
        )
        response.raise_for_status()
        data = response.json()
        signature = _tool_call_signature(data) if isinstance(data, Mapping) else ""
        if not signature:
            return False, "native tool probe returned no valid canonical tool call"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _initial_calibration_variant(variant: Any) -> bool:
    """Validate only the baseline/speculation search, not neutral tuning refinements."""

    return bool(
        int(getattr(variant, "ubatch", 0) or 0) == 0
        and int(getattr(variant, "parallel", 1) or 1) == 1
        and int(getattr(variant, "cache_reuse", 0) or 0) == 0
    )


def _install_tool_equivalence_policy(autotune: Any) -> None:
    """Reject a Qwen speculation candidate if its native tool transport is invalid."""

    current_run = getattr(autotune, "_mmm_run_tuning_variant", None)
    if not callable(current_run):
        raise RuntimeError("Qwen transport guard requires staged llama runtime tuning")

    if not getattr(current_run, _RUN_VARIANT_MARKER, False):

        @wraps(current_run)
        def qwen_calibrated_run_variant(
            binary: str,
            model_path: str,
            config: Any,
            benchmark_request: Any,
            variant: Any,
            **kwargs: Any,
        ) -> Any:
            token = _ACTIVE_CALIBRATION.set((config, variant))
            try:
                return current_run(
                    binary,
                    model_path,
                    config,
                    benchmark_request,
                    variant,
                    **kwargs,
                )
            finally:
                _ACTIVE_CALIBRATION.reset(token)

        setattr(qwen_calibrated_run_variant, _RUN_VARIANT_MARKER, True)
        autotune._mmm_run_tuning_variant = qwen_calibrated_run_variant

    current_probe = autotune._probe_server
    if getattr(current_probe, _PROBE_MARKER, False):
        return

    @wraps(current_probe)
    def qwen_tool_probe(
        base_url: str,
        request: Any,
        *,
        max_tokens: int,
        variant: Any,
    ) -> Any:
        probe = current_probe(
            base_url,
            request,
            max_tokens=max_tokens,
            variant=variant,
        )
        active = _ACTIVE_CALIBRATION.get()
        if active is None or max_tokens != 1 or not _initial_calibration_variant(variant):
            return probe

        config, active_variant = active
        if active_variant != variant or _family(config) is None:
            return probe

        valid, error = _tool_probe(base_url, autotune)
        if not valid:
            raise RuntimeError(f"Qwen native tool transport calibration failed: {error}")
        return probe

    setattr(qwen_tool_probe, _PROBE_MARKER, True)
    autotune._probe_server = qwen_tool_probe


def install() -> None:
    from . import llama_server_autotune

    _install_tool_equivalence_policy(llama_server_autotune)


__all__ = [
    "_initial_calibration_variant",
    "_install_tool_equivalence_policy",
    "_tool_call_signature",
    "install",
]
