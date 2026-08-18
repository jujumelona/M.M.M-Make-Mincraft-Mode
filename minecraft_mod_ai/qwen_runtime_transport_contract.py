from __future__ import annotations

"""Qwen native-server transport guards for context and speculative tool calls.

The production Qwen profiles share llama.cpp's native OpenAI/Jinja tool transport.
This contract keeps the deployment context explicit for Qwen3.6 and admits a
speculative winner only after the same model produces the same canonical function
call on both baseline and speculative servers. MCP-backed skills are exposed to the
model through that same function-tool transport, so the probe protects the parser
boundary without coupling inference code to any particular MCP provider.
"""

import hashlib
import json
import os
from functools import wraps
from typing import Any, Mapping

from .qwen_model_profiles import qwen_family

_BASE_ARGS_MARKER = "_mmm_qwen_context_contract_v1"
_FINGERPRINT_MARKER = "_mmm_qwen_context_fingerprint_v1"
_BENCHMARK_MARKER = "_mmm_qwen_speculative_tool_equivalence_v1"
_TOOL_NAME = "mmm_transport_probe"
_TOOL_VALUE = 7


def _config_identity(config: Any) -> tuple[str, str]:
    model_id = str(getattr(config, "model_id", ""))
    extra = getattr(config, "extra", {})
    filename = str(extra.get("gguf_filename", "")) if isinstance(extra, Mapping) else ""
    return model_id, filename


def _family(config: Any) -> str | None:
    model_id, filename = _config_identity(config)
    return qwen_family(model_id, filename)


def _qwen36_context(config: Any) -> int:
    """Resolve the Qwen3.6 server context, preserving an explicit operator override."""

    raw = os.environ.get("MMM_LLAMA_SERVER_CTX", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if value >= 0:
            return value
    try:
        configured = int(getattr(config, "max_context", 0) or 0)
    except (TypeError, ValueError):
        configured = 0
    return max(0, configured)


def _actual_qwen_context(config: Any) -> int:
    """Return the context that the installed Qwen launch stack will actually use."""

    if _family(config) == "qwen3.5":
        try:
            from .qwen35_mtp_hotpath_contract import _context_size, _is_qwen35_mtp

            if _is_qwen35_mtp(config):
                return int(_context_size(config))
        except (ImportError, TypeError, ValueError):
            pass
    return _qwen36_context(config)


def _set_option(args: list[str], names: tuple[str, ...], value: str) -> None:
    for name in names:
        if name not in args:
            continue
        index = args.index(name)
        if index + 1 < len(args):
            args[index + 1] = value
            return
    args.extend([names[0], value])


def _install_context_policy(autotune: Any) -> None:
    current_base = autotune._base_args
    if not getattr(current_base, _BASE_ARGS_MARKER, False):

        @wraps(current_base)
        def qwen_context_base_args(
            binary: str,
            model_path: str,
            config: Any,
            port: int,
        ) -> list[str]:
            args = list(current_base(binary, model_path, config, port))
            if _family(config) == "qwen3.6":
                _set_option(
                    args,
                    ("--ctx-size", "-c"),
                    str(_qwen36_context(config)),
                )
            return args

        setattr(qwen_context_base_args, _BASE_ARGS_MARKER, True)
        autotune._base_args = qwen_context_base_args

    current_fingerprint = autotune._fingerprint
    if not getattr(current_fingerprint, _FINGERPRINT_MARKER, False):

        @wraps(current_fingerprint)
        def qwen_context_fingerprint(config: Any, binary: str, model_path: str) -> str:
            base = str(current_fingerprint(config, binary, model_path))
            if _family(config) is None:
                return base
            payload = {
                "base": base,
                "qwen_context": _actual_qwen_context(config),
                "transport_contract": "v1",
            }
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()

        setattr(qwen_context_fingerprint, _FINGERPRINT_MARKER, True)
        autotune._fingerprint = qwen_context_fingerprint


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


def _tool_probe(base_url: str, autotune: Any) -> tuple[bool, str, str]:
    """Exercise llama.cpp's native function-tool parser with a tiny deterministic call."""

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
            return False, "", "native tool probe returned no valid canonical tool call"
        return True, signature, ""
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def _probe_variant_tools(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    variant: Any,
) -> tuple[bool, str, str]:
    port = autotune._free_port(autotune._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910))
    process = None
    try:
        process = autotune._start_server(binary, model_path, config, variant, port)
        url = autotune._wait_ready(process, port)
        return _tool_probe(url, autotune)
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    finally:
        autotune._stop_server(process)


def _baseline_variant(decision: Any) -> Any | None:
    for probe in getattr(decision, "probes", ()) or ():
        variant = getattr(probe, "variant", None)
        if str(getattr(variant, "name", "")) == "baseline":
            return variant
    return None


def _verify_speculative_tool_equivalence(
    autotune: Any,
    binary: str,
    model_path: str,
    config: Any,
    decision: Any,
) -> tuple[bool, str]:
    baseline = _baseline_variant(decision)
    selected = getattr(decision, "selected", None)
    if baseline is None or selected is None:
        return False, "missing baseline or selected variant"
    base_ok, base_signature, base_error = _probe_variant_tools(
        autotune, binary, model_path, config, baseline
    )
    if not base_ok:
        return False, f"baseline tool probe failed: {base_error}"
    selected_ok, selected_signature, selected_error = _probe_variant_tools(
        autotune, binary, model_path, config, selected
    )
    if not selected_ok:
        return False, f"speculative tool probe failed: {selected_error}"
    if selected_signature != base_signature:
        return False, "speculative native tool call differs from baseline"
    return True, ""


def _fallback_to_baseline(autotune: Any, decision: Any) -> Any:
    baseline = _baseline_variant(decision)
    if baseline is None:
        return decision
    baseline_tps = float(getattr(decision, "baseline_tps", 0.0) or 0.0)
    return autotune.AutotuneDecision(
        fingerprint=str(getattr(decision, "fingerprint", "")),
        selected=baseline,
        baseline_tps=baseline_tps,
        selected_tps=baseline_tps,
        speedup=1.0,
        probes=tuple(getattr(decision, "probes", ()) or ()),
    )


def _install_tool_equivalence_policy(autotune: Any) -> None:
    current = autotune._benchmark
    if getattr(current, _BENCHMARK_MARKER, False):
        return

    @wraps(current)
    def benchmark(
        binary: str,
        model_path: str,
        config: Any,
        request: Any,
        fingerprint: str,
    ) -> Any:
        decision = current(binary, model_path, config, request, fingerprint)
        if decision is None or _family(config) is None:
            return decision
        selected = getattr(decision, "selected", None)
        if selected is None or str(getattr(selected, "spec_type", "none")) == "none":
            return decision
        valid, reason = _verify_speculative_tool_equivalence(
            autotune,
            binary,
            model_path,
            config,
            decision,
        )
        if valid:
            return decision
        print(
            "llama autotune: rejecting speculative winner;",
            reason,
            flush=True,
        )
        return _fallback_to_baseline(autotune, decision)

    setattr(benchmark, _BENCHMARK_MARKER, True)
    autotune._benchmark = benchmark


def install() -> None:
    from . import llama_server_autotune

    _install_context_policy(llama_server_autotune)
    _install_tool_equivalence_policy(llama_server_autotune)


__all__ = [
    "_actual_qwen_context",
    "_install_context_policy",
    "_install_tool_equivalence_policy",
    "_qwen36_context",
    "_tool_call_signature",
    "install",
]
