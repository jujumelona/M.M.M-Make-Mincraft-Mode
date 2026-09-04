from __future__ import annotations

"""Install strict no-inference assistant-prefill calibration for llama.cpp.

Assistant-prefill calibration is template introspection, not generation.  Sending a
``max_tokens=0`` request to ``/chat/completions`` still enters the inference path on
some llama.cpp server versions and can produce model tokens.  That is both expensive
and semantically wrong for a host-owned calibration step.

The live llama.cpp server exposes ``/apply-template`` specifically to render the chat
template without inference.  This contract replaces the adapter's calibration helper
with that endpoint and derives the exact suffix following a unique sentinel.  Any
ambiguous/malformed rendering still fails closed; there is no non-fatal fallback to a
model completion.
"""

from collections.abc import Mapping
from typing import Any

_MARKER = "_mmm_apply_template_prefill_calibration_v1"
_GENERATION_ONLY_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stream",
        "stop",
        "n",
    }
)


def _template_payload(llama_cpp_module: Any, original: Mapping[str, Any]) -> dict[str, Any]:
    factory = getattr(llama_cpp_module, "_assistant_prefill_calibration_payload", None)
    if not callable(factory):
        raise RuntimeError("llama.cpp adapter lost assistant-prefill calibration payload builder")
    payload = dict(factory(original))
    for key in _GENERATION_ONLY_KEYS:
        payload.pop(key, None)
    return payload


def _post_apply_template(
    llama_cpp_module: Any,
    server_url: str,
    payload: Mapping[str, Any],
) -> Any:
    endpoint = f"{server_url.rstrip('/')}/apply-template"
    positive_timeout = getattr(llama_cpp_module, "_positive_env_float", None)
    default_timeout = float(
        getattr(llama_cpp_module, "_DEFAULT_COMPLETION_TIMEOUT_SECONDS", 120.0)
    )
    read_timeout = (
        positive_timeout("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS", default_timeout)
        if callable(positive_timeout)
        else default_timeout
    )
    httpx_module = getattr(llama_cpp_module, "httpx", None)
    if httpx_module is None:
        raise RuntimeError("llama.cpp adapter has no HTTP client module")
    timeout = httpx_module.Timeout(
        connect=30.0,
        read=read_timeout,
        write=30.0,
        pool=30.0,
    )

    try:
        default_post = getattr(llama_cpp_module, "_DEFAULT_HTTPX_POST", None)
        if getattr(httpx_module, "post", None) is not default_post:
            return httpx_module.post(endpoint, json=dict(payload), timeout=timeout)
        from .llama_stream_efficiency_contract import _client

        return _client(server_url).post(endpoint, json=dict(payload), timeout=timeout)
    except httpx_module.TimeoutException as exc:
        raise RuntimeError(
            "native llama-server apply-template made no readable progress for "
            f"{read_timeout:.0f}s"
        ) from exc


def _calibrator(llama_cpp_module: Any):
    sentinel = str(
        getattr(llama_cpp_module, "_PREFILL_CALIBRATION_SENTINEL", "") or ""
    )
    max_bytes = int(
        getattr(llama_cpp_module, "_MAX_PREFILL_TEMPLATE_BYTES", 512) or 512
    )
    if not sentinel:
        raise RuntimeError("llama.cpp adapter has no assistant-prefill calibration sentinel")

    def calibrate(server_url: str, original: Mapping[str, Any]) -> str:
        response = _post_apply_template(
            llama_cpp_module,
            server_url,
            _template_payload(llama_cpp_module, original),
        )
        if int(getattr(response, "status_code", 500)) >= 400:
            body_fn = getattr(llama_cpp_module, "_bounded_response_body", None)
            body = body_fn(response) if callable(body_fn) else ""
            raise RuntimeError(
                "assistant-prefill apply-template request was rejected"
                + (f": {body}" if body else "")
            )
        data = response.json()
        if not isinstance(data, Mapping):
            raise TypeError("assistant-prefill apply-template returned invalid JSON")
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("assistant-prefill apply-template returned no rendered prompt")
        if prompt.count(sentinel) != 1:
            raise RuntimeError(
                "assistant-prefill apply-template sentinel is missing or ambiguous"
            )
        suffix = prompt.split(sentinel, 1)[1]
        if not suffix:
            raise RuntimeError("assistant-prefill template suffix is empty or ambiguous")
        encoded = suffix.encode("utf-8")
        if len(encoded) > max_bytes:
            raise RuntimeError("assistant-prefill template suffix is unexpectedly large")
        return suffix

    setattr(calibrate, _MARKER, True)
    return calibrate


def install(llama_cpp_module: Any) -> None:
    current = getattr(
        llama_cpp_module,
        "_calibrate_assistant_prefill_generation_prompt",
        None,
    )
    if not callable(current):
        raise RuntimeError("llama.cpp adapter lost assistant-prefill calibrator")
    if getattr(current, _MARKER, False):
        return
    llama_cpp_module._calibrate_assistant_prefill_generation_prompt = _calibrator(
        llama_cpp_module
    )


__all__ = ["install"]
