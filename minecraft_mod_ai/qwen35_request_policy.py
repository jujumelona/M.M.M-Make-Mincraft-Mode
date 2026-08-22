from __future__ import annotations

"""Registry-driven request semantics for native llama.cpp runtimes.

The transport stays model-agnostic. Profiles opt in through AdapterConfig.extra
instead of being identified from repository names or GGUF filenames.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator, Literal, Mapping

from .qwen_family_capabilities import qwen_family_capabilities, qwen_family_name

SamplingMode = Literal["general_thinking", "precise_coding", "non_thinking"]

_PAYLOAD_MARKER = "_mmm_qwen35_request_policy_v4"
_BASE_ARGS_MARKER = "_mmm_qwen35_benchmark_reasoning_off_v3"
_BENCHMARK_MARKER = "_mmm_qwen35_decode_benchmark_scope_v3"
_VARIANT_MARKER = "_mmm_qwen35_tuning_variant_scope_v2"
_FINGERPRINT_MARKER = "_mmm_qwen35_request_policy_fingerprint_v4"
_BENCHMARK_ENV = "MMM_QWEN35_DECODE_BENCHMARK"
_POLICY_NAME = "task_aware_sampling"


def _extra(config: Any) -> Mapping[str, Any]:
    extra = getattr(config, "extra", {})
    return extra if isinstance(extra, Mapping) else {}


def _policy_enabled(config: Any) -> bool:
    return (
        qwen_family_name(config) == "qwen3.5"
        and str(_extra(config).get("request_policy", "")).strip().casefold()
        == _POLICY_NAME
    )


def _is_qwen35(config: Any) -> bool:
    """Compatibility name for the legacy hook; activation is registry-driven."""

    return _policy_enabled(config)


def _request_sampling_mode(config: Any, request: Any) -> SamplingMode:
    tools = getattr(request, "tools", ()) or ()
    # Tool-capable turns are action pages: the causal/planning layer has already
    # selected the visible action surface and this decode only has to materialize a
    # bounded, schema-validated call. Qwen3.5 thinks by default, so treating these as
    # precise-coding pages can spend the whole action allowance inside ``<think>``
    # before the tool envelope closes.
    if tools:
        return "non_thinking"
    structured_fill = (
        getattr(request, "response_format", None) == "json" and not tools
    )
    if structured_fill:
        return "non_thinking"

    role = str(getattr(config, "role", "")).strip().casefold()
    if role in {"coder", "coder_safe"}:
        return "precise_coding"
    return "general_thinking"


def _request_defaults(config: Any, request: Any) -> dict[str, Any]:
    profiles = _extra(config).get("sampling_profiles")
    if not isinstance(profiles, Mapping):
        return {}
    selected = profiles.get(_request_sampling_mode(config, request))
    if not isinstance(selected, Mapping):
        return {}
    return dict(selected)


def _install_payload_policy(hardware_policy: Any) -> None:
    current = hardware_policy._server_payload
    if getattr(current, _PAYLOAD_MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = current(adapter, request)
        config = getattr(adapter, "config", None)
        if not _policy_enabled(config):
            return result

        # A named/required one-tool request is a transport control turn, not a normal
        # agent sampling page. Reassert the shared wire invariant after this outer
        # profile wrapper so Qwen3.5 defaults cannot turn forced recovery stochastic.
        from .llama_server_hardware_policy import _enforce_required_tool_sampling

        if result.get("tool_choice") == "required":
            return _enforce_required_tool_sampling(result)

        mode = _request_sampling_mode(config, request)
        defaults = _request_defaults(config, request)
        result.pop("chat_template_kwargs", None)
        result.pop("thinking_budget_tokens", None)
        if "reasoning_effort" not in defaults:
            result.pop("reasoning_effort", None)
        result.update(defaults)
        if mode == "non_thinking":
            capabilities = qwen_family_capabilities(config, required=True)
            assert capabilities is not None
            # Qwen3.5's published hard switch is the template kwarg. Do not rely on
            # llama.cpp's OpenAI-specific reasoning_effort alias as a model contract.
            result.pop("reasoning_effort", None)
            result["chat_template_kwargs"] = capabilities.action_template_kwargs()
        return result

    setattr(payload, _PAYLOAD_MARKER, True)
    hardware_policy._server_payload = payload


def _set_reasoning_off(args: list[str]) -> None:
    for name in ("--reasoning", "-rea"):
        while name in args:
            index = args.index(name)
            del args[index]
            if index < len(args):
                del args[index]
    args.extend(["--reasoning", "off"])


def _install_benchmark_base_args(autotune: Any) -> None:
    current = autotune._base_args
    if getattr(current, _BASE_ARGS_MARKER, False):
        return

    @wraps(current)
    def base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
        args = list(current(binary, model_path, config, port))
        if (
            _policy_enabled(config)
            and os.environ.get(_BENCHMARK_ENV, "").strip() == "1"
        ):
            _set_reasoning_off(args)
        return args

    setattr(base_args, _BASE_ARGS_MARKER, True)
    autotune._base_args = base_args


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


@contextmanager
def _benchmark_scope(config: Any) -> Iterator[None]:
    if not _policy_enabled(config):
        yield
        return
    previous = os.environ.get(_BENCHMARK_ENV)
    os.environ[_BENCHMARK_ENV] = "1"
    try:
        yield
    finally:
        _restore_env(_BENCHMARK_ENV, previous)


def _install_benchmark_scope(autotune: Any) -> None:
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
        with _benchmark_scope(config):
            return current(binary, model_path, config, request, fingerprint)

    setattr(benchmark, _BENCHMARK_MARKER, True)
    autotune._benchmark = benchmark


def _install_tuning_variant_scope(autotune: Any) -> None:
    current = getattr(autotune, "_mmm_run_tuning_variant", None)
    if not callable(current) or getattr(current, _VARIANT_MARKER, False):
        return

    @wraps(current)
    def run_variant(
        binary: str,
        model_path: str,
        config: Any,
        benchmark_request: Any,
        variant: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with _benchmark_scope(config):
            return current(
                binary,
                model_path,
                config,
                benchmark_request,
                variant,
                *args,
                **kwargs,
            )

    setattr(run_variant, _VARIANT_MARKER, True)
    autotune._mmm_run_tuning_variant = run_variant


def _install_fingerprint(autotune: Any) -> None:
    current = autotune._fingerprint
    if getattr(current, _FINGERPRINT_MARKER, False):
        return

    @wraps(current)
    def fingerprint(config: Any, binary: str, model_path: str) -> str:
        base = str(current(config, binary, model_path))
        if not _policy_enabled(config):
            return base
        payload = {
            "base": base,
            "request_policy": _POLICY_NAME,
            "benchmark_reasoning": "off",
            "tool_action_mode": "non-thinking-v1",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    setattr(fingerprint, _FINGERPRINT_MARKER, True)
    autotune._fingerprint = fingerprint


def install(autotune: Any, hardware_policy: Any) -> None:
    """Install the registry-selected task-aware request policy."""

    _install_payload_policy(hardware_policy)
    _install_benchmark_base_args(autotune)
    _install_benchmark_scope(autotune)
    _install_tuning_variant_scope(autotune)
    _install_fingerprint(autotune)


__all__ = [
    "_is_qwen35",
    "_request_defaults",
    "install",
]
