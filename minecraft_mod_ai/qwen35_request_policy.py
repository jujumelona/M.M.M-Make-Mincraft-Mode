from __future__ import annotations

"""Qwen3.5-9B request semantics for the native llama.cpp runtime.

Generic llama-server transport stays model-agnostic. This module owns the small set
of Qwen3.5-specific choices that materially change inference quality or benchmark
cost: task-aware sampling, per-request thinking for host-owned structured fills, and
reasoning-free speculative-decode benchmarking.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator, Mapping

from .qwen_model_profiles import (
    QwenSamplingMode,
    is_qwen35_9b,
    qwen_sampling_profile,
)

_PAYLOAD_MARKER = "_mmm_qwen35_request_policy_v2"
_BASE_ARGS_MARKER = "_mmm_qwen35_benchmark_reasoning_off_v2"
_BENCHMARK_MARKER = "_mmm_qwen35_decode_benchmark_scope_v2"
_VARIANT_MARKER = "_mmm_qwen35_tuning_variant_scope_v1"
_FINGERPRINT_MARKER = "_mmm_qwen35_request_policy_fingerprint_v2"
_BENCHMARK_ENV = "MMM_QWEN35_DECODE_BENCHMARK"


def _model_identity(config: Any) -> tuple[object, object]:
    model_id = getattr(config, "model_id", "")
    extra = getattr(config, "extra", {})
    filename = extra.get("gguf_filename", "") if isinstance(extra, Mapping) else ""
    return model_id, filename


def _is_qwen35(config: Any) -> bool:
    model_id, filename = _model_identity(config)
    return is_qwen35_9b(model_id, filename)


def _request_sampling_mode(config: Any, request: Any) -> QwenSamplingMode:
    tools = getattr(request, "tools", ()) or ()
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
    model_id, filename = _model_identity(config)
    return qwen_sampling_profile(
        model_id,
        filename,
        mode=_request_sampling_mode(config, request),
    ) or {}


def _install_payload_policy(hardware_policy: Any) -> None:
    current = hardware_policy._server_payload
    if getattr(current, _PAYLOAD_MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = current(adapter, request)
        config = getattr(adapter, "config", None)
        if not _is_qwen35(config):
            return result

        defaults = _request_defaults(config, request)
        # Current llama.cpp accepts reasoning_effort="none" as the native
        # per-request disable switch. Do not also send the legacy template kwarg.
        result.pop("chat_template_kwargs", None)
        result.pop("thinking_budget_tokens", None)
        if "reasoning_effort" not in defaults:
            result.pop("reasoning_effort", None)
        result.update(defaults)
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
            _is_qwen35(config)
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
    if not _is_qwen35(config):
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
        if not _is_qwen35(config):
            return base
        payload = {
            "base": base,
            "qwen35_request_policy": "v2",
            "benchmark_reasoning": "off",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    setattr(fingerprint, _FINGERPRINT_MARKER, True)
    autotune._fingerprint = fingerprint


def install(autotune: Any, hardware_policy: Any) -> None:
    """Install Qwen3.5 behavior after generic and output-budget wrappers."""

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
