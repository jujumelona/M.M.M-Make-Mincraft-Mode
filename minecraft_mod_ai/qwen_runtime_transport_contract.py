from __future__ import annotations

"""Validate Qwen native tool transport and MTP runtime compatibility.

MMM exposes first-party MCP tools, reviewed external MCP tools, and temporary Skill
execution through the same OpenAI/Jinja function-tool transport. The staged
llama.cpp tuner already keeps each baseline/MTP candidate server alive for a warm-up
and an exact-output decode probe. This contract reuses that live server for one tiny,
deterministic function call before a candidate may participate in selection.

Qwen MTP is also kept single-slot. Current llama.cpp/Unsloth guidance does not support
``-np > 1`` with MTP, so the parallel refinement stage is skipped only when native
``draft-mtp`` won, and the final text launch is defensively pinned to one slot. Media
launches may replace MTP with baseline outside this contract and therefore retain the
normal baseline parallel policy.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Any, Iterator, Mapping

from .qwen_model_profiles import qwen_family

_TOOL_NAME = "mmm_transport_probe"
_TOOL_VALUE = 7
_BENCHMARK_MARKER = "_mmm_qwen_tool_calibration_benchmark_v1"
_RUN_VARIANT_MARKER = "_mmm_qwen_tool_calibration_context_v2"
_PROBE_MARKER = "_mmm_qwen_tool_calibration_probe_v2"
_PARALLEL_STAGE_MARKER = "_mmm_qwen_mtp_single_slot_stage_v1"
_LAUNCH_MARKER = "_mmm_qwen_mtp_single_slot_launch_v1"
_FINGERPRINT_MARKER = "_mmm_qwen_mtp_single_slot_fingerprint_v1"
_ACTIVE_BENCHMARK_CONFIG: ContextVar[Any | None] = ContextVar(
    "mmm_qwen_transport_benchmark_config",
    default=None,
)
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


def _mtp_variant(variant: Any) -> bool:
    return str(getattr(variant, "spec_type", "none")) == "draft-mtp"


def _single_slot_variant(variant: Any) -> Any:
    if max(1, int(getattr(variant, "parallel", 1) or 1)) == 1:
        return variant
    root_name = str(getattr(variant, "name", "mtp")).split("|p", 1)[0]
    return replace(variant, name=root_name, parallel=1)


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


@contextmanager
def _single_slot_launch_scope() -> Iterator[None]:
    """Make the inner runtime launcher observe the only supported MTP slot count."""

    previous = os.environ.get("MMM_LLAMA_PARALLEL")
    os.environ["MMM_LLAMA_PARALLEL"] = "1"
    try:
        yield
    finally:
        _restore_env("MMM_LLAMA_PARALLEL", previous)


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


def _install_mtp_single_slot_policy(autotune: Any) -> None:
    """Prevent unsupported Qwen MTP + multi-slot tuning and final launches."""

    from . import llama_server_runtime_tuning as runtime_tuning

    current_stage = runtime_tuning._run_parallel_stage
    if not getattr(current_stage, _PARALLEL_STAGE_MARKER, False):

        @wraps(current_stage)
        def qwen_mtp_parallel_stage(*args: Any, **kwargs: Any) -> Any:
            config = kwargs.get("config")
            selected = kwargs.get("selected")
            if _family(config) is not None and _mtp_variant(selected):
                return _single_slot_variant(selected), None, None, ()
            return current_stage(*args, **kwargs)

        setattr(qwen_mtp_parallel_stage, _PARALLEL_STAGE_MARKER, True)
        runtime_tuning._run_parallel_stage = qwen_mtp_parallel_stage

    current_launch = autotune._launch_selected
    if not getattr(current_launch, _LAUNCH_MARKER, False):

        @wraps(current_launch)
        def qwen_mtp_launch(
            binary: str,
            model_path: str,
            config: Any,
            selected: Any,
        ) -> str:
            if _family(config) is None or not _mtp_variant(selected):
                return current_launch(binary, model_path, config, selected)

            # Multimodal owns the MTP->baseline conversion for affected production
            # profiles. Do not pin the outer scope to p1 before that conversion, or
            # the inner baseline launch loses its otherwise-valid parallel policy.
            if os.environ.get("MMM_LLAMA_MULTIMODAL_ACTIVE", "").strip() == "1":
                from .llama_multimodal_contract import _requires_media_baseline

                if _requires_media_baseline(config):
                    return current_launch(binary, model_path, config, selected)

            selected = _single_slot_variant(selected)
            with _single_slot_launch_scope():
                return current_launch(binary, model_path, config, selected)

        setattr(qwen_mtp_launch, _LAUNCH_MARKER, True)
        autotune._launch_selected = qwen_mtp_launch

    current_fingerprint = autotune._fingerprint
    if not getattr(current_fingerprint, _FINGERPRINT_MARKER, False):

        @wraps(current_fingerprint)
        def qwen_mtp_fingerprint(config: Any, binary: str, model_path: str) -> str:
            base = str(current_fingerprint(config, binary, model_path))
            if _family(config) is None:
                return base
            payload = {
                "base": base,
                "qwen_mtp_parallel": "single-slot-v1",
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()

        setattr(qwen_mtp_fingerprint, _FINGERPRINT_MARKER, True)
        autotune._fingerprint = qwen_mtp_fingerprint


def _install_tool_equivalence_policy(autotune: Any) -> None:
    """Reject a Qwen speculation candidate if its native tool transport is invalid."""

    current_benchmark = getattr(autotune, "_benchmark", None)
    if not callable(current_benchmark):
        raise RuntimeError("Qwen transport guard requires staged llama runtime tuning")
    if not getattr(current_benchmark, _BENCHMARK_MARKER, False):

        @wraps(current_benchmark)
        def qwen_calibrated_benchmark(
            binary: str,
            model_path: str,
            config: Any,
            request: Any,
            fingerprint: str,
        ) -> Any:
            if _family(config) is None:
                return current_benchmark(binary, model_path, config, request, fingerprint)
            token = _ACTIVE_BENCHMARK_CONFIG.set(config)
            try:
                return current_benchmark(binary, model_path, config, request, fingerprint)
            finally:
                _ACTIVE_BENCHMARK_CONFIG.reset(token)

        setattr(qwen_calibrated_benchmark, _BENCHMARK_MARKER, True)
        autotune._benchmark = qwen_calibrated_benchmark

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
        if max_tokens != 1 or not _initial_calibration_variant(variant):
            return probe

        active_run = _ACTIVE_CALIBRATION.get()
        benchmark_config = _ACTIVE_BENCHMARK_CONFIG.get()
        config = active_run[0] if active_run is not None else benchmark_config
        if config is None or _family(config) is None:
            return probe
        if active_run is not None and active_run[1] != variant:
            return probe

        valid, error = _tool_probe(base_url, autotune)
        if not valid:
            raise RuntimeError(f"Qwen native tool transport calibration failed: {error}")
        return probe

    setattr(qwen_tool_probe, _PROBE_MARKER, True)
    autotune._probe_server = qwen_tool_probe


def install() -> None:
    from . import llama_server_autotune

    _install_mtp_single_slot_policy(llama_server_autotune)
    _install_tool_equivalence_policy(llama_server_autotune)


__all__ = [
    "_initial_calibration_variant",
    "_install_mtp_single_slot_policy",
    "_install_tool_equivalence_policy",
    "_mtp_variant",
    "_single_slot_variant",
    "_tool_call_signature",
    "install",
]
