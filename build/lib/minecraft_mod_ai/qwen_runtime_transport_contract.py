from __future__ import annotations

"""Validate registry-declared native tool transport and MTP runtime compatibility.

MMM exposes first-party MCP tools, reviewed external MCP tools, and temporary Skill
execution through the same OpenAI/Jinja function-tool transport. The staged
llama.cpp tuner already keeps each baseline/MTP candidate server alive for a warm-up
and an exact-output decode probe. This contract reuses that live server for one tiny,
deterministic function call before a candidate may participate in selection.

MTP is kept single-slot for runtime contracts that opt into this guard. Default draft
widths are read from model_registry metadata and explicit operator overrides remain
authoritative. No model id, version, parameter count, filename, context length, or
quantization is interpreted by this module.
"""

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace
from typing import Any

from .qwen_family_capabilities import qwen_family_capabilities

_TOOL_NAME = "mmm_transport_probe"
_TOOL_VALUE = 7
_TOOL_TRANSPORT_EPOCH = "qwen-family-jinja-prefill-host-v2"
_BENCHMARK_MARKER = "_mmm_qwen_tool_calibration_benchmark_v1"
_RUN_VARIANT_MARKER = "_mmm_qwen_tool_calibration_context_v2"
_PROBE_MARKER = "_mmm_qwen_tool_calibration_probe_v2"
_PARALLEL_STAGE_MARKER = "_mmm_qwen_mtp_single_slot_stage_v1"
_LAUNCH_MARKER = "_mmm_qwen_mtp_single_slot_launch_v1"
_FINGERPRINT_MARKER = "_mmm_qwen_mtp_single_slot_fingerprint_v2"
_WIDTH_MARKER = "_mmm_qwen_recommended_mtp_widths_v1"
_RUNTIME_CONTRACT = "qwen"
_ACTIVE_BENCHMARK_CONFIG: ContextVar[Any | None] = ContextVar(
    "mmm_qwen_transport_benchmark_config",
    default=None,
)
_ACTIVE_CALIBRATION: ContextVar[tuple[Any, Any] | None] = ContextVar(
    "mmm_qwen_transport_calibration",
    default=None,
)


def _config_extra(config: Any) -> Mapping[str, Any]:
    extra = getattr(config, "extra", {})
    return extra if isinstance(extra, Mapping) else {}


def _family(config: Any) -> str | None:
    capabilities = qwen_family_capabilities(config, required=True)
    return capabilities.family if capabilities is not None else None


def _recommended_mtp_widths(config: Any) -> str | None:
    raw = _config_extra(config).get("mtp_widths")
    if raw is None:
        return None
    values = (
        [part.strip() for part in raw.split(",")]
        if isinstance(raw, str)
        else [str(part).strip() for part in raw]
        if isinstance(raw, (list, tuple))
        else []
    )
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        try:
            width = int(value)
        except ValueError:
            return None
        if width <= 0:
            return None
        text = str(width)
        if text not in normalized:
            normalized.append(text)
    return ",".join(normalized) or None


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


def _install_mtp_width_policy(autotune: Any) -> None:
    """Bound default cold-start width search without overriding explicit tuning."""

    current = autotune.ensure_tuned_server
    if getattr(current, _WIDTH_MARKER, False):
        return

    @wraps(current)
    def ensure(config: Any, request: Any) -> str:
        widths = _recommended_mtp_widths(config)
        explicit = os.environ.get("MMM_LLAMA_MTP_WIDTHS", "")
        if widths is None or explicit.strip():
            return current(config, request)

        previous = os.environ.get("MMM_LLAMA_MTP_WIDTHS")
        os.environ["MMM_LLAMA_MTP_WIDTHS"] = widths
        try:
            return current(config, request)
        finally:
            _restore_env("MMM_LLAMA_MTP_WIDTHS", previous)

    setattr(ensure, _WIDTH_MARKER, True)
    autotune.ensure_tuned_server = ensure


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


def _tool_call_signature(response: Any) -> str:
    """Return a stable signature only for the exact expected semantic tool call."""

    calls = getattr(response, "tool_calls", None)
    if calls is not None:
        if len(calls) != 1:
            return ""
        call = calls[0]
        name = str(getattr(call, "name", "")).strip()
        arguments = _canonical_arguments(getattr(call, "arguments", {}))
    else:
        if not isinstance(response, Mapping):
            return ""
        choices = response.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
        ):
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


def _tool_probe_request() -> Any:
    from .model_adapters.base import GenerationRequest

    tool = {
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
    return GenerationRequest(
        messages=(
            {
                "role": "system",
                "content": "Use the requested function exactly once. Do not answer in text.",
            },
            {
                "role": "user",
                "content": f"Call {_TOOL_NAME} with value {_TOOL_VALUE}.",
            },
        ),
        tools=(tool,),
        tool_choice="required",
        parallel_tool_calls=False,
    )


def _tool_probe_payload(config: Any) -> tuple[Any, dict[str, Any]]:
    """Build calibration through the exact production Jinja/raw-host contract."""

    from .llama_server_hardware_policy import _server_payload

    request = _tool_probe_request()
    payload = _server_payload(SimpleNamespace(config=config), request)
    payload.update(
        {
            "max_tokens": 48,
            "seed": 1234,
            "cache_prompt": False,
            "stream": False,
        }
    )
    if payload.get("tool_choice") != "required":
        raise RuntimeError("tool calibration must render required Jinja tool choice")
    if payload.get("temperature") != 0.0:
        raise RuntimeError("tool calibration must use deterministic sampling")
    return request, payload


def _raw_tool_probe_turn(data: Mapping[str, Any], request: Any) -> Any:
    """Parse a calibration response exactly like a production Qwen tool turn."""

    choices = data.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise RuntimeError("native tool probe returned no completion choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("native tool probe returned no assistant message")
    from .model_adapters.llama_cpp_adapter import _qwen_tool_generation_response

    return _qwen_tool_generation_response(message, request)


def _tool_probe(base_url: str, autotune: Any, config: Any) -> tuple[bool, str]:
    """Exercise pure-content Jinja tools plus MMM's production host parser."""

    import httpx

    try:
        request, payload = _tool_probe_payload(config)
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=autotune._env_int("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", 300),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, Mapping):
            return False, "native tool probe returned a non-object response"
        turn = _raw_tool_probe_turn(data, request)
        signature = _tool_call_signature(turn)
        if not signature:
            return False, "host parser returned no valid canonical tool call"
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
    """Prevent guarded MTP runtimes from using unsupported multi-slot tuning."""

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

            # Multimodal owns the MTP->baseline conversion for affected configured
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
                "qwen_family": _family(config),
                "qwen_mtp_parallel": "single-slot-v1",
                "qwen_tool_transport": _TOOL_TRANSPORT_EPOCH,
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()

        setattr(qwen_mtp_fingerprint, _FINGERPRINT_MARKER, True)
        autotune._fingerprint = qwen_mtp_fingerprint


def _install_tool_equivalence_policy(autotune: Any) -> None:
    """Reject guarded speculation candidates with invalid native tool transport."""

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

        valid, error = _tool_probe(base_url, autotune, config)
        if not valid:
            raise RuntimeError(f"Qwen native tool transport calibration failed: {error}")
        return probe

    setattr(qwen_tool_probe, _PROBE_MARKER, True)
    autotune._probe_server = qwen_tool_probe


def install() -> None:
    from . import llama_server_autotune

    _install_mtp_width_policy(llama_server_autotune)
    _install_mtp_single_slot_policy(llama_server_autotune)
    _install_tool_equivalence_policy(llama_server_autotune)


__all__ = [
    "_initial_calibration_variant",
    "_install_mtp_single_slot_policy",
    "_install_mtp_width_policy",
    "_install_tool_equivalence_policy",
    "_mtp_variant",
    "_recommended_mtp_widths",
    "_single_slot_variant",
    "_tool_call_signature",
    "install",
]
