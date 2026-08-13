from __future__ import annotations

import os
from functools import wraps
from typing import Any


_ENSURE_MARKER = "_mmm_qwen35_bounded_cold_tuning_v1"
_PAYLOAD_MARKER = "_mmm_qwen35_unbounded_output_v1"
_CACHE_MARKER = "_mmm_qwen35_skip_cold_cache_reuse_probe_v1"
_KV_MARKER = "_mmm_qwen35_skip_main_kv_probe_v1"
_FAST_TUNING_ENV = "MMM_QWEN35_FAST_TUNING_ACTIVE"
_QWEN_ACTIVE_TUNING_ENV = "MMM_QWEN35_MTP_ACTIVE_TUNING"
_FAST_MTP_WIDTHS = "2,4,8"
_EXHAUSTIVE_MTP_WIDTHS = "1,2,3,4,5,6,8"


def _is_qwen35_mtp(config: Any) -> bool:
    from .qwen35_mtp_hotpath_contract import _is_qwen35_mtp as current

    return current(config)


def _tuning_mode() -> str:
    raw = os.environ.get("MMM_QWEN35_MTP_TUNING", "fast").strip().lower()
    return "exhaustive" if raw in {"full", "exhaustive", "deep"} else "fast"


def _output_token_limit() -> int:
    """Use llama.cpp's unlimited -1 default unless the operator requests a cap."""

    raw = os.environ.get("MMM_QWEN35_MAX_OUTPUT_TOKENS", "").strip()
    if not raw:
        return -1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MMM_QWEN35_MAX_OUTPUT_TOKENS must be -1 or a positive integer"
        ) from exc
    if value == -1 or value > 0:
        return value
    raise ValueError("MMM_QWEN35_MAX_OUTPUT_TOKENS must be -1 or a positive integer")


def _fast_tuning_defaults() -> dict[str, str]:
    """Return cold-start defaults that retain measured width selection on one T4."""

    if _tuning_mode() == "exhaustive":
        return {}
    return {
        # Width is also installed into Qwen's outer default below. Keeping it here
        # handles direct/non-standard call paths consistently.
        "MMM_LLAMA_MTP_WIDTHS": _FAST_MTP_WIDTHS,
        "MMM_LLAMA_MTP_CONFIDENCE_WIDTHS": "",
        "MMM_LLAMA_MTP_P_MIN_CANDIDATES": "0",
        "MMM_LLAMA_UBATCH_CANDIDATES": "512",
        "MMM_LLAMA_NGRAM_SPEC_TYPES": "",
        "MMM_LLAMA_AUTOTUNE_TOKENS": "96",
        # Draft KV search added three more full model/server reloads. q4_0 is the
        # memory-bandwidth-oriented T4 default; exhaustive mode retains measurement.
        "MMM_QWEN35_MTP_DRAFT_KV": "q4_0",
        _FAST_TUNING_ENV: "1",
    }


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _install_qwen_width_default() -> None:
    """Make the outer Qwen wrapper choose the fast width set before inner tuners run."""

    from . import qwen35_mtp_hotpath_contract as qwen_hotpath

    qwen_hotpath._DEFAULT_MTP_WIDTHS = (
        _EXHAUSTIVE_MTP_WIDTHS
        if _tuning_mode() == "exhaustive"
        else _FAST_MTP_WIDTHS
    )


def _install_output_policy(hardware_policy: Any) -> None:
    current = hardware_policy._server_payload
    if getattr(current, _PAYLOAD_MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = current(adapter, request)
        if _is_qwen35_mtp(getattr(adapter, "config", None)):
            # llama.cpp maps max_tokens to n_predict and defines -1 as unlimited.
            # Context, EOS and host validation terminate the response instead of the
            # historical fixed 8192-token transport ceiling.
            result["max_tokens"] = _output_token_limit()
        return result

    setattr(payload, _PAYLOAD_MARKER, True)
    hardware_policy._server_payload = payload


def _install_main_kv_probe_policy() -> None:
    """Skip generic main-KV search only inside Qwen's default fast cold tune."""

    from . import llama_decode_speed_contract as decode_speed

    current = decode_speed._kv_autotune_enabled
    if getattr(current, _KV_MARKER, False):
        return

    @wraps(current)
    def kv_autotune_enabled(autotune: Any) -> bool:
        if (
            _tuning_mode() != "exhaustive"
            and os.environ.get(_QWEN_ACTIVE_TUNING_ENV, "").strip() == "1"
        ):
            return False
        return bool(current(autotune))

    setattr(kv_autotune_enabled, _KV_MARKER, True)
    decode_speed._kv_autotune_enabled = kv_autotune_enabled


def _install_cache_probe_policy(runtime_tuning: Any) -> None:
    current = runtime_tuning._cache_reuse_candidates
    if getattr(current, _CACHE_MARKER, False):
        return

    @wraps(current)
    def cache_reuse_candidates() -> tuple[int, ...]:
        if os.environ.get(_FAST_TUNING_ENV, "").strip() == "1":
            return ()
        return tuple(current())

    setattr(cache_reuse_candidates, _CACHE_MARKER, True)
    runtime_tuning._cache_reuse_candidates = cache_reuse_candidates


def _install_cold_tuning_policy(autotune: Any) -> None:
    current = autotune.ensure_tuned_server
    if getattr(current, _ENSURE_MARKER, False):
        return

    @wraps(current)
    def ensure(config: Any, request: Any) -> str:
        if not _is_qwen35_mtp(config):
            return current(config, request)

        managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
        managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
        if managed_process is not None and managed_process.poll() is None and managed_url:
            return current(config, request)

        defaults = _fast_tuning_defaults()
        previous: dict[str, str | None] = {}
        changed: list[str] = []
        for name, value in defaults.items():
            existing = os.environ.get(name)
            previous[name] = existing
            if not (existing or "").strip():
                os.environ[name] = value
                changed.append(name)
        try:
            return current(config, request)
        finally:
            for name in reversed(changed):
                _restore_env(name, previous[name])

    setattr(ensure, _ENSURE_MARKER, True)
    autotune.ensure_tuned_server = ensure


def install(autotune: Any, hardware_policy: Any, runtime_tuning: Any) -> None:
    """Remove Qwen's 8K output cap and bound expensive cold-start measurements.

    Fast mode measures baseline plus native MTP widths 2/4/8, while duplicate
    confidence/p-min/main-KV/draft-KV/ubatch/ngram/cache-reuse reload stages are
    suppressed. Explicit environment settings remain authoritative, and
    MMM_QWEN35_MTP_TUNING=exhaustive restores the full measured search.
    """

    _install_qwen_width_default()
    _install_output_policy(hardware_policy)
    _install_main_kv_probe_policy()
    _install_cache_probe_policy(runtime_tuning)
    _install_cold_tuning_policy(autotune)


__all__ = ["_fast_tuning_defaults", "_output_token_limit", "install"]
