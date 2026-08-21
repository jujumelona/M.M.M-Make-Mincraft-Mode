from __future__ import annotations

import os
from functools import wraps
from typing import Any, Mapping

from .runtime_contract_wrappers import has_contract_marker, owns_contract_marker

_ENSURE_MARKER = "_mmm_qwen35_bounded_cold_tuning_v2"
_PAYLOAD_MARKER = "_mmm_qwen35_profile_output_default_v5"
_CACHE_MARKER = "_mmm_qwen35_skip_cold_cache_reuse_probe_v1"
_KV_MARKER = "_mmm_qwen35_skip_main_kv_probe_v1"
_PROBE_MARKER = "_mmm_qwen35_fast_primary_probe_v1"
_FAST_TUNING_ENV = "MMM_QWEN35_FAST_TUNING_ACTIVE"
_QWEN_ACTIVE_TUNING_ENV = "MMM_QWEN35_MTP_ACTIVE_TUNING"
_FAST_MTP_WIDTHS = "2,4,6"
_EXHAUSTIVE_MTP_WIDTHS = "1,2,3,4,5,6,8"
_RESEARCH_NOTE_MAX_TOKENS = 2048
_BOUNDED_SECTION_MAX_TOKENS = 2048


def _is_qwen35_mtp(config: Any) -> bool:
    from .qwen35_mtp_hotpath_contract import _is_qwen35_mtp as current
    return current(config)


def _tuning_mode() -> str:
    raw = os.environ.get("MMM_QWEN35_MTP_TUNING", "fast").strip().lower()
    return "exhaustive" if raw in {"full", "exhaustive", "deep"} else "fast"


def _output_token_limit() -> int | None:
    raw = os.environ.get("MMM_QWEN35_MAX_OUTPUT_TOKENS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MMM_QWEN35_MAX_OUTPUT_TOKENS must be -1 or a positive integer") from exc
    if value == -1 or value > 0:
        return value
    raise ValueError("MMM_QWEN35_MAX_OUTPUT_TOKENS must be -1 or a positive integer")


def _fast_tuning_defaults() -> dict[str, str]:
    if _tuning_mode() == "exhaustive":
        return {}
    return {
        "MMM_LLAMA_MTP_WIDTHS": _FAST_MTP_WIDTHS,
        "MMM_LLAMA_MTP_CONFIDENCE_WIDTHS": "",
        "MMM_LLAMA_MTP_P_MIN_CANDIDATES": "0",
        "MMM_LLAMA_UBATCH_CANDIDATES": "512",
        "MMM_LLAMA_NGRAM_SPEC_TYPES": "",
        "MMM_LLAMA_AUTOTUNE_TOKENS": "96",
        "MMM_QWEN35_MTP_DRAFT_KV": "f16",
        _FAST_TUNING_ENV: "1",
    }


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _bounded_section_request(request: Any) -> bool:
    section_purpose = str(getattr(request, "section_purpose", "") or "").strip()
    if section_purpose:
        return True
    schema = getattr(request, "response_schema", None)
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    return isinstance(properties, Mapping) and "section" in properties


def _research_note_request(request: Any) -> bool:
    schema = getattr(request, "response_schema", None)
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    return isinstance(properties, Mapping) and "research_note" in properties


def _bounded_output_limit(
    adapter: Any,
    payload: Mapping[str, Any],
    operator_limit: int | None,
    schema_limit: int,
) -> int:
    candidates = [schema_limit]
    for value in (
        getattr(getattr(adapter, "config", None), "max_new_tokens", None),
        payload.get("max_tokens"),
        operator_limit if operator_limit is not None and operator_limit > 0 else None,
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            candidates.append(parsed)
    return max(1, min(candidates))


def _research_note_output_limit(
    adapter: Any,
    payload: Mapping[str, Any],
    operator_limit: int | None,
) -> int:
    return _bounded_output_limit(
        adapter,
        payload,
        operator_limit,
        _RESEARCH_NOTE_MAX_TOKENS,
    )


def _install_output_policy(hardware_policy: Any) -> None:
    current = hardware_policy._server_payload
    if getattr(current, _PAYLOAD_MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = current(adapter, request)
        if _is_qwen35_mtp(getattr(adapter, "config", None)):
            limit = _output_token_limit()
            if _research_note_request(request):
                # Research is paginated by domain/page outside this transport call.
                # A schema-local cap prevents a malformed note from decoding for tens
                # of thousands of tokens. An operator may lower this bound, but -1
                # cannot disable the schema safety boundary.
                result["max_tokens"] = _research_note_output_limit(
                    adapter, result, limit
                )
            elif _bounded_section_request(request):
                # Section synthesis is itself paginated. Keep each decode bounded even
                # when the global operator policy requests unbounded output; -1 applies
                # only to genuinely unbounded top-level turns, never a bounded page.
                result["max_tokens"] = _bounded_output_limit(
                    adapter,
                    result,
                    limit,
                    _BOUNDED_SECTION_MAX_TOKENS,
                )
            else:
                if limit is None:
                    configured = getattr(getattr(adapter, "config", None), "max_new_tokens", None)
                    try:
                        configured = int(configured)
                    except (TypeError, ValueError):
                        configured = 0
                    if configured > 0:
                        # Reassert the profile value even if a live notebook still has
                        # the prior unbounded-output wrapper installed underneath us.
                        result["max_tokens"] = configured
                else:
                    result["max_tokens"] = limit
        return result

    setattr(payload, _PAYLOAD_MARKER, True)
    hardware_policy._server_payload = payload


def _install_main_kv_probe_policy() -> None:
    from . import llama_decode_speed_contract as decode_speed
    current = decode_speed._kv_autotune_enabled
    if getattr(current, _KV_MARKER, False):
        return

    @wraps(current)
    def kv_autotune_enabled(autotune: Any) -> bool:
        if _tuning_mode() != "exhaustive" and os.environ.get(_QWEN_ACTIVE_TUNING_ENV, "").strip() == "1":
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


def _install_fast_probe_policy(autotune: Any) -> None:
    """Avoid a second correctness generation for deterministic fast Qwen probes.

    The generic hardware probe wrapper appends a Java sentinel generation to every
    measured candidate. Fast Qwen tuning already admits a speculative candidate only
    when its deterministic primary output hash is byte-identical to the baseline, so
    repeating a second generated sentinel for each width adds decode cost without
    changing the fast-mode selection rule. Exhaustive tuning and non-Qwen probes keep
    the full generic sentinel path.
    """

    current = autotune._probe_server
    if has_contract_marker(current, _PROBE_MARKER):
        return
    primary = (
        getattr(current, "__wrapped__", None)
        if owns_contract_marker(current, "_mmm_correctness_sentinel")
        else None
    )

    @wraps(current)
    def probe(
        base_url: str,
        request: Any,
        *,
        max_tokens: int,
        variant: Any,
    ) -> Any:
        if (
            os.environ.get(_FAST_TUNING_ENV, "").strip() == "1"
            and callable(primary)
        ):
            return primary(
                base_url,
                request,
                max_tokens=max_tokens,
                variant=variant,
            )
        return current(
            base_url,
            request,
            max_tokens=max_tokens,
            variant=variant,
        )

    setattr(probe, _PROBE_MARKER, True)
    autotune._probe_server = probe


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
            replace_outer_default = (
                name == "MMM_LLAMA_MTP_WIDTHS"
                and _tuning_mode() != "exhaustive"
                and (existing or "").strip() == _EXHAUSTIVE_MTP_WIDTHS
            )
            if not (existing or "").strip() or replace_outer_default:
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
    _install_output_policy(hardware_policy)
    _install_main_kv_probe_policy()
    _install_cache_probe_policy(runtime_tuning)
    _install_fast_probe_policy(autotune)
    _install_cold_tuning_policy(autotune)


__all__ = [
    "_bounded_output_limit",
    "_fast_tuning_defaults",
    "_install_fast_probe_policy",
    "_output_token_limit",
    "_research_note_output_limit",
    "_research_note_request",
    "install",
]
