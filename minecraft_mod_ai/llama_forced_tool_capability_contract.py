from __future__ import annotations

"""Recoverable capability caching for local required-tool decoding.

A native required-tool probe is a capability check, not a permanent feature flag. One
transport timeout or server restart must not poison the endpoint/model cache for the
rest of the process. Positive evidence remains cached; protocol-negative evidence gets
a bounded TTL; transient failures get only a short probe cooldown and are never written
as permanent unsupported capability.
"""

import os
import time
from collections.abc import Mapping
from functools import wraps
from typing import Any

_PROBE_MARKER = "_mmm_recoverable_native_tool_probe_v1"
_MARK_MARKER = "_mmm_ttl_native_tool_negative_v1"
_DEFAULT_NEGATIVE_TTL_SECONDS = 60.0
_DEFAULT_TRANSIENT_COOLDOWN_SECONDS = 5.0


def _positive_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def install(forced_module: Any) -> None:
    negative_at = getattr(forced_module, "_mmm_native_probe_negative_at", None)
    if not isinstance(negative_at, dict):
        negative_at = {}
        forced_module._mmm_native_probe_negative_at = negative_at
    transient_at = getattr(forced_module, "_mmm_native_probe_transient_at", None)
    if not isinstance(transient_at, dict):
        transient_at = {}
        forced_module._mmm_native_probe_transient_at = transient_at

    current_probe = forced_module._native_required_supported
    if not getattr(current_probe, _PROBE_MARKER, False):

        @wraps(current_probe)
        def recoverable_native_required_supported(
            current: Any,
            adapter: Any,
            request: Any,
        ) -> bool:
            key = forced_module._native_probe_key(adapter, request)
            if key is None:
                return False
            now = time.monotonic()
            negative_ttl = _positive_seconds(
                "MMM_LLAMA_NATIVE_TOOL_NEGATIVE_TTL_SECONDS",
                _DEFAULT_NEGATIVE_TTL_SECONDS,
            )
            transient_cooldown = _positive_seconds(
                "MMM_LLAMA_NATIVE_TOOL_TRANSIENT_COOLDOWN_SECONDS",
                _DEFAULT_TRANSIENT_COOLDOWN_SECONDS,
            )

            with forced_module._NATIVE_PROBE_LOCK:
                cached = forced_module._NATIVE_PROBE_CACHE.get(key)
                negative_time = negative_at.get(key)
                transient_time = transient_at.get(key)
                if cached is True:
                    return True
                if cached is False and isinstance(negative_time, (int, float)):
                    if now - float(negative_time) < negative_ttl:
                        return False
                    forced_module._NATIVE_PROBE_CACHE.pop(key, None)
                    negative_at.pop(key, None)
                elif cached is False:
                    # Legacy permanent-negative entries predate this contract. Reprobe
                    # them instead of inheriting an unbounded false capability state.
                    forced_module._NATIVE_PROBE_CACHE.pop(key, None)
                if isinstance(transient_time, (int, float)) and (
                    now - float(transient_time) < transient_cooldown
                ):
                    return False

            supported = False
            try:
                turn = current(adapter, forced_module._native_probe_request(request))
                if forced_module._contains_exact_call(turn, forced_module._NATIVE_PROBE_TOOL):
                    call = tuple(getattr(turn, "tool_calls", ()) or ())[0]
                    arguments = getattr(call, "arguments", {})
                    supported = (
                        isinstance(arguments, Mapping)
                        and arguments.get("nonce") == "mmm"
                    )
            except Exception as exc:
                if forced_module._native_protocol_failure(exc):
                    with forced_module._NATIVE_PROBE_LOCK:
                        forced_module._NATIVE_PROBE_CACHE[key] = False
                        negative_at[key] = now
                        transient_at.pop(key, None)
                    print(
                        "llama native forced-tool preflight:",
                        " supported=no",
                        " reason=protocol",
                        f" model={key[1]}",
                        f" retry_after={negative_ttl:.0f}s",
                        flush=True,
                    )
                else:
                    with forced_module._NATIVE_PROBE_LOCK:
                        forced_module._NATIVE_PROBE_CACHE.pop(key, None)
                        transient_at[key] = now
                    print(
                        "llama native forced-tool preflight:",
                        " supported=unknown",
                        " reason=transient",
                        f" model={key[1]}",
                        f" retry_after={transient_cooldown:.0f}s",
                        flush=True,
                    )
                return False

            with forced_module._NATIVE_PROBE_LOCK:
                transient_at.pop(key, None)
                forced_module._NATIVE_PROBE_CACHE[key] = supported
                if supported:
                    negative_at.pop(key, None)
                else:
                    negative_at[key] = now
            print(
                "llama native forced-tool preflight:",
                f" supported={'yes' if supported else 'no'}",
                f" model={key[1]}",
                (
                    ""
                    if supported
                    else f" retry_after={negative_ttl:.0f}s"
                ),
                sep="",
                flush=True,
            )
            return supported

        setattr(recoverable_native_required_supported, _PROBE_MARKER, True)
        recoverable_native_required_supported.__wrapped__ = current_probe  # type: ignore[attr-defined]
        forced_module._native_required_supported = recoverable_native_required_supported

    current_mark = forced_module._mark_native_unsupported
    if not getattr(current_mark, _MARK_MARKER, False):

        @wraps(current_mark)
        def ttl_mark_native_unsupported(adapter: Any, request: Any) -> None:
            key = forced_module._native_probe_cache_key(adapter, request)
            current_mark(adapter, request)
            if key is not None:
                with forced_module._NATIVE_PROBE_LOCK:
                    negative_at[key] = time.monotonic()
                    transient_at.pop(key, None)

        setattr(ttl_mark_native_unsupported, _MARK_MARKER, True)
        ttl_mark_native_unsupported.__wrapped__ = current_mark  # type: ignore[attr-defined]
        forced_module._mark_native_unsupported = ttl_mark_native_unsupported


__all__ = ["install"]
