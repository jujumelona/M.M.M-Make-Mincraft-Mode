from __future__ import annotations

"""Fail-closed hot-path reuse for an already selected MMM llama-server.

This contract sits outside the composed llama tuning wrappers. It bypasses model-path
resolution, VRAM policy work, and HTTP health probes only when the live MMM-owned
process still matches the exact runtime-selection inputs that produced its receipt.
Any missing or stale proof falls through to the canonical server lifecycle unchanged.
"""

import json
import os
from functools import wraps
from typing import Any

_MARKER = "_mmm_managed_llama_exact_reuse_v1"


def _runtime_receipt(autotune: Any) -> dict[str, Any] | None:
    receipt = getattr(autotune, "_MMM_LLAMA_RUNTIME_RECEIPT", None)
    if isinstance(receipt, dict):
        return receipt
    raw = os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _reusable_managed_url(
    autotune: Any,
    runtime_tuning: Any,
    config: Any,
) -> str:
    """Return the live managed URL only when its selection receipt is exact."""

    lock = getattr(autotune, "_AUTOTUNE_LOCK", None)
    if lock is None:
        return ""
    with lock:
        process = getattr(autotune, "_MANAGED_PROCESS", None)
        managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "").strip().rstrip("/")
        configured_url = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
        if (
            process is None
            or process.poll() is not None
            or not managed_url
            or configured_url != managed_url
        ):
            return ""

        receipt = _runtime_receipt(autotune)
        if not isinstance(receipt, dict):
            return ""
        receipt_sha = str(receipt.get("selection_inputs_sha256", "")).strip()
        if not receipt_sha:
            return ""
        try:
            expected_sha = runtime_tuning._json_fingerprint(
                runtime_tuning._selection_inputs(config)
            )
        except Exception:
            # Reuse is an optimization, never an authority boundary. If current
            # selection inputs cannot be proven, preserve the canonical lifecycle.
            return ""
        return managed_url if receipt_sha == expected_sha else ""


def _install_fast_path(autotune: Any, runtime_tuning: Any) -> None:
    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_tuned_server(config: Any, request: Any) -> str:
        managed_url = _reusable_managed_url(autotune, runtime_tuning, config)
        if managed_url:
            return managed_url
        return current(config, request)

    setattr(ensure_tuned_server, _MARKER, True)
    ensure_tuned_server._mmm_managed_server_fast_path = True
    autotune.ensure_tuned_server = ensure_tuned_server


def install() -> None:
    from . import llama_server_autotune, llama_server_runtime_tuning

    _install_fast_path(llama_server_autotune, llama_server_runtime_tuning)


__all__ = ["_install_fast_path", "_reusable_managed_url", "install"]
