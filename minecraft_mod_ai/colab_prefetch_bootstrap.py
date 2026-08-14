from __future__ import annotations

import json
import os
import threading
from concurrent.futures import Future
from functools import wraps
from typing import Any


_PLATFORM_LOCK = threading.RLock()
_PLATFORM_FUTURE: Future[Any] | None = None
_PLATFORM_ORIGINAL: Any = None


def _start_platform_future() -> Future[Any]:
    global _PLATFORM_FUTURE
    with _PLATFORM_LOCK:
        current = _PLATFORM_FUTURE
        if current is not None and not current.cancelled():
            return current
        future: Future[Any] = Future()
        _PLATFORM_FUTURE = future

        def worker() -> None:
            try:
                future.set_result(_PLATFORM_ORIGINAL())
            except BaseException as exc:  # pragma: no cover - network boundary
                future.set_exception(exc)

        threading.Thread(
            target=worker,
            daemon=True,
            name="mmm_platform_prefetch",
        ).start()
        return future


def _install_platform_prefetch() -> None:
    """Overlap the first official Fabric Meta lookup and single-flight its miss."""

    global _PLATFORM_ORIGINAL, _PLATFORM_FUTURE

    from . import platform_live_discovery as live

    current = live.discover_game_versions
    if getattr(current, "_mmm_platform_singleflight_prefetch", False):
        _start_platform_future()
        return

    _PLATFORM_ORIGINAL = current

    @wraps(current)
    def discover_game_versions():
        global _PLATFORM_FUTURE
        future = _start_platform_future()
        try:
            return future.result()
        except BaseException:
            # The canonical caller already has an offline compatibility fallback.
            # Do not permanently memoize a transient network failure: a later call
            # may start a fresh official lookup after connectivity recovers.
            with _PLATFORM_LOCK:
                if _PLATFORM_FUTURE is future:
                    _PLATFORM_FUTURE = None
            raise

    discover_game_versions._mmm_platform_singleflight_prefetch = True
    discover_game_versions.__wrapped__ = current
    live.discover_game_versions = discover_game_versions
    _start_platform_future()


def _install_request_byte_fastpath() -> None:
    """Use the C JSON encoder for whole-string byte checks in lossless paging."""

    from . import game_design

    current = game_design._json_text_bytes
    if getattr(current, "_mmm_c_json_byte_count", False):
        return

    @wraps(current)
    def json_text_bytes(value: str) -> int:
        if not isinstance(value, str):
            return current(value)
        # json.dumps(string, ensure_ascii=False) adds exactly the two surrounding
        # quote bytes. Its escaping rules are the same rules implemented by
        # _json_character_bytes, but the full scan runs in the C encoder.
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8")) - 2

    json_text_bytes._mmm_c_json_byte_count = True
    json_text_bytes.__wrapped__ = current
    game_design._json_text_bytes = json_text_bytes


def start(model_registry_module: Any) -> None:
    """Start non-blocking metadata prefetch and install bootstrap hot paths."""

    del model_registry_module
    _install_request_byte_fastpath()
    _install_platform_prefetch()
    if not os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip():
        return
    os.environ.setdefault("MMM_DISCOVERY_WORKERS", "12")
    os.environ.setdefault("MMM_RESEARCH_WORKERS", "8")


__all__ = ["start"]
