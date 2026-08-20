from __future__ import annotations

"""Make asynchronous model prefetch retryable after transient resolution failures."""

from functools import wraps
from typing import Any


def install(*, parallel_runtime_module: Any) -> None:
    current = parallel_runtime_module._ensure_model_prefetch
    if getattr(current, "_mmm_failed_prefetch_eviction", False):
        return

    @wraps(current)
    def ensure_model_prefetch(config: Any):
        key = parallel_runtime_module._model_key(config)
        with parallel_runtime_module._PREFETCH_LOCK:
            cached = parallel_runtime_module._PREFETCH_FUTURES.get(key)
            if cached is not None and cached.done():
                failed = cached.cancelled()
                if not failed:
                    try:
                        failed = cached.exception() is not None
                    except BaseException:
                        failed = True
                if failed and parallel_runtime_module._PREFETCH_FUTURES.get(key) is cached:
                    parallel_runtime_module._PREFETCH_FUTURES.pop(key, None)

        future = current(config)
        if future is None:
            return None

        def evict_failed(done: Any) -> None:
            failed = done.cancelled()
            if not failed:
                try:
                    failed = done.exception() is not None
                except BaseException:
                    failed = True
            if not failed:
                return
            with parallel_runtime_module._PREFETCH_LOCK:
                if parallel_runtime_module._PREFETCH_FUTURES.get(key) is done:
                    parallel_runtime_module._PREFETCH_FUTURES.pop(key, None)

        future.add_done_callback(evict_failed)
        return future

    ensure_model_prefetch._mmm_failed_prefetch_eviction = True  # type: ignore[attr-defined]
    ensure_model_prefetch.__wrapped__ = current  # type: ignore[attr-defined]
    parallel_runtime_module._ensure_model_prefetch = ensure_model_prefetch


__all__ = ["install"]
