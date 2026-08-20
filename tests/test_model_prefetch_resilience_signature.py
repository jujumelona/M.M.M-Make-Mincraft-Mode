from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

from minecraft_mod_ai.model_prefetch_resilience import install


def test_prefetch_resilience_preserves_native_resolver_argument() -> None:
    calls: list[tuple[object, object]] = []
    future: Future[str] = Future()
    future.set_result("/tmp/model.gguf")

    def ensure_model_prefetch(config: object, resolver: object) -> Future[str]:
        calls.append((config, resolver))
        return future

    config = SimpleNamespace(model_id="repo/model", extra={})
    resolver = object()
    runtime = SimpleNamespace(
        _ensure_model_prefetch=ensure_model_prefetch,
        _model_key=lambda value: (value.model_id, ""),
        _PREFETCH_LOCK=__import__("threading").RLock(),
        _PREFETCH_FUTURES={(config.model_id, ""): future},
    )

    install(parallel_runtime_module=runtime)

    assert runtime._ensure_model_prefetch(config, resolver) is future
    assert calls == [(config, resolver)]


def test_prefetch_resilience_evicts_failed_future_for_retry() -> None:
    failed: Future[str] = Future()
    failed.set_exception(RuntimeError("transient"))
    replacement: Future[str] = Future()
    replacement.set_result("/tmp/model.gguf")
    calls = 0

    def ensure_model_prefetch(config: object, resolver: object) -> Future[str]:
        nonlocal calls
        calls += 1
        key = (config.model_id, "")
        cached = runtime._PREFETCH_FUTURES.get(key)
        if cached is not None:
            return cached
        runtime._PREFETCH_FUTURES[key] = replacement
        return replacement

    config = SimpleNamespace(model_id="repo/model", extra={})
    key = (config.model_id, "")
    runtime = SimpleNamespace(
        _ensure_model_prefetch=ensure_model_prefetch,
        _model_key=lambda value: (value.model_id, ""),
        _PREFETCH_LOCK=__import__("threading").RLock(),
        _PREFETCH_FUTURES={key: failed},
    )

    install(parallel_runtime_module=runtime)

    assert runtime._ensure_model_prefetch(config, object()) is replacement
    assert runtime._PREFETCH_FUTURES[key] is replacement
    assert calls == 1
