from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai.model_concurrency import (
    ReentrantReadWriteLock,
    router_native_model_parallelism,
    router_owns_native_model,
)


def test_concurrent_read_to_write_upgrades_do_not_deadlock() -> None:
    lock = ReentrantReadWriteLock()
    barrier = threading.Barrier(2)
    completed: list[int] = []
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            lock.acquire_read()
            try:
                barrier.wait(timeout=2.0)
                lock.acquire()
                try:
                    completed.append(index)
                finally:
                    lock.release()
            finally:
                lock.release_read()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(completed) == [0, 1]


class _Registry:
    def __init__(self, config) -> None:
        self.config = config

    def role(self, profile: str, role: str):
        assert role == "planner"
        return self.config


def _router(*, exclusive_gpu: bool, provider: str, adapter: str):
    config = SimpleNamespace(
        exclusive_gpu=exclusive_gpu,
        provider=provider,
        adapter=adapter,
    )
    return SimpleNamespace(profile="test", registry=_Registry(config))


def test_native_router_parallelism_requires_local_exclusive_model(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")

    owned = _router(exclusive_gpu=True, provider="local", adapter="llama_cpp")
    assert router_owns_native_model(owned) is True
    assert router_native_model_parallelism(owned) == 4

    remote = _router(exclusive_gpu=True, provider="remote", adapter="llama_cpp")
    assert router_owns_native_model(remote) is False
    assert router_native_model_parallelism(remote) == 1

    shared = _router(exclusive_gpu=False, provider="local", adapter="llama_cpp")
    assert router_owns_native_model(shared) is False
    assert router_native_model_parallelism(shared) == 1
