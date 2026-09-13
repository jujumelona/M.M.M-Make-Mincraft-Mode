from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.llama_server_runtime_tuning as tuning


def test_parallel_probe_reuses_one_worker_pool_across_probe_rounds(monkeypatch) -> None:
    real_executor = tuning.ThreadPoolExecutor
    executor_constructions: list[int] = []

    class CountingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            executor_constructions.append(1)
            self._pool = real_executor(*args, **kwargs)

        def __enter__(self):
            return self._pool.__enter__()

        def __exit__(self, exc_type, exc, tb):
            return self._pool.__exit__(exc_type, exc, tb)

    monkeypatch.setattr(tuning, "ThreadPoolExecutor", CountingExecutor)

    probe_calls: list[object] = []

    def probe_server(base_url, request, *, max_tokens, variant):
        probe_calls.append(request)
        return SimpleNamespace(
            ok=True,
            output_sha256="stable-output",
            predicted_tokens=max_tokens,
            predicted_tps=10.0,
            prompt_tps=20.0,
            elapsed_seconds=0.01,
        )

    autotune = SimpleNamespace(
        _probe_server=probe_server,
        ProbeResult=SimpleNamespace,
    )
    request = SimpleNamespace(messages=(), response_format="text")
    variant = tuning.ServerVariant("baseline")

    result = tuning._parallel_probe(
        autotune,
        "http://127.0.0.1:1",
        request,
        max_tokens=4,
        variant=variant,
        concurrency=2,
    )

    assert result.ok is True
    assert len(probe_calls) == 4
    assert len(executor_constructions) == 1
