from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai import ecosystem_discovery
from minecraft_mod_ai import parallel_runtime_contract as parallel


def test_ordered_parallel_map_runs_independent_work_concurrently() -> None:
    barrier = threading.Barrier(3)

    def work(value: int) -> int:
        barrier.wait(timeout=2)
        return value * 10

    assert parallel._ordered_parallel_map(work, [1, 2, 3], workers=3) == [10, 20, 30]


def test_model_prefetch_deduplicates_same_gguf(monkeypatch) -> None:
    calls: list[str] = []
    gate = threading.Event()

    def resolver(config) -> str:
        calls.append(config.model_id)
        gate.wait(timeout=2)
        return "/tmp/model.gguf"

    config = SimpleNamespace(
        provider="local",
        adapter="llama_cpp",
        model_id="owner/model-GGUF",
        extra={"gguf_filename": "model.gguf"},
    )
    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", "test-receipt")
    monkeypatch.setattr(parallel, "_MODEL_RESOLVER", resolver)
    with parallel._PREFETCH_LOCK:
        parallel._PREFETCH_FUTURES.clear()

    first = parallel._ensure_model_prefetch(config)
    second = parallel._ensure_model_prefetch(config)
    assert first is not None
    assert second is first
    gate.set()
    assert first.result(timeout=2) == "/tmp/model.gguf"
    assert calls == ["owner/model-GGUF"]


def test_ecosystem_routes_run_concurrently_and_keep_route_order(monkeypatch) -> None:
    barrier = threading.Barrier(3)

    class FakeClient:
        github_token = ""

        def search(
            self,
            provider: str,
            query: str,
            *,
            limit: int,
            target_profile: str,
        ) -> dict:
            barrier.wait(timeout=2)
            return {
                "provider": provider,
                "query": query,
                "target_profile": target_profile,
                "returned": 1,
                "candidates": [{"candidate_id": provider}],
            }

    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "auto")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("MMM_DISCOVERY_WORKERS", "8")
    monkeypatch.setattr(ecosystem_discovery, "EcosystemDiscoveryClient", FakeClient)

    concurrent = parallel._parallel_discover_seed_bundle_factory(
        ecosystem_discovery,
        ecosystem_discovery.discover_seed_bundle,
    )
    result = concurrent("make a small mod", {"title": "demo", "pitch": "demo"})

    assert result["candidate_count"] == 3
    assert [page["provider"] for page in result["pages"]] == [
        "modrinth",
        "openverse_images",
        "openverse_audio",
    ]


def test_planner_runtime_has_no_optional_future_overlap_owner() -> None:
    assert not hasattr(parallel, "_PLANNER_STATE")
    assert not hasattr(parallel, "_PLANNER_AUX_EXECUTOR")
    assert not hasattr(parallel, "_install_planner_overlap")
