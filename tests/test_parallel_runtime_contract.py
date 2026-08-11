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


def test_planner_overlaps_evidence_with_ecosystem_prefetch() -> None:
    evidence_started = threading.Event()
    ecosystem_started = threading.Event()
    brief = {"schema_version": "test"}
    design = {"title": "demo"}

    def parallel_retrieve(_brief):
        evidence_started.set()
        assert ecosystem_started.wait(timeout=2)
        return {"evidence": "ready"}

    def original_radar(prompt, research_brief, *args, **kwargs):
        assert prompt == "prompt"
        assert research_brief is brief
        assert evidence_started.wait(timeout=2)
        return {"radar": "ready"}

    def original_impl(prompt, game_design, research_brief=None):
        raise AssertionError("wrapped implementation path should be used")

    def original_collect(
        prompt,
        game_design,
        *,
        research_brief=None,
        page_builder=None,
        allow_legacy_terminal=False,
        **kwargs,
    ):
        assert prompt == "prompt"
        assert game_design is design
        assert research_brief is brief
        assert callable(page_builder)
        assert allow_legacy_terminal is True
        ecosystem_started.set()
        return {"ecosystem": "ready"}

    fake_module = SimpleNamespace(
        retrieve_domain_evidence=parallel_retrieve,
        discover_seed_bundle=lambda *args, **kwargs: {},
        collect_technology_radar=original_radar,
        _retrieve_implementation_evidence=original_impl,
        collect_ecosystem_seed_bundle=original_collect,
        normalize_research_brief=lambda prompt, game_design: brief,
    )

    parallel._PLANNER_STATE.evidence = None
    parallel._PLANNER_STATE.ecosystem = None
    parallel._install_planner_overlap(
        complete_planner_module=fake_module,
        parallel_retrieve=parallel_retrieve,
        parallel_discover=lambda *args, **kwargs: {},
    )

    assert fake_module.collect_technology_radar("prompt", brief) == {"radar": "ready"}
    assert fake_module._retrieve_implementation_evidence(
        "prompt", design, brief
    ) == {"evidence": "ready"}
    assert fake_module.collect_ecosystem_seed_bundle(
        "prompt",
        design,
        research_brief=brief,
        page_builder=fake_module.discover_seed_bundle,
        allow_legacy_terminal=True,
    ) == {"ecosystem": "ready"}
