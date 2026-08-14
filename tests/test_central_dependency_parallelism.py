from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from minecraft_mod_ai.central_intelligence_amplifier import (
    _dependency_waves,
    _research_domain_worker_count,
    install_parallel_core,
)
from minecraft_mod_ai.model_router import ModelRouter


class _Registry:
    def __init__(self, *, provider: str = "local") -> None:
        self.provider = provider

    def role(self, _profile, _role):
        return SimpleNamespace(
            exclusive_gpu=True,
            provider=self.provider,
            adapter="llama_cpp",
        )


class _Router(ModelRouter):
    profile = "test"

    def __init__(self, *, provider: str = "local") -> None:
        # This is an intentionally minimal managed-router probe. The capacity policy only
        # needs the production ModelRouter identity plus the planner role contract; no model
        # backend is constructed by these tests.
        self.registry = _Registry(provider=provider)


def _receipt(slots: int) -> str:
    return json.dumps(
        {
            "schema_version": "mmm/llama-runtime-receipt-v1",
            "slots": slots,
        }
    )


def test_dependency_waves_preserve_order_and_serialize_bad_graphs() -> None:
    domains = (
        {"domain_id": "a", "depends_on": []},
        {"domain_id": "b", "depends_on": ["a"]},
        {"domain_id": "c", "depends_on": []},
        {"domain_id": "d", "depends_on": ["b", "c"]},
    )
    assert _dependency_waves(domains) == ((0, 2), (1,), (3,))

    assert _dependency_waves(
        (
            {"domain_id": "a", "depends_on": ["b"]},
            {"domain_id": "b", "depends_on": ["a"]},
        )
    ) == ((0,), (1,))
    assert _dependency_waves(
        (
            {"domain_id": "a", "depends_on": ["missing"]},
            {"domain_id": "b", "depends_on": []},
        )
    ) == ((0,), (1,))
    assert _dependency_waves(
        (
            {"domain_id": "same", "depends_on": []},
            {"domain_id": "same", "depends_on": []},
        )
    ) == ((0,), (1,))


def test_model_fanout_requires_matching_managed_receipt_after_activation(monkeypatch) -> None:
    router = _Router()
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    monkeypatch.delenv("MMM_LLAMA_RUNTIME_RECEIPT", raising=False)
    assert _research_domain_worker_count(router, 4) == 1

    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _receipt(4))
    assert _research_domain_worker_count(router, 4) == 4
    assert _research_domain_worker_count(router, 2) == 2

    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _receipt(2))
    assert _research_domain_worker_count(router, 4) == 1

    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_RUNTIME_RECEIPT", raising=False)
    assert _research_domain_worker_count(router, 4) == 4
    assert _research_domain_worker_count(_Router(provider="remote"), 4) == 1


def test_parallel_core_runs_research_in_dependency_waves_and_design_on_p2(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    monkeypatch.setenv("MMM_LLAMA_RUNTIME_RECEIPT", _receipt(2))

    domains = [
        {"domain_id": "a", "depends_on": [], "queries": ["a"]},
        {"domain_id": "b", "depends_on": ["a"], "queries": ["b"]},
        {"domain_id": "c", "depends_on": [], "queries": ["c"]},
        {"domain_id": "d", "depends_on": ["b", "c"], "queries": ["d"]},
    ]
    specs = (
        ("identity", ("title",), {}),
        ("systems", ("progression",), {}),
        ("modules", ("modules",), {}),
        ("quality", ("acceptance_tests",), {}),
    )

    research_lock = threading.Lock()
    research_barrier = threading.Barrier(2)
    research_active = 0
    research_max_active = 0
    started: dict[str, float] = {}
    finished: dict[str, float] = {}

    def research_worker(
        _router,
        *,
        prompt,
        domain,
        deterministic,
        trace_metadata,
    ):
        nonlocal research_active, research_max_active
        del prompt, deterministic, trace_metadata
        domain_id = domain["domain_id"]
        with research_lock:
            research_active += 1
            research_max_active = max(research_max_active, research_active)
            started[domain_id] = time.monotonic()
        try:
            if domain_id in {"a", "c"}:
                research_barrier.wait(timeout=2)
            time.sleep(0.02)
            return {
                "domain_id": domain_id,
                "claims": [],
                "gaps": [],
                "next_queries": [],
                "sufficient": True,
            }
        finally:
            with research_lock:
                finished[domain_id] = time.monotonic()
                research_active -= 1

    design_lock = threading.Lock()
    design_barrier = threading.Barrier(2)
    design_active = 0
    design_max_active = 0
    media_seen: dict[str, tuple[str, ...]] = {}

    def section_worker(
        _router,
        *,
        prompt,
        section_id,
        fields,
        properties,
        research,
        media_paths,
        trace_metadata,
    ):
        nonlocal design_active, design_max_active
        del prompt, properties, research, trace_metadata
        with design_lock:
            design_active += 1
            design_max_active = max(design_max_active, design_active)
            media_seen[section_id] = tuple(media_paths)
        try:
            design_barrier.wait(timeout=2)
            time.sleep(0.02)
            return {fields[0]: section_id}
        finally:
            with design_lock:
                design_active -= 1

    module = SimpleNamespace(
        collect_pre_design_research=lambda *_args, **_kwargs: {},
        generate_sectioned_game_design=lambda *_args, **_kwargs: {},
        normalize_research_brief=lambda _prompt, _seed: {"domains": list(domains)},
        retrieve_domain_evidence=lambda _brief: {"status": "ok"},
        collect_technology_radar=lambda *_args, **_kwargs: {"status": "ok"},
        build_technology_radar=lambda *_args, **_kwargs: {},
        collect_ecosystem_seed_bundle=lambda *_args, **_kwargs: {"status": "ok"},
        discover_seed_bundle=lambda *_args, **_kwargs: {},
        _error=lambda stage, exc: {"stage": stage, "error": str(exc)},
        _research_domain_with_agent=research_worker,
        _json_sha256=lambda _value: "sha256:test",
        _SECTION_SPECS=specs,
        _generate_section=section_worker,
    )
    install_parallel_core(module)

    research = module.collect_pre_design_research(_Router(), "research")
    assert [note["domain_id"] for note in research["domain_notes"]] == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert research_max_active == 2
    assert started["b"] >= finished["a"]
    assert started["d"] >= finished["b"]
    assert started["d"] >= finished["c"]
    assert research["method"]["parallel_specialist_workers"] == 2
    assert research["method"]["dependency_wave_count"] == 3

    game_design_module = SimpleNamespace(_validate_design=lambda _value: None)
    design = module.generate_sectioned_game_design(
        game_design_module,
        _Router(),
        "design",
        media_paths=("reference.png",),
        research=research,
    )
    assert design_max_active == 2
    assert media_seen["identity"] == ("reference.png",)
    assert all(
        media_seen[section_id] == ()
        for section_id in ("systems", "modules", "quality")
    )
    assert design == {
        "title": "identity",
        "progression": "systems",
        "modules": "modules",
        "acceptance_tests": "quality",
    }
