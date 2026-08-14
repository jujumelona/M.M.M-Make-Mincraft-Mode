from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from minecraft_mod_ai.central_intelligence_amplifier import (
    build_central_committee,
    install_parallel_core,
    review_design,
)


class _CouncilRouter:
    def __init__(self) -> None:
        self._committee_barrier = threading.Barrier(3)
        self._review_barrier = threading.Barrier(2)
        self._lock = threading.Lock()
        self.profile = "test"
        self.registry = SimpleNamespace(
            role=lambda _profile, _role: SimpleNamespace(
                exclusive_gpu=True,
                provider="local",
                adapter="llama_cpp",
            )
        )
        self.committee_active = 0
        self.committee_max = 0
        self.review_active = 0
        self.review_max = 0

    def generate_text(
        self,
        _role,
        _messages,
        *,
        response_format="text",
        response_schema=None,
        **_kwargs,
    ):
        assert response_format == "json"
        required = set((response_schema or {}).get("required", ()))
        if "analysis" in required:
            with self._lock:
                self.committee_active += 1
                self.committee_max = max(self.committee_max, self.committee_active)
            try:
                self._committee_barrier.wait(timeout=2)
            finally:
                with self._lock:
                    self.committee_active -= 1
            return json.dumps(
                {
                    "analysis": {
                        "must_preserve": ["requested feature"],
                        "must_not_invent": ["unrequested map"],
                        "subproblems": ["state", "behavior"],
                        "risks": ["integration"],
                        "research_questions": ["which API is authoritative?"],
                        "confidence": 0.8,
                    }
                }
            )
        if "synthesis" in required:
            return json.dumps(
                {
                    "synthesis": {
                        "requirements": ["requested feature"],
                        "negative_constraints": ["unrequested map"],
                        "subproblem_order": ["state", "behavior"],
                        "acceptance_observables": ["observable in game"],
                        "unresolved_questions": [],
                    }
                }
            )
        if "review" in required:
            with self._lock:
                self.review_active += 1
                self.review_max = max(self.review_max, self.review_active)
            try:
                self._review_barrier.wait(timeout=2)
            finally:
                with self._lock:
                    self.review_active -= 1
            return json.dumps(
                {
                    "review": {
                        "missing_requirements": [],
                        "unsupported_additions": [],
                        "contradictions": [],
                        "research_gaps": [],
                        "affected_sections": [],
                        "severity": "none",
                        "confidence": 0.9,
                    }
                }
            )
        raise AssertionError(f"unexpected schema: {response_schema}")


def test_specialist_committee_and_adversarial_review_really_overlap() -> None:
    router = _CouncilRouter()
    committee = build_central_committee(router, "implement only the requested feature")
    assert committee["parallel"] is True
    assert committee["workers"] >= 3
    assert router.committee_max >= 3

    reviews = review_design(
        router,
        "implement only the requested feature",
        {"title": "x"},
        research={},
    )
    assert len(reviews) == 2
    assert router.review_max >= 2


def test_provider_domain_and_design_fanout_are_parallel_with_deterministic_merge() -> None:
    provider_barrier = threading.Barrier(3)
    domain_barrier = threading.Barrier(3)
    design_barrier = threading.Barrier(3)
    lock = threading.Lock()
    active = {"provider": 0, "domain": 0, "design": 0}
    maxima = {"provider": 0, "domain": 0, "design": 0}

    def overlap(kind: str, barrier: threading.Barrier) -> None:
        with lock:
            active[kind] += 1
            maxima[kind] = max(maxima[kind], active[kind])
        try:
            barrier.wait(timeout=2)
        finally:
            with lock:
                active[kind] -= 1

    def provider(payload):
        overlap("provider", provider_barrier)
        return payload

    def domain_worker(
        _router,
        *,
        prompt,
        domain,
        deterministic,
        trace_metadata,
    ):
        del prompt, deterministic, trace_metadata
        overlap("domain", domain_barrier)
        return {
            "domain_id": domain["domain_id"],
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "sufficient": True,
        }

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
        del prompt, properties, research, media_paths, trace_metadata
        overlap("design", design_barrier)
        return {fields[0]: section_id}

    def old_collect(_router, prompt, *, trace_metadata=None):
        del prompt, trace_metadata
        return {"old": True}

    module = SimpleNamespace(
        collect_pre_design_research=old_collect,
        generate_sectioned_game_design=lambda *_args, **_kwargs: {"old": True},
        normalize_research_brief=lambda _prompt, _seed: {
            "domains": [
                {"domain_id": "a", "queries": ["a"]},
                {"domain_id": "b", "queries": ["b"]},
                {"domain_id": "c", "queries": ["c"]},
            ]
        },
        retrieve_domain_evidence=lambda brief: provider({"brief": brief}),
        collect_technology_radar=lambda *args, **kwargs: provider({"technology": True}),
        collect_ecosystem_seed_bundle=lambda *args, **kwargs: provider({"ecosystem": True}),
        build_technology_radar=object(),
        discover_seed_bundle=object(),
        _research_domain_with_agent=domain_worker,
        _error=lambda stage, exc: {"stage": stage, "error": str(exc)},
        _json_sha256=lambda _value: "sha256:test",
        _SECTION_SPECS=(
            ("identity", ("title",), {}),
            ("systems", ("progression",), {}),
            ("quality", ("acceptance_tests",), {}),
        ),
        _generate_section=section_worker,
    )

    install_parallel_core(module)
    native_router = SimpleNamespace(
        profile="test",
        registry=SimpleNamespace(
            role=lambda _profile, _role: SimpleNamespace(
                exclusive_gpu=True,
                provider="local",
                adapter="llama_cpp",
            )
        ),
    )
    result = module.collect_pre_design_research(native_router, "test")

    assert maxima["provider"] >= 3
    assert maxima["domain"] >= 3
    assert [row["domain_id"] for row in result["domain_notes"]] == ["a", "b", "c"]
    assert result["method"]["parallel_specialists"]

    game_design_module = SimpleNamespace(_validate_design=lambda _value: None)
    design = module.generate_sectioned_game_design(
        game_design_module,
        native_router,
        "test",
        research=result,
    )
    assert maxima["design"] >= 3
    assert design == {
        "title": "identity",
        "progression": "systems",
        "acceptance_tests": "quality",
    }
