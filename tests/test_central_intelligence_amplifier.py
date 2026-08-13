from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from minecraft_mod_ai.central_intelligence_amplifier import (
    build_central_committee,
    install_parallel_core,
    review_design,
)


class _CommitteeRouter:
    def __init__(self) -> None:
        self.committee_barrier = threading.Barrier(3)
        self.review_barrier = threading.Barrier(2)
        self.lock = threading.Lock()
        self.committee_active = 0
        self.committee_max = 0
        self.review_active = 0
        self.review_max = 0

    def generate_text(
        self,
        role,
        messages,
        *,
        response_format="text",
        response_schema=None,
        enable_tools=True,
        **_kwargs,
    ):
        del role, messages, enable_tools
        assert response_format == "json"
        required = set(response_schema.get("required", [])) if response_schema else set()
        if "analysis" in required:
            with self.lock:
                self.committee_active += 1
                self.committee_max = max(self.committee_max, self.committee_active)
            try:
                self.committee_barrier.wait(timeout=2)
            finally:
                with self.lock:
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
                        "acceptance_observables": ["feature is observable in game"],
                        "unresolved_questions": [],
                    }
                }
            )
        if "review" in required:
            with self.lock:
                self.review_active += 1
                self.review_max = max(self.review_max, self.review_active)
            try:
                self.review_barrier.wait(timeout=2)
            finally:
                with self.lock:
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


def test_central_committee_and_reviewers_run_in_parallel() -> None:
    router = _CommitteeRouter()
    committee = build_central_committee(router, "요청한 기능만 구현")
    assert committee["parallel"] is True
    assert router.committee_max >= 3

    reviews = review_design(
        router,
        "요청한 기능만 구현",
        {
            "title": "테스트",
            "pitch": "요청",
            "core_loop": [],
            "progression": [],
            "combat": {},
            "mod_context": {},
            "modules": [],
            "assets": [],
            "acceptance_tests": [],
        },
        research={},
    )
    assert len(reviews) == 2
    assert router.review_max >= 2


def test_parallel_core_overlaps_providers_domains_and_sections() -> None:
    provider_barrier = threading.Barrier(3)
    domain_barrier = threading.Barrier(3)
    section_barrier = threading.Barrier(3)
    lock = threading.Lock()
    maxima = {"provider": 0, "domain": 0, "section": 0}
    active = {"provider": 0, "domain": 0, "section": 0}

    def enter(kind: str, barrier: threading.Barrier) -> None:
        with lock:
            active[kind] += 1
            maxima[kind] = max(maxima[kind], active[kind])
        try:
            barrier.wait(timeout=2)
        finally:
            with lock:
                active[kind] -= 1

    def provider(value):
        enter("provider", provider_barrier)
        return value

    def domain_worker(
        _router,
        *,
        prompt,
        domain,
        deterministic,
        trace_metadata,
    ):
        del prompt, deterministic, trace_metadata
        enter("domain", domain_barrier)
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
        enter("section", section_barrier)
        values = {}
        for field in fields:
            if field in {"title", "pitch"}:
                values[field] = section_id
            elif field in {"combat", "mod_context", "art_direction"}:
                values[field] = {}
            else:
                values[field] = []
        return values

    def original_collect(_router, prompt, *, trace_metadata=None):
        del prompt, trace_metadata
        return {"original": True}

    def original_generate(
        _game_design_module,
        _router,
        _prompt,
        *,
        media_paths=(),
        research,
        trace_metadata=None,
    ):
        del media_paths, research, trace_metadata
        return {"original": True}

    module = SimpleNamespace(
        collect_pre_design_research=original_collect,
        generate_sectioned_game_design=original_generate,
        normalize_research_brief=lambda prompt, _seed: {
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
        _json_sha256=lambda value: "sha256:test",
        _SECTION_SPECS=(
            ("identity", ("title", "pitch", "core_loop"), {}),
            ("systems", ("progression", "combat", "mod_context"), {}),
            ("quality", ("acceptance_tests", "art_direction"), {}),
        ),
        _generate_section=section_worker,
    )

    install_parallel_core(module)

    research = module.collect_pre_design_research(object(), "test")
    assert [item["domain_id"] for item in research["domain_notes"]] == ["a", "b", "c"]
    assert maxima["provider"] >= 3
    assert maxima["domain"] >= 3

    game_design_module = SimpleNamespace(_validate_design=lambda value: None)
    result = module.generate_sectioned_game_design(
        game_design_module,
        object(),
        "test",
        research=research,
    )
    assert result["title"] == "identity"
    assert maxima["section"] >= 3
