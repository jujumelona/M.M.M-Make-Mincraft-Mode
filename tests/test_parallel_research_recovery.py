from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace

from minecraft_mod_ai import central_intelligence_amplifier as amplifier
from minecraft_mod_ai.minecraft_knowledge_contract import (
    compile_minecraft_knowledge_plan,
    evaluate_route_coverage,
)


def _sha(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _native_router():
    return SimpleNamespace(
        profile="test",
        registry=SimpleNamespace(
            role=lambda _profile, _role: SimpleNamespace(
                exclusive_gpu=True,
                provider="local",
                adapter="llama_cpp",
            )
        ),
    )


def _fake_agentic(research_domain):
    domains = [
        {
            "domain_id": "mk_entity",
            "objective": "entity",
            "requirements": ["mk:entity.registration"],
            "evidence_kinds": ["minecraft_api"],
            "queries": ["entity registration"],
            "providers": ["official_docs"],
            "depends_on": [],
        },
        {
            "domain_id": "mk_quality",
            "objective": "quality",
            "requirements": ["mk:quality.compile"],
            "evidence_kinds": ["testing"],
            "queries": ["gradle build"],
            "providers": ["official_docs"],
            "depends_on": [],
        },
    ]

    def collect(*args, **kwargs):
        raise AssertionError("parallel core must own collection")

    return SimpleNamespace(
        collect_pre_design_research=collect,
        normalize_research_brief=lambda prompt, design: {"domains": domains},
        retrieve_domain_evidence=lambda brief: {},
        collect_technology_radar=lambda *args, **kwargs: {},
        build_technology_radar=lambda *args, **kwargs: {},
        collect_ecosystem_seed_bundle=lambda *args, **kwargs: {},
        discover_seed_bundle=lambda *args, **kwargs: {},
        _research_domain_with_agent=research_domain,
        _error=lambda key, exc: {"stage": key, "error": str(exc)},
        _json_sha256=_sha,
        generate_sectioned_game_design=lambda *args, **kwargs: {},
        _SECTION_SPECS=(),
    )


def test_parallel_domain_failure_recovers_serially_without_weakening_terminal_semantics() -> None:
    attempts: dict[str, int] = {}

    def research_domain(router, *, prompt, domain, deterministic, trace_metadata):
        domain_id = domain["domain_id"]
        attempts[domain_id] = attempts.get(domain_id, 0) + 1
        if threading.current_thread().name.startswith("mmm_research_domain"):
            raise RuntimeError("shared local router rejected concurrent request")
        return {
            "domain_id": domain_id,
            "claims": [],
            "gaps": ["exact API lookup deferred"],
            "next_queries": [],
            "sufficient": False,
            "fixed_point": True,
        }

    fake = _fake_agentic(research_domain)
    amplifier.install_parallel_core(fake)
    result = fake.collect_pre_design_research(_native_router(), "boss")

    assert attempts == {"mk_entity": 2, "mk_quality": 2}
    assert [note["domain_id"] for note in result["domain_notes"]] == [
        "mk_entity",
        "mk_quality",
    ]
    assert all(note.get("fixed_point") is True for note in result["domain_notes"])
    assert all("worker_error" not in note for note in result["domain_notes"])


def test_persistent_parallel_failure_is_not_replayed() -> None:
    attempts: dict[str, int] = {}

    def research_domain(router, *, prompt, domain, deterministic, trace_metadata):
        domain_id = domain["domain_id"]
        attempts[domain_id] = attempts.get(domain_id, 0) + 1
        raise RuntimeError("persistent research failure")

    fake = _fake_agentic(research_domain)
    amplifier.install_parallel_core(fake)
    result = fake.collect_pre_design_research(_native_router(), "boss")

    assert attempts == {"mk_entity": 1, "mk_quality": 1}
    assert all(note["sufficient"] is False for note in result["domain_notes"])
    assert all(note["worker_error"] is True for note in result["domain_notes"])
    assert all(not note.get("fixed_point", False) for note in result["domain_notes"])
    assert all("parallel_error" in note for note in result["domain_notes"])
    assert all("retry_error" not in note for note in result["domain_notes"])


def test_single_slot_failure_is_not_replayed() -> None:
    attempts: dict[str, int] = {}

    def research_domain(router, *, prompt, domain, deterministic, trace_metadata):
        domain_id = domain["domain_id"]
        attempts[domain_id] = attempts.get(domain_id, 0) + 1
        raise RuntimeError("deterministic validation failure")

    fake = _fake_agentic(research_domain)
    amplifier.install_parallel_core(fake)
    result = fake.collect_pre_design_research(object(), "boss")

    assert attempts == {"mk_entity": 1, "mk_quality": 1}
    assert all(note["sufficient"] is False for note in result["domain_notes"])
    assert all(note["worker_error"] is True for note in result["domain_notes"])
    assert all(
        bool(note.get("serial_error") or note.get("parallel_error"))
        for note in result["domain_notes"]
    )
    assert all("retry_error" not in note for note in result["domain_notes"])


def test_fixed_point_recovery_is_accepted_only_with_real_route_receipts() -> None:
    plan = compile_minecraft_knowledge_plan("새 보스 몬스터를 추가해줘.")
    domains = list(plan["research_domains"])
    research = {
        "research_brief": {"domains": domains},
        "deterministic": {
            "forced_project_rag": {
                "domains": [
                    {
                        "domain_id": domain["domain_id"],
                        "queries": [
                            {
                                "query": query,
                                "query_sha256": "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest(),
                            }
                            for query in domain["queries"]
                        ],
                    }
                    for domain in domains
                ]
            }
        },
        "domain_notes": [
            {
                "domain_id": domain["domain_id"],
                "claims": [],
                "gaps": ["deferred"],
                "next_queries": [],
                "sufficient": False,
                "fixed_point": True,
            }
            for domain in domains
        ],
    }
    assert evaluate_route_coverage(plan, research)["status"] == "PASS"

    research["deterministic"]["forced_project_rag"]["domains"].pop()
    assert evaluate_route_coverage(plan, research)["status"] == "BLOCK"
