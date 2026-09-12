from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

from minecraft_mod_ai import planning_state_research as planning_research
from minecraft_mod_ai import pre_design_grounded_rag as project_rag
from minecraft_mod_ai import pre_design_research_pipeline as research_pipeline
from minecraft_mod_ai import task_template_batch_runner as batch_runner
from minecraft_mod_ai.model_adapters import LlamaCppAdapter


def _native_parallel_router() -> SimpleNamespace:
    config = SimpleNamespace(
        exclusive_gpu=True,
        provider="local",
        adapter="llama_cpp",
    )
    return SimpleNamespace(
        profile="test",
        registry=SimpleNamespace(role=lambda _profile, _role: config),
    )


def test_record_templates_use_measured_native_slots_and_preserve_context(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")
    router = _native_parallel_router()
    active = 0
    max_active = 0
    lock = threading.Lock()
    target = ContextVar("template_target", default="missing")
    seen: list[str] = []

    def run_one(_router, identifier, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            seen.append(target.get())
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"identifier": identifier}

    monkeypatch.setattr(batch_runner, "run_bounded_record_template", run_one)
    identifiers = ("feature/a", "feature/b", "feature/c")
    assert batch_runner.supports_record_batching(router) is True
    assert batch_runner.record_template_batches(identifiers) == (identifiers,)

    token = target.set("fabric-1.21.8")
    try:
        result = batch_runner.run_record_template_batch(
            router,
            identifiers,
            context={"requirement": "test"},
            allowed_refs=set(),
        )
    finally:
        target.reset(token)

    assert list(result) == list(identifiers)
    assert max_active >= 2
    assert seen == ["fabric-1.21.8"] * 3


def test_planning_research_domains_run_in_parallel_but_merge_deterministically(
    monkeypatch,
) -> None:
    router = SimpleNamespace()
    active = 0
    max_active = 0
    lock = threading.Lock()
    target = ContextVar("research_target", default="missing")
    seen_targets: list[str] = []

    monkeypatch.setattr(planning_research, "validate_planning_state", lambda *_a, **_k: None)
    monkeypatch.setattr(planning_research, "_rehash", lambda value: value)
    monkeypatch.setattr(planning_research, "router_native_model_parallelism", lambda _router: 2)
    monkeypatch.setattr(planning_research, "forced_rag_bundle", lambda *_a, **_k: {})
    monkeypatch.setattr(planning_research, "project_repository_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        planning_research,
        "merge_repository_candidates",
        lambda current, added: [*current, *added],
    )
    monkeypatch.setattr(
        research_pipeline,
        "_grounded_domain_evidence",
        lambda domain_id, _bundle: {"domain_id": domain_id, "queries": []},
    )
    monkeypatch.setattr(
        research_pipeline,
        "_validate_document_grounding",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        project_rag,
        "_materialize_domain_evidence_document",
        lambda domain_id, _payload: {"domain_id": domain_id},
    )

    def research_document(_agentic, _project_rag, _router, *, domain, **_kwargs):
        nonlocal active, max_active
        domain_id = str(domain["domain_id"])
        with lock:
            active += 1
            max_active = max(max_active, active)
            seen_targets.append(target.get())
        time.sleep(0.03)
        with lock:
            active -= 1
        return {
            "domain_id": domain_id,
            "claims": [
                {
                    "claim": f"claim-{domain_id}",
                    "evidence_refs": [f"evidence:{domain_id}"],
                }
            ],
            "sufficient": True,
            "source_body_count": 1,
            "host_grounded_evidence_card_count": 1,
            "evidence_extraction_status": "complete",
            "research_failures": [],
        }

    monkeypatch.setattr(planning_research, "research_document_domain", research_document)

    state = {
        "repository_candidates": [],
        "references": [],
        "unresolved": [],
        "research_queue": [
            {
                "research_id": "r1",
                "status": "pending",
                "objective": "first",
                "information_needed": "first evidence",
                "source_kinds": ["minecraft_docs"],
                "queries": ["first query"],
                "resolves": [],
            },
            {
                "research_id": "r2",
                "status": "pending",
                "objective": "second",
                "information_needed": "second evidence",
                "source_kinds": ["minecraft_docs"],
                "queries": ["second query"],
                "resolves": [],
            },
        ],
        "evidence": [],
        "resolved": [],
        "blockers": [],
    }

    token = target.set("planning-trace")
    try:
        result = planning_research.collect_planning_state_research(
            router,
            "test prompt",
            state,
        )
    finally:
        target.reset(token)

    assert max_active >= 2
    assert seen_targets == ["planning-trace", "planning-trace"]
    assert [row["research_ref"] for row in result["evidence"]] == ["r1", "r2"]
    assert [row["status"] for row in result["research_queue"]] == ["complete", "complete"]


def test_llama_prefill_cache_is_shared_across_request_local_adapters() -> None:
    cache = LlamaCppAdapter._prefill_template_prefix_cache
    assert isinstance(cache, dict)
    first = object.__new__(LlamaCppAdapter)
    second = object.__new__(LlamaCppAdapter)
    assert getattr(first, "_prefill_template_prefix_cache") is cache
    assert getattr(second, "_prefill_template_prefix_cache") is cache
