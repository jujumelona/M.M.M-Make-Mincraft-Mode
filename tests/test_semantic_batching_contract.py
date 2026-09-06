from __future__ import annotations

import json
import math
import threading
from typing import Any

import pytest

from minecraft_mod_ai import evidence_request_guard
from minecraft_mod_ai import planning_authority
from minecraft_mod_ai import semantic_batching_contract as batching
from minecraft_mod_ai import semantic_requirement_authority as semantic
from minecraft_mod_ai.semantic_batching_contract import build_bounded_requirement_catalog


PROMPT = "Collect lunar ore.\nBuild a spacecraft.\nLaunch the spacecraft."


class SemanticRouter:
    def __init__(
        self,
        *,
        batch_receipt: dict[str, Any] | None = None,
        foreign_source_index: bool = False,
        malformed: bool = False,
    ) -> None:
        self.compile_calls: list[list[dict[str, Any]]] = []
        self.foreign_source_index = foreign_source_index
        self.malformed = malformed
        if batch_receipt is not None:
            self.semantic_extraction_batch_receipt = batch_receipt

    def generate_tool_decision(
        self,
        role: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert role == "planner"
        assert kwargs["tool_name"] == "compile_semantic_requirements"
        payload = json.loads(messages[-1]["content"])
        clauses = list(payload["host_owned_clauses"])
        self.compile_calls.append(clauses)
        if self.malformed:
            return {"unexpected": []}
        requirements: list[dict[str, Any]] = []
        for clause in clauses:
            source_index = int(clause["source_clause_index"])
            emitted_index = 999 if self.foreign_source_index else source_index
            text = str(clause["text"]).strip()
            requirements.append(
                {
                    "source_clause_index": emitted_index,
                    "capability_id": "custom.semantic",
                    "source_anchor": text,
                    "semantic_statement": text,
                    "given": f"authored state for clause {source_index}",
                    "when": f"the player performs clause {source_index}",
                    "then": f"clause {source_index} outcome is observable",
                    "semantic_type": "gameplay_mechanic",
                }
            )
        return {"requirements": requirements}


def _clauses() -> list[dict[str, Any]]:
    clauses = semantic._clause_records(PROMPT)
    assert len(clauses) >= 3
    return clauses


def _measured_receipt(size: int) -> dict[str, Any]:
    return {
        "status": "MEASURED",
        "max_clauses_per_turn": size,
        "model_identity_sha256": "sha256:" + "a" * 64,
        "runtime_profile_sha256": "sha256:" + "b" * 64,
        "benchmark_receipt_sha256": "sha256:" + "c" * 64,
    }


def test_unmeasured_profile_uses_one_single_pass_call_per_clause_batch():
    clauses = _clauses()
    router = SemanticRouter()

    catalog = build_bounded_requirement_catalog(PROMPT, router=router)
    audit = catalog["semantic_audit"]

    assert len(router.compile_calls) == len(clauses)
    assert all(len(call) == 1 for call in router.compile_calls)
    assert audit["semantic_batch_size"] == 1
    assert audit["semantic_batch_count"] == len(clauses)
    assert audit["semantic_batch_parallel_workers"] == 1
    assert audit["semantic_batch_size_measured"] is False
    assert audit["semantic_batch_size_source"] == "unmeasured_conservative_single_clause"
    assert audit["semantic_generation_protocol"] == (
        "host_clause_defaults_then_one_structured_refinement"
    )
    assert audit["semantic_model_calls_total_observed"] == len(clauses)
    assert audit["semantic_repair_turns_used"] == 0
    assert audit["semantic_base_stage_calls_per_batch"] == 1

    receipts = audit["semantic_batches"]
    assert len(receipts) == len(clauses)
    assert all(receipt["semantic_model_calls_total"] == 1 for receipt in receipts)
    assert all(receipt["host_fallback_count"] == 0 for receipt in receipts)


def test_measured_profile_bounds_single_pass_calls_and_records_runtime_receipts():
    clauses = _clauses()
    router = SemanticRouter(batch_receipt=_measured_receipt(2))

    catalog = build_bounded_requirement_catalog(PROMPT, router=router)
    audit = catalog["semantic_audit"]
    batch_count = math.ceil(len(clauses) / 2)

    assert len(router.compile_calls) == batch_count
    assert all(1 <= len(call) <= 2 for call in router.compile_calls)
    assert audit["semantic_batch_size"] == 2
    assert audit["semantic_batch_count"] == batch_count
    assert audit["semantic_model_calls_total_observed"] == batch_count
    assert audit["semantic_batch_size_measured"] is True
    assert audit["semantic_batch_model_identity_sha256"] == "sha256:" + "a" * 64
    assert audit["semantic_batch_runtime_profile_sha256"] == "sha256:" + "b" * 64
    assert audit["semantic_batch_benchmark_receipt_sha256"] == "sha256:" + "c" * 64


def test_malformed_model_output_keeps_host_default_requirements():
    clauses = _clauses()
    catalog = build_bounded_requirement_catalog(PROMPT, router=SemanticRouter(malformed=True))
    audit = catalog["semantic_audit"]

    assert len(catalog["requirements"]) == len(clauses)
    assert audit["semantic_host_fallback_count"] == len(clauses)
    assert all(
        requirement["template_profile"]["model_capability_choice"] == "custom.semantic"
        for requirement in catalog["requirements"]
    )


def test_foreign_source_index_is_ignored_and_host_clause_survives():
    clauses = _clauses()
    catalog = build_bounded_requirement_catalog(
        PROMPT, router=SemanticRouter(foreign_source_index=True)
    )
    audit = catalog["semantic_audit"]

    assert len(catalog["requirements"]) == len(clauses)
    assert audit["semantic_host_fallback_count"] == len(clauses)
    assert {
        requirement["source_span"]["source_clause_index"]
        for requirement in catalog["requirements"]
    } == {int(clause["clause_index"]) for clause in clauses}


def test_independent_batches_use_proven_slots_and_merge_in_source_order(monkeypatch):
    clauses = _clauses()[:2]
    barrier = threading.Barrier(2)

    def fake_compile(router: Any, batch_index: int, batch):
        barrier.wait(timeout=2.0)
        node = batching._host_default_node(batch[0], ordinal=0)
        return [node], {
            "batch_index": batch_index,
            "semantic_model_calls_total": 1,
            "semantic_repair_turns_used": 0,
            "compound_demotions": 0,
            "non_executable_drops": 0,
            "host_fallback_count": 1,
        }

    monkeypatch.setattr(batching, "_compile_bounded_batch", fake_compile)
    monkeypatch.setattr(batching, "router_native_model_parallelism", lambda router: 2)
    nodes, receipts = batching._generate_bounded_nodes(object(), clauses, batch_size=1)

    assert [item["source_clause_index"] for item in nodes] == [
        int(clauses[0]["clause_index"]),
        int(clauses[1]["clause_index"]),
    ]
    assert [item["batch_index"] for item in receipts] == [0, 1]


def test_unproven_router_never_gets_parallel_semantic_batches(monkeypatch):
    clauses = _clauses()[:2]
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_compile(router: Any, batch_index: int, batch):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            active -= 1
        return [batching._host_default_node(batch[0], ordinal=0)], {
            "batch_index": batch_index,
            "semantic_model_calls_total": 1,
            "semantic_repair_turns_used": 0,
            "compound_demotions": 0,
            "non_executable_drops": 0,
            "host_fallback_count": 1,
        }

    monkeypatch.setattr(batching, "_compile_bounded_batch", fake_compile)
    monkeypatch.setattr(batching, "router_native_model_parallelism", lambda router: 1)
    batching._generate_bounded_nodes(object(), clauses, batch_size=1)
    assert peak == 1


def test_planning_compiler_preserves_single_pass_protocol():
    clauses = _clauses()
    router = SemanticRouter(batch_receipt=_measured_receipt(2))

    catalog = planning_authority._compile_semantic_catalog(PROMPT, router)
    audit = catalog["semantic_audit"]
    batch_count = math.ceil(len(clauses) / 2)

    assert audit["semantic_model_turns"] == batch_count
    assert audit["semantic_base_stage_calls_per_batch"] == 1
    assert audit["feature_dependency_owner"] == "host_minecraft_feature_model"
    assert audit["semantic_repair_turns_used"] == 0


def test_measured_batch_receipt_without_runtime_identity_fails_closed():
    receipt = _measured_receipt(2)
    receipt["runtime_profile_sha256"] = "missing"
    router = SemanticRouter(batch_receipt=receipt)

    with pytest.raises(semantic._evidence.EvidencePlanError, match="runtime_profile_sha256"):
        build_bounded_requirement_catalog(PROMPT, router=router)


def test_reconciliation_rechecks_static_owner_after_initial_install(monkeypatch):
    batching.install_semantic_batching_contract()

    def unbounded_replacement(prompt: str, router: Any | None = None) -> dict[str, Any]:
        return {"prompt": prompt, "router": router}

    monkeypatch.setattr(
        evidence_request_guard,
        "build_authoritative_request_catalog",
        unbounded_replacement,
    )

    with pytest.raises(RuntimeError, match="not statically wired"):
        batching.install_semantic_batching_contract()
