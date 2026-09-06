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
    ) -> None:
        self.segment_calls: list[list[dict[str, Any]]] = []
        self.classify_calls: list[list[dict[str, Any]]] = []
        self.foreign_source_index = foreign_source_index
        if batch_receipt is not None:
            self.semantic_extraction_batch_receipt = batch_receipt

    def generate_tool_decision(
        self,
        role: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert role == "planner"
        payload = json.loads(messages[-1]["content"])
        tool_name = kwargs["tool_name"]

        if tool_name == "segment_semantic_requirements":
            clauses = list(payload["host_owned_clauses"])
            self.segment_calls.append(clauses)
            leaves: list[dict[str, Any]] = []
            for clause in clauses:
                source_index = int(clause["source_clause_index"])
                emitted_index = 999 if self.foreign_source_index else source_index
                text = str(clause["text"]).strip()
                leaves.append(
                    {
                        "source_clause_index": emitted_index,
                        "source_anchor": text,
                        "semantic_statement": f"Implement authored clause {source_index}",
                        "given": f"authored state for clause {source_index}",
                        "when": f"the player performs clause {source_index}",
                        "then": f"clause {source_index} outcome is observable",
                        "semantic_type": "gameplay_mechanic",
                    }
                )
            return {"leaves": leaves}

        if tool_name == "classify_semantic_requirements":
            leaves = list(payload["host_grounded_leaves"])
            self.classify_calls.append(leaves)
            return {
                "classifications": [
                    {
                        "leaf_index": int(leaf["leaf_index"]),
                        "capability_id": "custom.semantic",
                    }
                    for leaf in leaves
                ]
            }

        raise AssertionError(f"unexpected semantic tool: {tool_name}")


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


def test_unmeasured_profile_uses_single_clause_batches_without_claiming_optimality():
    clauses = _clauses()
    router = SemanticRouter()

    catalog = build_bounded_requirement_catalog(PROMPT, router=router)
    audit = catalog["semantic_audit"]

    assert len(router.segment_calls) == len(clauses)
    assert len(router.classify_calls) == len(clauses)
    assert all(len(call) == 1 for call in router.segment_calls)
    assert all(len(call) == 1 for call in router.classify_calls)
    assert audit["semantic_batch_size"] == 1
    assert audit["semantic_batch_count"] == len(clauses)
    assert audit["semantic_batch_parallel_workers"] == 1
    assert audit["semantic_batch_size_measured"] is False
    assert audit["semantic_batch_size_source"] == "unmeasured_conservative_single_clause"
    assert audit["semantic_generation_protocol"] == "segment_then_host_ground_then_classify"
    assert audit["semantic_model_calls_total_observed"] == len(clauses) * 2
    assert "optimal" not in audit["semantic_batch_size_source"]

    receipts = audit["semantic_batches"]
    assert len(receipts) == len(clauses)
    expected = {int(item["clause_index"]): item for item in clauses}
    for receipt in receipts:
        assert receipt["semantic_model_calls_total"] == 2
        assert len(receipt["source_clauses"]) == 1
        source = receipt["source_clauses"][0]
        clause = expected[int(source["source_clause_index"])]
        assert source["char_start"] == clause["char_start"]
        assert source["char_end"] == clause["char_end"]
        assert source["text_sha256"] == clause["text_sha256"]


def test_measured_profile_bounds_two_stage_calls_and_records_runtime_receipts():
    clauses = _clauses()
    router = SemanticRouter(batch_receipt=_measured_receipt(2))

    catalog = build_bounded_requirement_catalog(PROMPT, router=router)
    audit = catalog["semantic_audit"]
    batch_count = math.ceil(len(clauses) / 2)

    assert len(router.segment_calls) == batch_count
    assert len(router.classify_calls) == batch_count
    assert all(1 <= len(call) <= 2 for call in router.segment_calls)
    assert all(1 <= len(call) <= 2 for call in router.classify_calls)
    assert audit["semantic_batch_size"] == 2
    assert audit["semantic_batch_count"] == batch_count
    assert audit["semantic_model_calls_total_observed"] == batch_count * 2
    assert audit["semantic_batch_size_measured"] is True
    assert audit["semantic_batch_size_source"] == "measured_model_runtime_receipt"
    assert audit["semantic_batch_model_identity_sha256"] == "sha256:" + "a" * 64
    assert audit["semantic_batch_runtime_profile_sha256"] == "sha256:" + "b" * 64
    assert audit["semantic_batch_benchmark_receipt_sha256"] == "sha256:" + "c" * 64


def test_independent_batches_use_proven_slots_and_merge_in_source_order(monkeypatch):
    clauses = _clauses()[:2]
    barrier = threading.Barrier(2)

    def fake_compile(router: Any, batch):
        barrier.wait(timeout=2.0)
        clause = batch[0]
        clause_index = int(clause["clause_index"])
        return [
            {
                "source_clause_index": clause_index,
                "source_start": int(clause["char_start"]),
                "capability_id": "custom.semantic",
                "semantic_statement": f"clause {clause_index}",
            }
        ], {
            "segmentation_attempts": 1,
            "classification_attempts": 1,
            "semantic_model_calls_total": 2,
            "semantic_repair_turns_used": 0,
            "segmentation_repaired": False,
            "classification_repaired": False,
        }

    monkeypatch.setattr(batching, "compile_semantic_batch", fake_compile)
    monkeypatch.setattr(batching, "router_native_model_parallelism", lambda router: 2)
    monkeypatch.setattr(batching._semantic, "_assign_local_ids", lambda nodes: list(nodes))

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

    def fake_compile(router: Any, batch):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        with lock:
            active -= 1
        clause = batch[0]
        return [
            {
                "source_clause_index": int(clause["clause_index"]),
                "source_start": int(clause["char_start"]),
                "capability_id": "custom.semantic",
                "semantic_statement": "bounded",
            }
        ], {
            "segmentation_attempts": 1,
            "classification_attempts": 1,
            "semantic_model_calls_total": 2,
            "semantic_repair_turns_used": 0,
            "segmentation_repaired": False,
            "classification_repaired": False,
        }

    monkeypatch.setattr(batching, "compile_semantic_batch", fake_compile)
    monkeypatch.setattr(batching, "router_native_model_parallelism", lambda router: 1)
    monkeypatch.setattr(batching._semantic, "_assign_local_ids", lambda nodes: list(nodes))

    batching._generate_bounded_nodes(object(), clauses, batch_size=1)
    assert peak == 1


def test_planning_compiler_preserves_bounded_two_stage_protocol():
    clauses = _clauses()
    router = SemanticRouter(batch_receipt=_measured_receipt(2))

    catalog = planning_authority._compile_semantic_catalog(PROMPT, router)
    audit = catalog["semantic_audit"]
    batch_count = math.ceil(len(clauses) / 2)

    assert audit["semantic_generation_protocol"] == "bounded_host_catalog_classification_batches"
    assert audit["semantic_model_turns"] == batch_count * 2
    assert audit["semantic_base_stage_calls_per_batch"] == 2
    assert audit["feature_dependency_owner"] == "host_minecraft_feature_model"


def test_measured_batch_receipt_without_runtime_identity_fails_closed():
    receipt = _measured_receipt(2)
    receipt["runtime_profile_sha256"] = "missing"
    router = SemanticRouter(batch_receipt=receipt)

    with pytest.raises(semantic._evidence.EvidencePlanError, match="runtime_profile_sha256"):
        build_bounded_requirement_catalog(PROMPT, router=router)


def test_model_cannot_emit_a_source_index_outside_its_current_batch():
    router = SemanticRouter(foreign_source_index=True)

    with pytest.raises(semantic._evidence.EvidencePlanError, match="REQ_SOURCE_CLAUSE"):
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
