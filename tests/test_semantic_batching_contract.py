from __future__ import annotations

import json
import math
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
        last_clause_index: int,
        batch_receipt: dict[str, Any] | None = None,
        foreign_source_index: bool = False,
    ) -> None:
        self.last_clause_index = last_clause_index
        self.calls: list[list[dict[str, Any]]] = []
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
        assert kwargs["tool_name"] == "compile_semantic_requirements"
        payload = json.loads(messages[-1]["content"])
        clauses = list(payload["host_owned_clauses"])
        self.calls.append(clauses)

        requirements: list[dict[str, Any]] = []
        for clause in clauses:
            source_index = int(clause["source_clause_index"])
            emitted_index = 999 if self.foreign_source_index else source_index
            capability = f"feature.clause_{source_index}"
            required = []
            if source_index == self.last_clause_index and source_index != 0:
                required = ["feature.clause_0"]
            text = str(clause["text"])
            requirements.append(
                {
                    "source_clause_index": emitted_index,
                    "capability_id": capability,
                    "source_anchor": text.strip(),
                    "semantic_statement": f"Implement authored clause {source_index}",
                    "given": f"authored state for clause {source_index}",
                    "when": f"the player performs clause {source_index}",
                    "then": f"clause {source_index} outcome is observable",
                    "semantic_type": "gameplay_mechanic",
                    "required_prerequisite_capabilities": required,
                    "optional_prerequisite_capabilities": [],
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


def test_unmeasured_profile_uses_single_clause_batches_without_claiming_optimality():
    clauses = _clauses()
    router = SemanticRouter(last_clause_index=int(clauses[-1]["clause_index"]))

    catalog = build_bounded_requirement_catalog(PROMPT, router=router)
    audit = catalog["semantic_audit"]

    assert len(router.calls) == len(clauses)
    assert all(len(call) == 1 for call in router.calls)
    assert audit["semantic_batch_size"] == 1
    assert audit["semantic_batch_count"] == len(clauses)
    assert audit["semantic_batch_size_measured"] is False
    assert audit["semantic_batch_size_source"] == "unmeasured_conservative_single_clause"
    assert audit["semantic_generation_protocol"] == "bounded_host_owned_semantic_batches"
    assert "optimal" not in audit["semantic_batch_size_source"]

    receipts = audit["semantic_batches"]
    assert len(receipts) == len(clauses)
    expected = {int(item["clause_index"]): item for item in clauses}
    for receipt in receipts:
        assert len(receipt["source_clauses"]) == 1
        source = receipt["source_clauses"][0]
        clause = expected[int(source["source_clause_index"])]
        assert source["char_start"] == clause["char_start"]
        assert source["char_end"] == clause["char_end"]
        assert source["text_sha256"] == clause["text_sha256"]


def test_measured_profile_bounds_calls_and_records_exact_runtime_receipts():
    clauses = _clauses()
    router = SemanticRouter(
        last_clause_index=int(clauses[-1]["clause_index"]),
        batch_receipt=_measured_receipt(2),
    )

    catalog = build_bounded_requirement_catalog(PROMPT, router=router)
    audit = catalog["semantic_audit"]

    assert len(router.calls) == math.ceil(len(clauses) / 2)
    assert all(1 <= len(call) <= 2 for call in router.calls)
    assert audit["semantic_batch_size"] == 2
    assert audit["semantic_batch_count"] == math.ceil(len(clauses) / 2)
    assert audit["semantic_batch_size_measured"] is True
    assert audit["semantic_batch_size_source"] == "measured_model_runtime_receipt"
    assert audit["semantic_batch_model_identity_sha256"] == "sha256:" + "a" * 64
    assert audit["semantic_batch_runtime_profile_sha256"] == "sha256:" + "b" * 64
    assert audit["semantic_batch_benchmark_receipt_sha256"] == "sha256:" + "c" * 64


def test_cross_batch_required_capability_is_resolved_after_global_merge():
    clauses = _clauses()
    router = SemanticRouter(last_clause_index=int(clauses[-1]["clause_index"]))

    catalog = build_bounded_requirement_catalog(PROMPT, router=router)
    by_capability = {item["capability"]: item for item in catalog["requirements"]}
    first = by_capability["feature.clause_0"]
    last = by_capability[f"feature.clause_{clauses[-1]['clause_index']}"]

    assert first["requirement_id"] in last["depends_on"]
    assert [first["requirement_id"], last["requirement_id"]] in catalog[
        "requirement_graph"
    ]["edges"]
    assert catalog["semantic_audit"]["cross_batch_prerequisite_reconciliation"] == (
        "global_catalog_capability_resolution"
    )


def test_planning_compiler_preserves_bounded_protocol_instead_of_one_call_claim():
    clauses = _clauses()
    router = SemanticRouter(
        last_clause_index=int(clauses[-1]["clause_index"]),
        batch_receipt=_measured_receipt(2),
    )

    catalog = planning_authority._compile_semantic_catalog(PROMPT, router)
    audit = catalog["semantic_audit"]

    assert audit["semantic_generation_protocol"] == "bounded_host_owned_semantic_batches"
    assert audit["semantic_generation_protocol"] != "all_clauses_one_structured_batch"
    assert audit["semantic_model_turns"] == math.ceil(len(clauses) / 2)
    assert audit["global_dependency_reconciliation"] == (
        "global_catalog_capability_resolution_then_host_causal_dag"
    )


def test_measured_batch_receipt_without_runtime_identity_fails_closed():
    clauses = _clauses()
    receipt = _measured_receipt(2)
    receipt["runtime_profile_sha256"] = "missing"
    router = SemanticRouter(
        last_clause_index=int(clauses[-1]["clause_index"]),
        batch_receipt=receipt,
    )

    with pytest.raises(semantic._evidence.EvidencePlanError, match="runtime_profile_sha256"):
        build_bounded_requirement_catalog(PROMPT, router=router)


def test_model_cannot_emit_a_source_index_outside_its_current_batch():
    clauses = _clauses()
    router = SemanticRouter(
        last_clause_index=int(clauses[-1]["clause_index"]),
        foreign_source_index=True,
    )

    with pytest.raises(
        semantic._evidence.EvidencePlanError,
        match="rejected bounded-batch model output",
    ):
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
