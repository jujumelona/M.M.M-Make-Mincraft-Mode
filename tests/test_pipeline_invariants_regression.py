from __future__ import annotations

import inspect

from minecraft_mod_ai import evidence_obligation_contract as obligations
from minecraft_mod_ai import reuse_planner


def _catalog(count: int = 2) -> dict:
    return {
        "schema_version": "mmm/approved-requirement-graph-v1",
        "catalog_sha256": "sha256:test",
        "requirements": [
            {
                "requirement_id": f"req_{index}",
                "capability": f"capability_{index}",
                "semantic_statement": f"implement capability {index}",
            }
            for index in range(count)
        ],
    }


def test_pre_target_freeze_only_retrieves_target_neutral_reuse_evidence():
    brief = obligations.build_evidence_obligation_brief(
        "test",
        _catalog(),
        {"_platform_selection": {"target_frozen": False}},
    )
    nodes = brief["evidence_obligation_dag"]["nodes"]
    assert len(nodes) == 2
    assert {node["kind"] for node in nodes} == {"reusable_implementation"}
    assert brief["target_frozen"] is False
    deferred = set(brief["deferred_obligation_kinds"])
    assert {
        "target_compatibility",
        "implementation_api",
        "dependency_closure",
        "license_provenance",
        "validation_mechanism",
    } <= deferred


def test_frozen_target_expands_complete_obligation_dag():
    brief = obligations.build_evidence_obligation_brief(
        "test",
        _catalog(1),
        {"_platform_selection": {"target_frozen": True}},
    )
    nodes = brief["evidence_obligation_dag"]["nodes"]
    assert len(nodes) == len(obligations._OBLIGATIONS)
    assert {node["kind"] for node in nodes} == {
        str(spec["kind"]) for spec in obligations._OBLIGATIONS
    }
    assert brief["target_frozen"] is True
    assert brief["deferred_obligation_kinds"] == []


def test_reuse_planner_does_not_gate_grounded_donors_on_public_discovery():
    source = inspect.getsource(reuse_planner.optimize_platform_and_reuse)
    assert "grounded_donors_available" in source
    assert "__mmm_grounded_donors__" in source
    assert "client if evidence_discovery_enabled else None" in source
