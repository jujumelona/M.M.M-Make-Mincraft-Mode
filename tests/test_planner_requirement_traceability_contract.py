from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.reuse_planner import compile_pre_retrieval_plan, validate_pre_retrieval_plan


def _design() -> dict[str, object]:
    return {
        "_evidence_request_catalog": {
            "catalog_sha256": "sha256:test-catalog",
            "purpose": "trace authored requirements",
            "requirements": [
                {
                    "requirement_id": "req_alpha",
                    "capability": "resource.mining",
                    "provides": ["capability:resource.mining"],
                    "semantic_statement": "mine lunar crystal",
                    "acceptance": ["player can mine lunar crystal"],
                    "depends_on": [],
                },
                {
                    "requirement_id": "req_beta",
                    "capability": "economy.trade",
                    "provides": ["capability:economy.trade"],
                    "semantic_statement": "trade credits at colony market",
                    "acceptance": ["player can trade credits"],
                    "depends_on": ["req_alpha"],
                },
            ],
        }
    }


def test_each_planned_work_item_keeps_exact_authored_requirement_identity(monkeypatch) -> None:
    from minecraft_mod_ai import evidence_first_planning

    monkeypatch.setattr(evidence_first_planning, "_validate_request_catalog", lambda *_args, **_kwargs: None)
    design = _design()
    plan = compile_pre_retrieval_plan("trace requirements", design)

    assert [item["requirement_ref"] for item in plan["planned_work"]] == [
        "req_alpha",
        "req_beta",
    ]
    by_ref = {item["requirement_ref"]: item for item in plan["planned_work"]}
    assert by_ref["req_beta"]["depends_on"] == [by_ref["req_alpha"]["work_id"]]


def test_unknown_authored_dependency_fails_closed(monkeypatch) -> None:
    from minecraft_mod_ai import evidence_first_planning

    monkeypatch.setattr(evidence_first_planning, "_validate_request_catalog", lambda *_args, **_kwargs: None)
    design = _design()
    design["_evidence_request_catalog"]["requirements"][1]["depends_on"] = ["req_missing"]
    with pytest.raises(ValueError, match="unknown dependencies"):
        compile_pre_retrieval_plan("trace requirements", design)


def test_tampering_with_requirement_binding_invalidates_plan(monkeypatch) -> None:
    from minecraft_mod_ai import evidence_first_planning

    monkeypatch.setattr(evidence_first_planning, "_validate_request_catalog", lambda *_args, **_kwargs: None)
    design = _design()
    plan = compile_pre_retrieval_plan("trace requirements", design)
    plan["planned_work"][0]["requirement_ref"] = "req_beta"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_pre_retrieval_plan(plan, prompt="trace requirements", design=design)


def test_retired_requirement_traceability_monkeypatch_owner_is_absent() -> None:
    package = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    assert not (package / "planner_requirement_traceability_contract.py").exists()
