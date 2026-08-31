from __future__ import annotations

import pytest

from minecraft_mod_ai.planner_requirement_traceability_contract import (
    enforce_requirement_design_traceability,
)
from minecraft_mod_ai.spec import SpecValidationError


def _plan(*, source: str, label: str, detail: str) -> dict:
    capability = "design.module.test" if source.startswith("game_design.modules") else "design.loop.test"
    return {
        "schema_version": "mmm/pre-retrieval-semantic-plan-v1",
        "planned_work": [
            {
                "work_id": "work_alpha",
                "requirement_ref": "req_alpha",
                "objective": "mine lunar crystal",
                "capabilities": ["resource.mining", capability],
                "acceptance": ["player can mine lunar crystal"],
            },
            {
                "work_id": "work_beta",
                "requirement_ref": "req_beta",
                "objective": "trade credits at colony market",
                "capabilities": ["economy.trade"],
                "acceptance": ["player can trade credits"],
            },
        ],
        "capability_graph": {
            "nodes": ["resource.mining", "economy.trade", capability],
            "edges": [{"from": "resource.mining", "to": capability}],
            "sources": [],
            "search_terms": [],
        },
        "design_retrieval_facets": [
            {
                "capability": capability,
                "work_id": "work_alpha",
                "requirement_ref": "req_alpha",
                "source": source,
                "label": label,
                "detail": detail,
            }
        ],
        "plan_sha256": "placeholder",
    }


def _capabilities_by_requirement(plan: dict) -> dict[str, list[str]]:
    return {
        str(item["requirement_ref"]): list(item["capabilities"])
        for item in plan["planned_work"]
    }


def test_explicit_module_requirement_refs_override_wrong_prior_owner() -> None:
    plan = _plan(
        source="game_design.modules[0]",
        label="colony market",
        detail="implement colony market transactions",
    )
    design = {
        "modules": [
            {
                "plugin_id": "colony_market",
                "requirement_refs": ["req_beta"],
                "implementation_obligations": ["persist offers and execute credit trades"],
            }
        ]
    }

    traced = enforce_requirement_design_traceability(plan, design)
    owned = _capabilities_by_requirement(traced)

    assert "design.module.test" not in owned["req_alpha"]
    assert "design.module.test" in owned["req_beta"]
    assert traced["design_retrieval_facets"] == [
        {
            "capability": "design.module.test",
            "work_id": "work_beta",
            "requirement_ref": "req_beta",
            "source": "game_design.modules[0]",
            "label": "colony market",
            "detail": "implement colony market transactions",
            "binding_basis": "explicit_module_requirement_refs",
        }
    ]
    assert {tuple(edge.values()) for edge in traced["capability_graph"]["edges"]} == {
        ("economy.trade", "design.module.test")
    }


def test_unrelated_facet_is_preserved_under_all_requirements_not_positionally_guessed() -> None:
    plan = _plan(
        source="game_design.core_loop[0]",
        label="ritual",
        detail="activate the astral beacon",
    )

    traced = enforce_requirement_design_traceability(plan, {"modules": []})
    owned = _capabilities_by_requirement(traced)

    assert "design.loop.test" in owned["req_alpha"]
    assert "design.loop.test" in owned["req_beta"]
    rows = traced["design_retrieval_facets"]
    assert {row["requirement_ref"] for row in rows} == {"req_alpha", "req_beta"}
    assert {row["binding_basis"] for row in rows} == {"conservative_all_requirements"}
    assert {tuple(edge.values()) for edge in traced["capability_graph"]["edges"]} == {
        ("resource.mining", "design.loop.test"),
        ("economy.trade", "design.loop.test"),
    }


def test_positive_lexical_evidence_binds_without_fallback_order() -> None:
    plan = _plan(
        source="game_design.progression[0]",
        label="market trade",
        detail="trade credits at colony market",
    )

    traced = enforce_requirement_design_traceability(plan, {"modules": []})
    owned = _capabilities_by_requirement(traced)

    assert "design.loop.test" not in owned["req_alpha"]
    assert "design.loop.test" in owned["req_beta"]
    assert traced["design_retrieval_facets"][0]["binding_basis"] == "positive_lexical_evidence"


def test_unknown_explicit_requirement_ref_fails_closed() -> None:
    plan = _plan(
        source="game_design.modules[0]",
        label="colony market",
        detail="implement colony market transactions",
    )
    design = {
        "modules": [
            {
                "plugin_id": "colony_market",
                "requirement_refs": ["req_missing"],
                "implementation_obligations": ["execute trades"],
            }
        ]
    }

    with pytest.raises(SpecValidationError, match="unknown requirement ids"):
        enforce_requirement_design_traceability(plan, design)
