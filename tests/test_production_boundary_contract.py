from __future__ import annotations

import pytest

from minecraft_mod_ai import production_boundary_contract as boundary
from minecraft_mod_ai.production_contract import ProductionContractError


def _plan():
    return {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_trade",
                    "capability": "economy.trade",
                    "semantic_statement": "Trade gathered resources.",
                    "source_span": {"text": "trade gathered resources"},
                    "acceptance": [
                        "Given gathered resources; when the player trades them; then the agreed exchange is observable."
                    ],
                },
                {
                    "requirement_id": "req_travel",
                    "capability": "world.travel",
                    "semantic_statement": "Travel to another world.",
                    "source_span": {"text": "travel to another world"},
                    "acceptance": [
                        "Given travel is unlocked; when the player departs; then they arrive at the selected destination."
                    ],
                },
            ]
        },
        "tasks": [
            {
                "task_id": "task_trade_ui",
                "requirement_refs": ["req_trade"],
                "conditional_predicates": ["needs_client_render"],
                "artifact_obligations": [
                    {"kind": "client_visual_or_ui_resource"}
                ],
            },
            {
                "task_id": "task_travel",
                "requirement_refs": ["req_travel"],
                "conditional_predicates": [],
                "artifact_obligations": [],
            },
        ],
    }


def test_public_acceptance_is_exactly_the_approved_requirement_contract():
    approved = boundary._approved_requirements(_plan())
    assert boundary._approved_acceptance(approved["req_trade"]) == (
        "Given gathered resources; when the player trades them; then the agreed exchange is observable."
    )


def test_internal_task_language_is_rejected_from_public_acceptance():
    requirement = {
        "requirement_id": "req_bad",
        "acceptance": [
            "task_bad: all declared provides exist and owned anchors pass integrity checks"
        ],
    }
    with pytest.raises(
        ProductionContractError,
        match="has no safe public acceptance projection",
    ):
        boundary._approved_acceptance(requirement)


def test_conditional_quality_is_bound_only_to_triggering_requirement():
    active = {
        "correctness",
        "build",
        "research",
        "runtime",
        "visual_3d",
        "multiplayer",
        "state_save_migration",
    }
    trade = boundary._conditional_dimensions(_plan(), "req_trade", active)
    travel = boundary._conditional_dimensions(_plan(), "req_travel", active)

    assert "visual_3d" in trade
    assert "multiplayer" not in trade
    assert "state_save_migration" not in trade
    assert travel == []


def test_requirements_keep_single_authoritative_ids():
    approved = boundary._approved_requirements(_plan())
    assert set(approved) == {"req_trade", "req_travel"}
    assert all(not value.startswith("requirement:") for value in approved)


def test_canonical_public_acceptance_preserves_multiple_checks_per_requirement():
    checks = ["Given A; when B; then C.", "Given D; when E; then F."]
    assert boundary._approved_acceptance(
        {
            "requirement_id": "req_many",
            "acceptance": checks,
        }
    ) == "; ".join(checks)


def test_space_economy_regression_keeps_all_public_acceptance_checks():
    checks = [
        "플레이어가 자원을 채굴하거나 수집할 수 있는 메커니즘이 존재한다.",
        "플레이어가 자원을 통해 돈을 획득할 수 있는 메커니즘이 존재한다.",
        "플레이어가 다른 플레이어 또는 NPC와 자원을 거래할 수 있는 인터페이스가 존재한다.",
    ]
    projected = boundary._approved_acceptance(
        {
            "requirement_id": "req_001",
            "acceptance": checks,
        }
    )
    assert projected == "; ".join(checks)
    assert all(check in projected for check in checks)
