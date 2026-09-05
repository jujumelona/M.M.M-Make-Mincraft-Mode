from __future__ import annotations

import json
from typing import Any

import pytest

from minecraft_mod_ai.research_derived_requirements import (
    FACETS,
    ResearchRequirementError,
    _model_requirement_facets,
)
from minecraft_mod_ai.research_requirement_plan_slice import host_facet_baseline


class _StructuredRequirementRouter:
    def __init__(self, *, invalid_facet: bool = False) -> None:
        self.invalid_facet = invalid_facet
        self.calls: list[dict[str, Any]] = []

    def generate_text(
        self,
        role: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "kwargs": kwargs,
            }
        )
        payload = json.loads(messages[-1]["content"])
        baseline = payload["host_baseline"]
        facets = [dict(item) for item in baseline]
        if self.invalid_facet:
            facets[0]["facet"] = "platform_loader_constraint"
        return json.dumps({"facets": facets})


def _requirement() -> dict[str, Any]:
    return {
        "requirement_id": "req_spacecraft_upgrade",
        "capability": "spacecraft.performance_upgrade",
        "statement": "Upgrade spacecraft performance through trade and purchases.",
        "implementation_capabilities": [
            "economy.transaction_service",
            "persistence.spacecraft_state",
            "network.server_authority",
        ],
        "acceptance": ["an upgrade changes the authoritative spacecraft state"],
    }


def _tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "task_upgrade_service",
            "requirement_refs": ["req_spacecraft_upgrade"],
            "semantic_outcome": (
                "Implement server-authoritative upgrade transaction service and state transition"
            ),
            "consumes": ["spacecraft_schema:upgrade"],
            "provides": ["spacecraft_service:upgrade"],
            "required_gates": ["source_static_validation", "target_compile"],
            "acceptance": ["invalid or insufficient-cost upgrades are rejected"],
            "owned_anchors": [
                {"kind": "symbol", "locator": "src/main/java/X.java#X"},
                {"kind": "test", "locator": "src/test/java/XTest.java#XTest"},
            ],
        },
        {
            "task_id": "task_upgrade_state",
            "requirement_refs": ["req_spacecraft_upgrade"],
            "semantic_outcome": "Persist, reload, and synchronize spacecraft upgrade state",
            "consumes": ["spacecraft_service:upgrade"],
            "provides": ["persistent_state:upgrade"],
            "required_gates": ["target_compile"],
            "acceptance": ["upgrade state survives reload"],
            "owned_anchors": [
                {"kind": "symbol", "locator": "src/main/java/Y.java#Y"},
                {"kind": "test", "locator": "src/test/java/YTest.java#YTest"},
            ],
        },
    ]


def _evidence() -> list[dict[str, Any]]:
    return [
        {
            "evidence_ref": "evidence:upgrade00000001",
            "path": "technical_evidence.upgrade[0]",
            "summary": {
                "requirement_ref": "req_spacecraft_upgrade",
                "capability": "spacecraft.performance_upgrade",
                "claim": (
                    "server network persistence verification evidence for the exact upgrade requirement"
                ),
                "loader": "fabric",
                "minecraft_version": "1.21.1",
            },
        }
    ]


def test_research_facets_use_one_fixed_structured_requirement_template() -> None:
    router = _StructuredRequirementRouter()
    requirement = _requirement()
    tasks = _tasks()
    baseline = host_facet_baseline(requirement, tasks)
    refs = {facet: ("evidence:upgrade00000001",) for facet in FACETS}

    rows = _model_requirement_facets(
        router,
        requirement=requirement,
        tasks=tasks,
        baseline=baseline,
        evidence_window=_evidence(),
        relevant_refs=refs,
    )

    assert list(rows) == list(FACETS)
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["role"] == "planner"
    assert call["kwargs"]["response_format"] == "json"
    assert call["kwargs"]["enable_tools"] is False
    schema = call["kwargs"]["response_schema"]
    assert schema["properties"]["facets"]["minItems"] == len(FACETS)
    assert schema["properties"]["facets"]["maxItems"] == len(FACETS)

    payload = json.loads(call["messages"][-1]["content"])
    assert len(payload["host_baseline"]) == len(FACETS)
    assert len(payload["evidence_catalog"]) <= 12
    assert {item["task_id"] for item in payload["host_task_slice"]} == {
        "task_upgrade_service",
        "task_upgrade_state",
    }
    assert all(
        facet in payload["facet_relevant_evidence_refs"]
        for facet in FACETS
    )


def test_model_cannot_smuggle_an_eighth_facet() -> None:
    router = _StructuredRequirementRouter(invalid_facet=True)
    requirement = _requirement()
    tasks = _tasks()
    baseline = host_facet_baseline(requirement, tasks)

    with pytest.raises(ResearchRequirementError, match="invalid or duplicate facet"):
        _model_requirement_facets(
            router,
            requirement=requirement,
            tasks=tasks,
            baseline=baseline,
            evidence_window=_evidence(),
            relevant_refs={facet: ("evidence:upgrade00000001",) for facet in FACETS},
        )


def test_host_baseline_is_complete_without_any_model_turn() -> None:
    baseline = host_facet_baseline(_requirement(), _tasks())
    assert list(baseline) == list(FACETS)
    assert all(
        item["disposition"] in {"already_covered", "not_applicable"}
        for item in baseline.values()
    )
    assert baseline["persistence_reload"]["disposition"] == "already_covered"
    assert baseline["server_network_authority"]["disposition"] == "already_covered"
    assert baseline["failure_edge_cases"]["disposition"] == "already_covered"
    assert baseline["verification_testing"]["disposition"] == "already_covered"
