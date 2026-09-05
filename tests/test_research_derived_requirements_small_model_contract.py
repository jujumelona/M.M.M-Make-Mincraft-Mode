from __future__ import annotations

import json
from typing import Any

from minecraft_mod_ai.research_derived_requirements import (
    FACETS,
    _model_facet_augmentation,
)
from minecraft_mod_ai.research_requirement_plan_slice import host_facet_baseline
from minecraft_mod_ai.research_requirement_template import (
    FACET_AUGMENTATION_RESPONSE_SCHEMA,
)


class _BoundedFacetRouter:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_text(
        self,
        role: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        return json.dumps(self.response)


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
            "semantic_outcome": "Implement server-authoritative upgrade transaction service",
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


def _slot() -> dict[str, Any]:
    return {
        "schema_version": "mmm/research-requirement-slot-v1",
        "host_owned": {
            "facet": "server_network_authority",
            "host_baseline": {
                "disposition": "already_covered",
                "acceptance": ["server owns upgrade mutation"],
            },
            "host_task_slice": _tasks(),
            "allowed_evidence_refs": ["evidence:upgrade00000001"],
            "evidence_catalog": [
                {
                    "evidence_ref": "evidence:upgrade00000001",
                    "summary": {"claim": "server validates upgrade requests"},
                }
            ],
        },
        "model_slot": {
            "question": "Does evidence add a missing server-authority obligation?",
            "allowed_decisions": [
                "no_addition",
                "add_obligation",
                "insufficient_evidence",
            ],
        },
    }


def test_research_model_receives_exactly_one_bounded_facet_slot() -> None:
    router = _BoundedFacetRouter(
        {
            "decision": "no_addition",
            "rationale": "The host baseline already covers the evidence-backed rule.",
            "evidence_refs": ["evidence:upgrade00000001"],
            "implementation_obligations": [],
            "acceptance": [],
        }
    )
    candidate, calls, events = _model_facet_augmentation(
        router,
        parent="req_spacecraft_upgrade",
        facet="server_network_authority",
        slot=_slot(),
        allowed_refs=["evidence:upgrade00000001"],
    )

    assert calls == 1
    assert candidate is not None
    assert candidate["decision"] == "no_addition"
    assert events[-1]["status"] == "accepted"
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["role"] == "planner"
    assert call["kwargs"]["response_format"] == "json"
    assert call["kwargs"]["enable_tools"] is False
    assert call["kwargs"]["response_schema"] is FACET_AUGMENTATION_RESPONSE_SCHEMA
    payload = json.loads(call["messages"][1]["content"])
    assert payload["host_owned"]["facet"] == "server_network_authority"
    assert "facets" not in payload


def test_disallowed_evidence_cannot_escape_the_facet_slot() -> None:
    router = _BoundedFacetRouter(
        {
            "decision": "add_obligation",
            "rationale": "Invented evidence should be rejected.",
            "evidence_refs": ["evidence:not-allowed"],
            "implementation_obligations": ["Do an unsupported thing."],
            "acceptance": ["Unsupported behavior occurs."],
        }
    )
    candidate, calls, events = _model_facet_augmentation(
        router,
        parent="req_spacecraft_upgrade",
        facet="server_network_authority",
        slot=_slot(),
        allowed_refs=["evidence:upgrade00000001"],
    )

    assert candidate is None
    assert calls == 2
    assert events[-1]["status"] == "fallback_host_baseline"


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
