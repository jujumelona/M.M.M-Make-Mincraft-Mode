from __future__ import annotations

from minecraft_mod_ai import complete_spec, planner_template_schema
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.implementation_kind_boundary_contract import (
    _is_evidence_owned,
    _route_evidence_owned_custom,
)


def _evidence_config() -> dict:
    return {
        "evidence_plan_sha256": "sha256:" + "a" * 64,
        "requirement_refs": ["REQ-QUEST"],
        "evidence_task": {
            "task_id": "task_quest",
            "requirement_refs": ["REQ-QUEST"],
            "semantic_outcome": "Player can complete the authored quest behavior.",
        },
    }


def test_evidence_owned_semantic_kind_routes_to_custom_without_changing_authority():
    module = ProductionModule(
        module_id="quest_flow",
        kind="quest",
        config=_evidence_config(),
    )

    routed = _route_evidence_owned_custom(module, ProductionModule)

    assert routed.kind == "custom_java"
    assert routed.config["requested_kind"] == "quest"
    assert routed.config["implementation_classifier_role"] == "routing_hint_only"
    assert routed.config["semantic_authority"] == "evidence_task"
    assert routed.config["evidence_task"] == module.config["evidence_task"]
    assert routed.depends_on == module.depends_on
    assert routed.required_gates == module.required_gates


def test_legacy_semantic_kind_is_not_reinterpreted_by_boundary():
    module = ProductionModule(
        module_id="legacy_quest",
        kind="quest",
        config={"objective": "manual"},
    )

    assert not _is_evidence_owned(module)
    assert _route_evidence_owned_custom(module, ProductionModule) is module


def test_integration_identity_is_preserved_when_evidence_owned():
    module = ProductionModule(
        module_id="integration_feature",
        kind="integration",
        config={**_evidence_config(), "integration_type": "external_contract"},
    )

    assert _route_evidence_owned_custom(module, ProductionModule) is module


def test_installed_template_uses_canonical_execution_classifier_catalog():
    assert planner_template_schema.MODULE_KINDS is complete_spec.MODULE_KINDS
    assert planner_template_schema.ASSET_KINDS is complete_spec.ASSET_KINDS
    assert complete_spec.IMPLEMENTATION_KINDS is complete_spec.MODULE_KINDS
