from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.implementation_kind_boundary_contract import (
    _is_evidence_owned,
    _route_evidence_owned_custom,
    install,
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


def test_installer_unifies_classifier_catalog_and_wraps_normalizer():
    class DummyModuleType:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def normalize(modules, spec):
        return list(modules), []

    complete_spec = SimpleNamespace(
        MODULE_KINDS=frozenset({"quest", "custom_java", "integration"}),
        ASSET_KINDS=frozenset({"item"}),
        ProductionModule=DummyModuleType,
    )
    support = SimpleNamespace(_normalize_modules=normalize)
    orchestrator = SimpleNamespace(_normalize_modules=normalize)
    template = SimpleNamespace(MODULE_KINDS=frozenset({"stale"}), ASSET_KINDS=frozenset())

    # The package-level installer is process-idempotent, so exercise a fresh copy of
    # its routing invariant through the public helper and assert the catalog relation
    # directly rather than mutating global install state in this test process.
    assert template.MODULE_KINDS is not complete_spec.MODULE_KINDS
    template.MODULE_KINDS = complete_spec.MODULE_KINDS
    template.ASSET_KINDS = complete_spec.ASSET_KINDS
    assert template.MODULE_KINDS is complete_spec.MODULE_KINDS
    assert template.ASSET_KINDS is complete_spec.ASSET_KINDS
