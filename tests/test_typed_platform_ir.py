from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
from minecraft_mod_ai.typed_plan_support import typed_plan_support_issues


def _base_plan() -> dict:
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(b"typed platform").hexdigest(),
        "functions": [
            {
                "id": "noop",
                "parameters": [],
                "return_type": "void",
                "body": [{"op": "return"}],
                "covers": ["behavior_contract.outputs"],
            }
        ],
        "initialize": [],
    }


def _section(section: str, concern: str, rows: list[dict]) -> dict:
    specification = {
        name: []
        for name in DETAIL_RECORDS[section]
    }
    specification[concern] = rows
    specification["inapplicable_concerns"] = []
    return {
        section: {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }


def test_content_platform_module_covers_real_resource_concern() -> None:
    plan = _base_plan()
    plan["platform_modules"] = [
        {
            "module_id": "marker_item",
            "kind": "item",
            "config": {"display_name_en": "Marker"},
            "covers": ["resources_and_ui.registries"],
        }
    ]
    structured = _section(
        "resources_and_ui",
        "registries",
        [
            {
                "purpose": "item",
                "identifier": "marker_item",
                "binding_requirement": "register one marker item",
            }
        ],
    )

    validated = validate_typed_plan_ir(plan)

    assert validated["platform_modules"][0]["kind"] == "item"
    assert typed_plan_support_issues(structured, validated) == ()


def test_platform_kind_cannot_claim_unimplemented_network_concern() -> None:
    plan = _base_plan()
    plan["platform_modules"] = [
        {
            "module_id": "marker_item",
            "kind": "item",
            "config": {},
            "covers": ["authority_and_network.packets"],
        }
    ]

    with pytest.raises(ValueError, match="cannot implement"):
        validate_typed_plan_ir(plan)


def test_platform_coverage_must_reference_active_canonical_concern() -> None:
    plan = _base_plan()
    plan["platform_modules"] = [
        {
            "module_id": "marker_item",
            "kind": "item",
            "config": {},
            "covers": ["resources_and_ui.registries"],
        }
    ]
    validated = validate_typed_plan_ir(plan)

    issues = typed_plan_support_issues({}, validated)

    assert issues == (
        "platform.coverage_without_active_concern:resources_and_ui.registries",
    )


def test_state_store_is_the_generic_persistence_backend() -> None:
    plan = _base_plan()
    plan["platform_modules"] = [
        {
            "module_id": "persistent_state",
            "kind": "state_store",
            "config": {"namespace": "player_state"},
            "covers": ["persistence.stored_state"],
        }
    ]
    structured = _section(
        "persistence",
        "stored_state",
        [
            {
                "state": "credits",
                "owner": "player",
                "scope": "world",
            }
        ],
    )

    validated = validate_typed_plan_ir(plan)

    assert typed_plan_support_issues(structured, validated) == ()
