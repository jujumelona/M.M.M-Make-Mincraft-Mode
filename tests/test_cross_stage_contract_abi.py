"""Cross-stage ABI checks using real production registries, without model or Gradle.

Unlike fixture-only tests, these pass the CURRENT host-owned producer data into
the CURRENT consumer and reject unknown/mistyped metadata. Keep this lane fast.
"""
from __future__ import annotations

import copy
import hashlib

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.authored_content_contract import (
    CONTENT_CONCERN_MINIMUM_ENTITY_COUNT,
    CONTENT_GRAPH_DRIVER_CONCERNS,
    GAMEPLAY_CONTENT_DRIVER_CONCERNS,
)
from minecraft_mod_ai.authored_reuse_bridge import (
    authored_capability_graph,
    materialize_verified_authored_sources,
    resolve_authored_source_reuse,
)
from minecraft_mod_ai.content_design_contract import (
    CONTENT_FACT_TO_PRODUCTION_KIND,
    CONTENT_KIND_TO_FACT_TYPE,
    CONTENT_CONCERN_KINDS,
)
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.typed_event_ir import (
    EVENT_PARAMETERS,
    EVENT_SIGNATURES,
    infer_event_type,
    validate_event_bindings,
)
from minecraft_mod_ai.typed_host_capabilities import (
    render_typed_host_capabilities_java,
    typed_host_capability_contracts,
)
from minecraft_mod_ai.typed_plan_authoring import semantic_dispatch_schema
from minecraft_mod_ai.typed_plan_ir import (
    validate_typed_host_capability_contracts,
    validate_typed_plan_ir,
)
from minecraft_mod_ai.typed_platform_ir import PLATFORM_KINDS, platform_config_schema


def _minimal_ir():
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(b"contract preflight").hexdigest(),
        "functions": [{
            "id": "example", "parameters": [], "return_type": "int",
            "body": [{"op": "return", "value": {
                "op": "literal", "type": "int", "value": 1,
            }}],
            "covers": [],
        }],
        "initialize": [],
    }


def test_production_capability_producer_matches_validator_and_java_renderers():
    actual = typed_host_capability_contracts()
    normalized = validate_typed_host_capability_contracts(actual)
    assert validate_typed_plan_ir(_minimal_ir(), capabilities=actual)
    assert set(actual) == set(normalized)
    assert normalized["player.grant_item"]["gameplay_mutation"] is True
    assert normalized["player.add_status_effect"]["gameplay_mutation"] is True
    assert normalized["player.send_message"]["gameplay_mutation"] is False

    for version in ("1.21.1", "26.2"):
        java = render_typed_host_capabilities_java(
            "test.contract", minecraft_version=version,
        )
        for contract in actual.values():
            assert " " + contract["method"] + "(" in java, (version, contract)


def test_production_capability_metadata_fail_closed_and_mutation_routing():
    actual = typed_host_capability_contracts()
    bad = copy.deepcopy(actual)
    bad["player.grant_item"]["gameplay_mutation"] = "true"
    with pytest.raises(ValueError, match="gameplay_mutation"):
        validate_typed_host_capability_contracts(bad)

    bad = copy.deepcopy(actual)
    bad["player.grant_item"]["invented_java_binding"] = "foo"
    with pytest.raises(ValueError, match="unknown fields"):
        validate_typed_host_capability_contracts(bad)

    schema = semantic_dispatch_schema(
        {}, actual, allowed_events=("command",), mutation_only=True,
    )
    branches = schema["oneOf"]
    ids = {
        branch["properties"]["capability_id"]["const"]
        for branch in branches
        if "capability_id" in branch["properties"]
    }
    assert ids == {
        cap for cap, contract in actual.items()
        if contract.get("gameplay_mutation") is True
    }
    assert all(
        branch["properties"]["trigger_event"]["enum"] == ["command"]
        for branch in branches
    )


def test_event_producer_signatures_match_host_validator():
    assert set(EVENT_PARAMETERS) == set(EVENT_SIGNATURES)
    bindings = []
    for index, (event, fields) in enumerate(EVENT_PARAMETERS.items()):
        expected = tuple(type_name for _, type_name in fields)
        assert EVENT_SIGNATURES[event][0] == expected
        assert infer_event_type(event) == event
        bindings.append({
            "event": event, "function": "handle_" + event,
            "entry_point_index": index,
            "config": {"literal": "test_command"} if event == "command" else {},
        })
    signatures = {
        "handle_" + event: signature
        for event, signature in EVENT_SIGNATURES.items()
    }
    validated = validate_event_bindings(bindings, signatures=signatures)
    assert len(validated) == len(bindings)
    assert {row["event"] for row in validated} == set(EVENT_PARAMETERS)


def test_content_driver_concerns_are_real_canonical_template_records():
    for section, concerns in GAMEPLAY_CONTENT_DRIVER_CONCERNS.items():
        assert section in DETAIL_RECORDS
        assert set(concerns) <= set(DETAIL_RECORDS[section])
    assert set(CONTENT_GRAPH_DRIVER_CONCERNS) <= set(
        DETAIL_RECORDS["resources_and_ui"]
    )
    assert set(CONTENT_CONCERN_MINIMUM_ENTITY_COUNT) <= set(
        DETAIL_RECORDS["resources_and_ui"]
    )
    assert set(CONTENT_GRAPH_DRIVER_CONCERNS) <= set(CONTENT_CONCERN_KINDS)
    assert set(CONTENT_KIND_TO_FACT_TYPE.values()) <= set(
        CONTENT_FACT_TO_PRODUCTION_KIND
    )


def test_platform_authoring_schemas_are_json_schema_valid():
    for kind in PLATFORM_KINDS:
        Draft202012Validator.check_schema(platform_config_schema(kind))


def test_reuse_bridge_producer_receipt_matches_materialization_boundary(monkeypatch, tmp_path):
    from minecraft_mod_ai import grounded_source_reuse

    captured = []

    def no_network_reuse(design):
        captured.append(design)
        graph = design["_pre_retrieval_plan"]["capability_graph"]
        return {
            "schema_version": "mmm/grounded-repository-reuse-plan-v2",
            "source_plan_sha256": design["_pre_retrieval_plan"]["plan_sha256"],
            "capability_graph": graph,
            "capabilities": [
                {"capability": cap, "mode": "fresh"}
                for cap in graph["nodes"]
            ],
        }

    monkeypatch.setattr(grounded_source_reuse, "build_repository_reuse_plan", no_network_reuse)
    result = resolve_authored_source_reuse(
        "Spaceship fuel and colony trading", {},
        minecraft_version="26.2", loader="fabric",
    )
    assert captured
    assert result["bound_target"] == {
        "minecraft_version": "26.2", "loader": "fabric",
    }
    staged = materialize_verified_authored_sources(
        str(tmp_path), result,
        minecraft_version="26.2", loader="fabric",
    )
    assert staged["donor_count"] == 0
    with pytest.raises(ValueError, match="SOURCE_REUSE_TARGET_MISMATCH"):
        materialize_verified_authored_sources(
            str(tmp_path), result,
            minecraft_version="1.21.1", loader="fabric",
        )
    assert authored_capability_graph("build spaceship", {})["nodes"]
