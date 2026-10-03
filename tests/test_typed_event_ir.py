from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
from minecraft_mod_ai.typed_plan_java import render_typed_plan_java
from minecraft_mod_ai.typed_plan_support import typed_plan_support_issues


def _structured_entry_point(trigger: str) -> dict:
    specification = {
        name: []
        for name in DETAIL_RECORDS["integration"]
    }
    specification["entry_points"] = [
        {
            "boundary": "server",
            "trigger": trigger,
            "owner": "server",
        }
    ]
    specification["inapplicable_concerns"] = []
    return {
        "integration": {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }


def _plan(*, parameters: list[dict], event_bindings: list[dict]) -> dict:
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(b"event plan").hexdigest(),
        "functions": [
            {
                "id": "handleEvent",
                "parameters": parameters,
                "return_type": "void",
                "body": [{"op": "return"}],
                "covers": ["integration.entry_points"],
            }
        ],
        "initialize": [],
        "event_bindings": event_bindings,
    }


def test_server_started_binding_is_host_validated_and_rendered() -> None:
    plan = _plan(
        parameters=[{"name": "server", "type": "object"}],
        event_bindings=[
            {
                "event": "server_started",
                "function": "handleEvent",
                "entry_point_index": 0,
                "config": {},
            }
        ],
    )
    structured = _structured_entry_point("server startup")

    validated = validate_typed_plan_ir(plan)
    assert typed_plan_support_issues(structured, validated) == ()

    source = render_typed_plan_java(validated, package="fixture")
    assert (
        "ServerLifecycleEvents.SERVER_STARTED.register("
        "server -> fn_handleEvent(server));"
    ) in source


def test_runtime_entry_point_without_event_binding_fails_support_gate() -> None:
    plan = _plan(
        parameters=[{"name": "server", "type": "object"}],
        event_bindings=[],
    )
    structured = _structured_entry_point("server startup")

    validated = validate_typed_plan_ir(plan)
    issues = typed_plan_support_issues(structured, validated)

    assert len(issues) == 1
    assert issues[0].startswith("integration.entry_points[0].trigger=")


def test_mod_initialize_uses_host_scaffold_without_duplicate_event_binding() -> None:
    plan = _plan(
        parameters=[],
        event_bindings=[],
    )
    structured = _structured_entry_point("mod initialize")

    validated = validate_typed_plan_ir(plan)
    assert typed_plan_support_issues(structured, validated) == ()


def test_event_binding_rejects_wrong_function_signature() -> None:
    plan = _plan(
        parameters=[],
        event_bindings=[
            {
                "event": "server_started",
                "function": "handleEvent",
                "entry_point_index": 0,
                "config": {},
            }
        ],
    )

    with pytest.raises(ValueError, match="signature"):
        validate_typed_plan_ir(plan)


def test_command_binding_uses_fixed_host_api_shape() -> None:
    plan = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(b"command plan").hexdigest(),
        "functions": [
            {
                "id": "runCommand",
                "parameters": [{"name": "source", "type": "object"}],
                "return_type": "int",
                "body": [
                    {
                        "op": "return",
                        "value": {"op": "literal", "type": "int", "value": 1},
                    }
                ],
                "covers": ["integration.entry_points"],
            }
        ],
        "initialize": [],
        "event_bindings": [
            {
                "event": "command",
                "function": "runCommand",
                "entry_point_index": 0,
                "config": {"literal": "typedtest", "permission_level": 2},
            }
        ],
    }

    source = render_typed_plan_java(plan, package="fixture")
    assert "CommandRegistrationCallback.EVENT.register" in source
    assert ".requires(source -> source.hasPermissionLevel(2))" in source
    assert ".executes(context -> fn_runCommand(context.getSource()))" in source
