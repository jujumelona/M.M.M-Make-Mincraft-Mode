from __future__ import annotations

import copy

import pytest

from minecraft_mod_ai.implementation_template_contract import (
    SCHEMA,
    _validate_contract,
    build_implementation_template,
)


def _task() -> dict[str, object]:
    return {
        "task_id": "task_ship_runtime",
        "task_sha256": "sha256:" + "a" * 64,
        "sequence": 3,
        "execution_role": "production_with_verification",
        "semantic_outcome": "The player can launch the spacecraft.",
        "requirement_refs": ["req_launch"],
        "depends_on": ["task_build_ship"],
        "target_cell": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "yarn",
            "java_version": "21",
        },
        "owned_anchors": [
            {
                "kind": "symbol",
                "locator": "src/main/java/demo/ShipRuntime.java#ShipRuntime",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "main",
            },
            {
                "kind": "test",
                "locator": "src/gametest/demo/ShipRuntimeGameTest.java#ShipRuntimeGameTest",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "gametest",
            },
        ],
        "implementation_capabilities": ["space.launch"],
        "implementation_obligations": [
            "Bind the launch action to authoritative server-side ship state."
        ],
        "consumes": ["ship_ready"],
        "provides": ["ship_launched"],
        "required_gates": ["source_static_validation", "target_compile", "gametest"],
        "public_acceptance": ["The player can launch a completed spacecraft."],
        "runtime_acceptance": ["Launch changes authoritative runtime state."],
        "artifact_obligations": [
            {
                "kind": "source_code",
                "locator": "src/main/java/demo/ShipRuntime.java#ShipRuntime",
                "purpose": "runtime launch implementation",
            }
        ],
    }


def test_coder_contract_is_complete_host_owned_v2() -> None:
    contract = build_implementation_template(_task())

    assert contract["schema_version"] == SCHEMA == "mmm/coder-execution-contract-v2"
    assert contract["task_ref"] == "task_ship_runtime"
    assert contract["depends_on"] == ["task_build_ship"]
    assert contract["target_constraints"] == {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "yarn",
        "mappings_applicable": True,
        "naming_regime": "mapped_obfuscated",
        "java_version": "21",
        "policy": "Use only the immutable host-selected target and compatible evidence.",
    }
    assert [target["path"] for target in contract["targets"]] == [
        "src/main/java/demo/ShipRuntime.java",
        "src/gametest/demo/ShipRuntimeGameTest.java",
    ]
    assert contract["targets"][0]["symbol"] == "ShipRuntime"
    assert contract["targets"][0]["operation"] == "create_or_modify"
    assert contract["protected_boundaries"]["writable_paths"] == [
        "src/main/java/demo/ShipRuntime.java",
        "src/gametest/demo/ShipRuntimeGameTest.java",
    ]
    assert contract["completion_predicate"]["operator"] == "all"
    assert contract["completion_predicate"]["model_self_report_is_authoritative"] is False
    _validate_contract(contract)


def test_native_target_keeps_blank_mappings_in_coder_contract() -> None:
    task = _task()
    task["target_cell"] = {
        "minecraft_version": "26.2",
        "loader": "fabric",
        "mappings": "",
        "mappings_applicable": False,
        "java_version": "25",
    }
    contract = build_implementation_template(task)
    assert contract["target_constraints"]["mappings"] == ""
    assert contract["target_constraints"]["mappings_applicable"] is False
    assert contract["target_constraints"]["naming_regime"] == "native_unobfuscated"


def test_coder_steps_are_deterministic_and_bind_to_exact_targets() -> None:
    first = build_implementation_template(_task())
    second = build_implementation_template(copy.deepcopy(_task()))
    assert first == second
    assert first["contract_sha256"] == second["contract_sha256"]
    assert [step["sequence"] for step in first["implementation_steps"]] == list(range(len(first["implementation_steps"])))
    target_refs = [target["locator"] for target in first["targets"]]
    assert all(step["target_refs"] == target_refs for step in first["implementation_steps"])
    assert all(step["consumes"] == ["ship_ready"] for step in first["implementation_steps"])
    assert all(step["must_provide"] == ["ship_launched"] for step in first["implementation_steps"])


def test_verification_plan_carries_host_gates_and_observable_acceptance() -> None:
    contract = build_implementation_template(_task())
    gates = [item["gate"] for item in contract["verification_plan"]]
    assert gates == ["source_static_validation", "target_compile", "gametest", "observable_acceptance"]
    observable = contract["verification_plan"][-1]
    assert observable["executor"] == "host_acceptance_runner"
    assert observable["public_acceptance"] == ["The player can launch a completed spacecraft."]
    assert observable["runtime_acceptance"] == ["Launch changes authoritative runtime state."]


def test_contract_rejects_target_or_hash_tampering() -> None:
    contract = build_implementation_template(_task())
    tampered = copy.deepcopy(contract)
    tampered["targets"][0]["path"] = "src/main/java/evil/Outside.java"
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_contract(tampered)


def test_contract_requires_exact_owned_target() -> None:
    task = _task()
    task["owned_anchors"] = []
    with pytest.raises(ValueError, match="no owned target anchor"):
        build_implementation_template(task)


def test_semantic_only_task_is_rejected_instead_of_becoming_code_step() -> None:
    task = _task()
    task["implementation_capabilities"] = []
    task["implementation_obligations"] = []
    task["artifact_obligations"] = []
    task["design_resolution_obligations"] = []
    with pytest.raises(ValueError, match="semantic-only"):
        build_implementation_template(task)
