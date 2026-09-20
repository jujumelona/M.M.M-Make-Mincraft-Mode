from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.small_model_atomic_coder_execution import atomicize_coder_messages
from minecraft_mod_ai.small_model_task_capsule_contract import compile_task_capsule


def test_debug_fixture_uses_real_custom_task_contract(tmp_path: Path) -> None:
    plan_path = write_debug_example_plan(
        tmp_path / "proposal.json",
        minecraft_version="1.21.8",
        loader="fabric",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(payload["modules"]) == 1

    raw = payload["modules"][0]
    assert raw["module_id"] == "debug_token"
    assert raw["kind"] == "custom_java"
    assert raw["required_gates"] == ["target_compile"]

    config = raw["config"]
    task = config["evidence_task"]
    assert task["task_id"] == raw["module_id"]
    assert task["engineering_worksheet"]
    assert task["implementation_obligations"]
    assert task["production_bindings"]
    assert config["coder_execution_contract"]["task_ref"] == task["task_id"]

    module = ProductionModule(
        module_id=raw["module_id"],
        kind=raw["kind"],
        config=config,
        depends_on=tuple(raw.get("depends_on") or ()),
        required_gates=tuple(raw.get("required_gates") or ()),
    )
    capsule = compile_task_capsule(module)
    assert capsule is not None
    assert capsule.task_id == "debug_token"
    assert capsule.primary_path == "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    assert capsule.primary_symbol == "DebugToken"
    assert capsule.required_gates == ("target_compile",)
    assert capsule.reuse_action == "fresh"


def test_debug_fixture_constraint_steps_compile_to_one_coder_state_transition(
    tmp_path: Path,
) -> None:
    plan_path = write_debug_example_plan(
        tmp_path / "proposal.json",
        minecraft_version="1.21.8",
        loader="fabric",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    raw = payload["modules"][0]
    task = dict(raw["config"]["evidence_task"])
    task["coder_execution_contract"] = raw["config"]["coder_execution_contract"]
    messages = (
        {"role": "system", "content": "coder"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "task": "Implement the approved debug fixture.",
                    "module": {
                        "module_id": raw["module_id"],
                        "kind": raw["kind"],
                        "evidence_task": task,
                    },
                }
            ),
        },
    )

    batches = atomicize_coder_messages(messages)

    implementation_steps = task["coder_execution_contract"]["implementation_steps"]
    assert len(implementation_steps) == len(task["implementation_obligations"])
    assert len(implementation_steps) == 6
    assert len(batches) == 1
    lowered = json.loads(batches[0][-1]["content"])
    assert lowered["atomic_execution"]["step_count"] == 1
    obligation = lowered["module"]["evidence_task"]["coder_execution_contract"]["step"][
        "obligation"
    ]
    assert "host-grounded item registration API" in obligation
    assert "owned DebugToken.java target" in obligation
    assert "Do not implement ModInitializer" in obligation


def test_debug_target_compile_is_source_owned_and_satisfied_by_real_build_receipt(
    tmp_path: Path,
) -> None:
    plan_path = write_debug_example_plan(
        tmp_path / "proposal.json",
        minecraft_version="1.21.8",
        loader="fabric",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    from minecraft_mod_ai.complete_spec import CompleteProposal

    proposal = CompleteProposal.from_dict(payload)
    proposal.validate()
    failures = CompleteProductionOrchestrator._required_gate_failures(
        proposal,
        generated_receipts=(),
        project_root=tmp_path,
        source_validation={"status": "PASS"},
        jdt_receipt=None,
        build_report={
            "status": "PASS",
            "commands": [
                {
                    "name": "build",
                    "exit_code": 0,
                    "timed_out": False,
                }
            ],
        },
        jar_validation=None,
        blockbench_receipts=(),
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )

    assert failures == []
