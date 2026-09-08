from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
from minecraft_mod_ai.complete_spec import ProductionModule
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
