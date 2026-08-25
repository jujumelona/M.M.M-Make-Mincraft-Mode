from __future__ import annotations

import json
import shutil
from pathlib import Path

from minecraft_mod_ai import model_router
from minecraft_mod_ai.generated_counterexample_tests import (
    build_generated_test_spec,
    install_generated_junit,
    run_generated_test_spec,
)
from minecraft_mod_ai.procedural_memory_hierarchy import build_hierarchy
from minecraft_mod_ai.trajectory_memory import build_work_trajectory


def _verified_receipt() -> dict[str, object]:
    return {
        "procedure": [
            {"tool": "inspect_existing_mod", "status": "PASS"},
            {"tool": "search_code_rag", "status": "PASS"},
            {
                "tool": "apply_source_patch",
                "status": "PASS",
                "content": "SECRET SOURCE MUST NOT ENTER PROCEDURE MEMORY",
                "path": "/private/project/Foo.java",
            },
        ],
        "build": {
            "commands": [
                {"name": "clean_build", "exit_code": 0, "timed_out": False},
                {"name": "gametest", "exit_code": 0, "timed_out": False},
            ]
        },
    }


def test_hierarchical_memory_contains_real_ordered_procedure_not_only_labels() -> None:
    task = {
        "node_id": "repair-demo",
        "stage": "repair",
        "payload": {"kind": "custom_java", "members": [{"module_id": "demo"}]},
    }
    row = build_work_trajectory(task, outcome="SUCCESS", receipt=_verified_receipt())
    actions = [step["action"] for step in row["procedure"]["steps"]]
    assert actions[:3] == ["inspect_existing_mod", "search_code_rag", "apply_source_patch"]
    assert "clean_build" in actions
    assert "gametest" in actions
    rendered = json.dumps(row["procedure"], ensure_ascii=False)
    assert "SECRET SOURCE" not in rendered
    assert "/private/project" not in rendered

    hierarchy = build_hierarchy([row])
    workflow_procedures = hierarchy["workflow"]["procedures"]
    assert any(
        "inspect_existing_mod > search_code_rag > apply_source_patch" in item
        for item in workflow_procedures
    )
    assert hierarchy["subtask"]["procedures"]
    assert hierarchy["function"]["procedures"]


def _counterexample_project(root: Path) -> None:
    (root / "src/main/java/demo").mkdir(parents=True)
    (root / "src/main/resources/data/demo/recipes").mkdir(parents=True)
    (root / "src/main/resources/assets/demo/blockstates").mkdir(parents=True)
    (root / "src/main/resources/assets/demo/models/block").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True, exist_ok=True)
    (root / "src/main/resources/fabric.mod.json").write_text(
        json.dumps({"id": "demo", "entrypoints": {"main": ["demo.Main"]}}),
        encoding="utf-8",
    )
    (root / "src/main/java/demo/Main.java").write_text(
        "package demo; public final class Main {}\n", encoding="utf-8"
    )
    (root / "src/main/java/demo/Registry.java").write_text(
        'package demo; public final class Registry { String id = "demo:widget"; }\n',
        encoding="utf-8",
    )
    (root / "src/main/resources/data/demo/recipes/use_widget.json").write_text(
        json.dumps({"type": "minecraft:crafting_shapeless", "ingredient": {"item": "demo:widget"}}),
        encoding="utf-8",
    )
    (root / "src/main/resources/assets/demo/blockstates/widget.json").write_text(
        json.dumps({"variants": {"": {"model": "demo:block/widget"}}}),
        encoding="utf-8",
    )
    (root / "src/main/resources/assets/demo/models/block/widget.json").write_text(
        json.dumps({"parent": "minecraft:block/cube_all"}), encoding="utf-8"
    )


def test_generated_counterexample_is_same_for_a_b_and_actually_discriminates(tmp_path: Path) -> None:
    root = tmp_path / "project"
    _counterexample_project(root)
    focus = ["src/main/java/demo/Registry.java"]
    left_ops = [
        {
            "operation": "replace",
            "path": focus[0],
            "expected_sha256": "sha256:" + "0" * 64,
            "content": 'package demo; public final class Registry { String id = "demo:widget"; }\n',
        }
    ]
    right_ops = [
        {
            "operation": "replace",
            "path": focus[0],
            "expected_sha256": "sha256:" + "0" * 64,
            "content": 'package demo; public final class Registry { String id = "demo:gadget"; }\n',
        }
    ]
    spec = build_generated_test_spec(
        root,
        focus_paths=focus,
        left_operations=left_ops,
        right_operations=right_ops,
        evidence_seed={"diagnostics": []},
    )
    assert spec["same_test_for_both_candidates"] is True
    assert any(item["kind"] == "external_identifier_contract" for item in spec["assertions"])

    left = tmp_path / "left"
    right = tmp_path / "right"
    shutil.copytree(root, left)
    shutil.copytree(root, right)
    (left / focus[0]).write_text(str(left_ops[0]["content"]), encoding="utf-8")
    (right / focus[0]).write_text(str(right_ops[0]["content"]), encoding="utf-8")
    left_result = run_generated_test_spec(left, spec)
    right_result = run_generated_test_spec(right, spec)
    assert left_result["status"] == "PASS"
    assert right_result["status"] == "FAIL"
    assert left_result["same_test_for_both_candidates"] is True
    assert right_result["same_test_for_both_candidates"] is True


def test_generated_junit_uses_existing_harness_and_same_static_oracle(tmp_path: Path) -> None:
    root = tmp_path / "project"
    _counterexample_project(root)
    (root / "build.gradle").write_text(
        "dependencies { testImplementation 'org.junit.jupiter:junit-jupiter:5.10.2' }\n"
        "test { useJUnitPlatform() }\n",
        encoding="utf-8",
    )
    focus = ["src/main/resources/assets/demo/models/block/widget.json"]
    operation = {
        "operation": "replace",
        "path": focus[0],
        "expected_sha256": "sha256:" + "0" * 64,
        "content": json.dumps({"parent": "minecraft:block/cube_all"}),
    }
    spec = build_generated_test_spec(
        root,
        focus_paths=focus,
        left_operations=[operation],
        right_operations=[{**operation, "content": json.dumps({"parent": "minecraft:block/cube"})}],
    )
    assert any(item["kind"] == "unchanged_resource_reference" for item in spec["assertions"])
    installed = install_generated_junit(root, spec)
    assert installed["status"] == "INSTALLED"
    assert installed["harness"] == "junit5"
    generated = root / "src/test/java/mmm/generated/MmmGeneratedCounterexampleTest.java"
    assert generated.is_file()
    source = generated.read_text(encoding="utf-8")
    assert "org.junit.jupiter.api.Test" in source
    assert "models/block/widget.json" in source
