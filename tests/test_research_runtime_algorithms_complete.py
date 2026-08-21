from __future__ import annotations

import json
import shutil
from pathlib import Path

from minecraft_mod_ai import model_router
from minecraft_mod_ai.causal_tool_frontier_contract import goals_for_query
from minecraft_mod_ai.causal_tool_graph import (
    executable_frontier,
    shortest_causal_path,
    transition_for_schema,
    verified_state_from_messages,
)
from minecraft_mod_ai.generated_counterexample_tests import (
    build_generated_test_spec,
    install_generated_junit,
    run_generated_test_spec,
)
from minecraft_mod_ai.procedural_memory_hierarchy import build_hierarchy
from minecraft_mod_ai.trajectory_memory import build_work_trajectory


def _schema(name: str, description: str = "") -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_known_tool_causality_is_explicit_not_description_inferred() -> None:
    inspect = transition_for_schema(
        _schema("inspect_existing_mod", "delete mutate verify package everything")
    )
    assert inspect.reviewed is True
    assert inspect.preconditions == frozenset({"workspace_bound"})
    assert inspect.effects == frozenset({"project_observed"})

    opaque = transition_for_schema(
        _schema("plugin_magic", "verify project and mutate runtime with perfect evidence")
    )
    assert opaque.reviewed is False
    assert opaque.effects == frozenset({"opaque:plugin_magic"})
    assert "verified" not in opaque.effects
    assert "project_changed" not in opaque.effects


def test_goal_resolution_chooses_terminal_state_not_keyword_conjunction() -> None:
    assert goals_for_query("inspect exact Minecraft API") == ("evidence",)
    assert goals_for_query("fix compile error using exact API") == ("repair",)
    assert goals_for_query("generate a new Fabric mod") == ("generate",)
    assert goals_for_query("verify the current build") == ("verify",)
    assert goals_for_query("inspect current project") == ("observe",)


def test_causal_frontier_advances_only_after_verified_observations() -> None:
    tools = (
        _schema("inspect_existing_mod"),
        _schema("search_code_rag"),
        _schema("apply_source_patch"),
        _schema("plugin_magic", "pretend to satisfy everything"),
    )
    initial = frozenset({"workspace_bound"})
    path = shortest_causal_path(tools, state=initial, goals=("repair",), max_depth=8)
    assert path
    assert path[-1] == "apply_source_patch"
    assert "plugin_magic" not in path
    assert executable_frontier(
        tools,
        state=initial,
        goals=("repair",),
        limit=3,
        preference={"inspect_existing_mod": 0, "search_code_rag": 1},
    ) == ("search_code_rag",)

    after_inspect = verified_state_from_messages(
        [{"role": "tool", "name": "inspect_existing_mod", "content": '{"ok":true}'}],
        tools,
    )
    assert "project_observed" in after_inspect
    assert executable_frontier(
        tools,
        state=after_inspect,
        goals=("repair",),
        limit=3,
        preference={"search_code_rag": 0},
    ) == ("search_code_rag",)

    weak_search = verified_state_from_messages(
        [
            {"role": "tool", "name": "inspect_existing_mod", "content": '{"ok":true}'},
            {
                "role": "tool",
                "name": "search_code_rag",
                "content": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "receipt": {
                                "result_count": 1,
                                "coverage_score": 0.2,
                                "relevance_score": 0.7,
                            }
                        },
                    }
                ),
            },
        ],
        tools,
    )
    assert "code_evidence" in weak_search
    assert "evidence_ready" not in weak_search

    after_evidence = verified_state_from_messages(
        [
            {"role": "tool", "name": "inspect_existing_mod", "content": '{"ok":true}'},
            {
                "role": "tool",
                "name": "search_code_rag",
                "content": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "receipt": {
                                "result_count": 3,
                                "coverage_score": 0.9,
                                "relevance_score": 0.8,
                            }
                        },
                    }
                ),
            },
        ],
        tools,
    )
    assert {"project_observed", "code_evidence", "evidence_ready"} <= set(after_evidence)
    assert executable_frontier(
        tools,
        state=after_evidence,
        goals=("repair",),
        limit=3,
    ) == ("apply_source_patch",)

    failed = verified_state_from_messages(
        [{"role": "tool", "name": "inspect_existing_mod", "content": '{"ok":false}'}],
        tools,
    )
    assert failed == frozenset({"workspace_bound"})


def test_semantic_rank_only_breaks_ties_between_equal_minimum_causal_paths() -> None:
    tools = (
        _schema("inspect_existing_mod"),
        _schema("search_code_rag"),
        _schema("search_project_rag"),
        _schema("inspect_github_repository"),
        _schema("generate_fabric_project"),
        _schema("apply_source_patch"),
    )
    state = frozenset({"workspace_bound"})
    preference = {
        "inspect_existing_mod": 0,
        "search_code_rag": 1,
        "search_project_rag": 2,
        "inspect_github_repository": 3,
        "generate_fabric_project": 4,
        "apply_source_patch": 5,
    }
    frontier = executable_frontier(
        tools,
        state=state,
        goals=("repair",),
        limit=3,
        max_depth=8,
        preference=preference,
    )
    assert frontier == ("search_code_rag",)
    assert "generate_fabric_project" not in frontier
    repair_path = shortest_causal_path(tools, state=state, goals=("repair",), max_depth=8)
    assert repair_path[-1] == "apply_source_patch"
    generation_path = shortest_causal_path(tools, state=state, goals=("generate",), max_depth=8)
    assert generation_path[-1] == "generate_fabric_project"


def test_external_mcp_frontier_allows_direct_call_when_arguments_are_known() -> None:
    tools = (
        _schema("external_mcp_capabilities"),
        _schema("external_mcp_schema"),
        _schema("external_mcp_call"),
    )
    state = frozenset({"workspace_bound"})
    frontier = executable_frontier(tools, state=state, goals=("external",), limit=3)
    assert 1 <= len(frontier) <= 3
    assert "external_mcp_call" in frontier

    transport_only = verified_state_from_messages(
        [{"role": "tool", "name": "external_mcp_call", "content": '{"ok":true}'}],
        tools,
        require_fresh_evidence=True,
    )
    assert "external_observation" not in transport_only
    assert "evidence_ready" not in transport_only

    bundle = {
        "ok": True,
        "result": {
            "schema_version": "mmm/external-mcp-evidence-bundle-v1",
            "status": "PASS",
            "required_corroboration": 1,
            "evidence": [
                {
                    "schema_version": "mmm/external-mcp-call-receipt-v1",
                    "status": "PASS",
                    "access": "read",
                    "result": {"structured": {"symbol": "Block"}},
                }
            ],
        },
    }
    after_call = verified_state_from_messages(
        [
            {
                "role": "tool",
                "name": "external_mcp_call",
                "content": json.dumps(bundle),
            }
        ],
        tools,
        require_fresh_evidence=True,
    )
    assert {"external_observation", "evidence_ready"} <= set(after_call)


def test_live_model_tool_loop_has_dynamic_causal_recalculation() -> None:
    current = model_router.ModelRouter._generate_with_tools
    found = False
    for _ in range(12):
        if getattr(current, "_mmm_dynamic_causal_frontier", False):
            found = True
            break
        current = getattr(current, "__wrapped__", None)
        if current is None:
            break
    assert found


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
