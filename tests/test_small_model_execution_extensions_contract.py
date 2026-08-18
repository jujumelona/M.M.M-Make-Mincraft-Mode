from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai.small_model_execution_extensions_contract import (
    _compose_skills,
    _materialize_model_source_edit,
)
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher


def _project(workspace):
    project = workspace / "demo"
    (project / "src/main/java/example").mkdir(parents=True)
    (project / "settings.gradle").write_text('rootProject.name = "demo"\n', encoding="utf-8")
    return project


def _skill(identity: str, *, requires=(), provides=(), confidence=1.0):
    return {
        "skill_id": identity,
        "name": identity,
        "activate_when": [identity],
        "steps": [f"run {identity}"],
        "constraints": [],
        "requires": list(requires),
        "provides": list(provides),
        "confidence": confidence,
    }


def test_partial_source_edit_coalesces_exact_edits_and_host_hash(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    old = "final class Example {\n    int oldValue;\n}\n"
    source.write_text(old, encoding="utf-8")

    payload = _materialize_model_source_edit(
        agent_tool_runtime,
        workspace,
        {
            "edits": [
                {
                    "operation": "replace_exact",
                    "path": "src/main/java/example/Example.java",
                    "old": "oldValue",
                    "new": "newValue",
                },
                {
                    "operation": "insert_before",
                    "path": "src/main/java/example/Example.java",
                    "anchor": "}\n",
                    "content": "    void run() {}\n",
                },
            ]
        },
    )

    assert payload["project_root"] == "demo"
    assert len(payload["operations"]) == 1
    operation = payload["operations"][0]
    assert operation["operation"] == "edit"
    assert operation["expected_sha256"] == "sha256:" + hashlib.sha256(old.encode()).hexdigest()
    assert len(operation["replacements"]) == 2

    receipt = TransactionalSourcePatcher(project).apply(payload["operations"])
    assert receipt["status"] == "APPLIED"
    assert source.read_text(encoding="utf-8") == (
        "final class Example {\n    int newValue;\n    void run() {}\n}\n"
    )


def test_partial_source_edit_rejects_ambiguous_anchor_without_mutation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = _project(workspace)
    source = project / "src/main/java/example/Example.java"
    before = "token token\n"
    source.write_text(before, encoding="utf-8")

    with pytest.raises(agent_tool_runtime.AgentToolRuntimeError, match="expected 1 matches, found 2"):
        _materialize_model_source_edit(
            agent_tool_runtime,
            workspace,
            {
                "edits": [
                    {
                        "operation": "insert_after",
                        "path": "src/main/java/example/Example.java",
                        "anchor": "token",
                        "content": "!",
                    }
                ]
            },
        )
    assert source.read_text(encoding="utf-8") == before


def test_ordered_skill_composition_adds_provider_before_consumer() -> None:
    provider = _skill("provider", provides=["compiled registry"])
    consumer = _skill("consumer", requires=["compiled registry"])

    result = _compose_skills("consumer", [consumer, provider], [consumer])

    assert [item["skill_id"] for item in result["ordered_skills"]] == ["provider", "consumer"]
    assert result["unresolved_requirements"] == []
    assert result["cycles"] == []
    assert result["composition_policy"] == "explicit_requires_provides_only"


def test_ordered_skill_composition_blocks_unresolved_requirement() -> None:
    consumer = _skill("consumer", requires=["missing capability"])

    result = _compose_skills("consumer", [consumer], [consumer])

    assert result["ordered_skills"] == []
    assert result["unresolved_requirements"] == [
        {"skill_id": "consumer", "requirement": "missing capability"}
    ]
    assert result["blocked_skill_ids"] == ["consumer"]


def test_ordered_skill_composition_detects_cycle_and_does_not_execute_it() -> None:
    first = _skill("first", requires=["b"], provides=["a"])
    second = _skill("second", requires=["a"], provides=["b"])

    result = _compose_skills("first", [first, second], [first])

    assert result["ordered_skills"] == []
    assert result["cycles"] == [["first", "second"]]
    assert result["blocked_skill_ids"] == ["first", "second"]


def test_skill_composition_does_not_infer_dependencies_from_similar_words() -> None:
    first = _skill("registry producer", provides=["registry output"])
    second = _skill("registry consumer")

    result = _compose_skills("registry consumer", [first, second], [second])

    assert [item["skill_id"] for item in result["ordered_skills"]] == ["registry consumer"]
    assert result["dependency_edges"] == []
