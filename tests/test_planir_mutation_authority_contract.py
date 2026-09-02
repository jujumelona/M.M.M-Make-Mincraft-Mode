from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.production_tools import ProductionToolService
from minecraft_mod_ai.spec import SpecValidationError


JAVA_PATH = "src/main/java/example/Foo.java"


def _task_payload() -> dict:
    task_id = "task_owned_anchor_authority"
    return {
        "phase": "implement_module",
        "module": {
            "module_id": task_id,
            "kind": "custom_java",
            "config": {
                "evidence_task": {
                    "task_id": task_id,
                    "owned_anchors": [
                        {
                            "kind": "symbol",
                            "locator": f"{JAVA_PATH}#Foo",
                            "status": "host_reserved",
                        },
                        {
                            "kind": "resource",
                            "locator": "resource:src/main/resources/example.json",
                            "status": "host_reserved",
                        },
                        {
                            "kind": "registry",
                            "locator": "registry:example:foo",
                            "status": "host_reserved",
                        },
                        {
                            "kind": "module",
                            "locator": "module:common",
                            "status": "host_reserved",
                        },
                    ],
                    "production_bindings": [
                        {
                            "task_ref": task_id,
                            "reuse_action": "fresh",
                            "owned_anchors": [
                                {
                                    "kind": "symbol",
                                    "locator": f"{JAVA_PATH}#Foo",
                                    "status": "host_reserved",
                                }
                            ],
                        }
                    ],
                }
            },
        },
    }


def test_owned_anchors_are_exact_file_authority_not_abstract_locator_guesses() -> None:
    writable, creatable = tool_loop._planir_owned_anchor_sets(_task_payload())

    assert writable == (JAVA_PATH,)
    assert creatable == (JAVA_PATH,)
    assert all("resource:" not in path for path in writable)
    assert all("registry:" not in path for path in writable)
    assert all("module:" not in path for path in writable)


def test_planir_owned_path_survives_unrelated_localization_pin() -> None:
    messages = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "target_file": "build.gradle",
                    "source": "plugins { id 'fabric-loom' }",
                }
            ),
        },
        {"role": "user", "content": json.dumps(_task_payload())},
    ]
    state = tool_loop.HostRunState()

    assert tool_loop.is_mutation_ready(messages, state) is True
    assert state.mutation_context is not None
    assert state.mutation_context.target_path == "build.gradle"
    assert "build.gradle" in state.mutation_context.writable_paths
    assert JAVA_PATH in state.mutation_context.writable_paths
    assert JAVA_PATH in state.mutation_context.creatable_paths

    assert (
        tool_loop._mutation_target_error(
            "apply_source_edit",
            {"operation": "create_file", "path": JAVA_PATH},
            state.mutation_context,
        )
        is None
    )
    drift = tool_loop._mutation_target_error(
        "apply_source_edit",
        {"operation": "create_file", "path": "src/main/java/example/NotOwned.java"},
        state.mutation_context,
    )
    assert drift is not None
    assert "MUTATION_TARGET_DRIFT" in drift


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": "", "parameters": {"type": "object"}},
    }


def test_read_reuse_source_is_hidden_without_approved_donor_receipt() -> None:
    messages = [{"role": "user", "content": "Please reuse source if useful."}]
    assert tool_loop._approved_donor_source_authority(messages) is False

    filtered = tool_loop._filter_donor_tool_schemas(
        (_schema("search_code_rag"), _schema("read_reuse_source"))
    )
    assert [schema["function"]["name"] for schema in filtered] == ["search_code_rag"]


def test_read_reuse_source_authority_requires_materialized_path_and_immutable_receipt() -> None:
    receipt = {
        "schema_version": "mmm/reuse-source-authority-v1",
        "materialized_path": (
            "/tmp/project/.minecraft_ai/reuse/donors/abc123/"
            "src/main/java/example/Trade.java"
        ),
        "repository": "example/trade",
        "commit_sha": "b" * 40,
        "license_id": "MIT",
        "sha256": "sha256:" + "c" * 64,
    }
    messages = [{"role": "system", "content": json.dumps(receipt)}]
    assert tool_loop._approved_donor_source_authority(messages) is True

    no_path = dict(receipt)
    no_path.pop("materialized_path")
    assert (
        tool_loop._approved_donor_source_authority(
            [{"role": "system", "content": json.dumps(no_path)}]
        )
        is False
    )


def test_donor_root_escape_guard_remains_fail_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    donor = project / ".minecraft_ai" / "reuse" / "donors" / ("a" * 20)
    donor.mkdir(parents=True)
    manifest = {
        "repository": "example/trade",
        "commit_sha": "b" * 40,
        "license_id": "MIT",
        "capability": "trade.transaction",
        "files": [],
    }
    (donor / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    outside = project / "src/main/java/example/Outside.java"
    outside.parent.mkdir(parents=True)
    outside.write_text("class Outside {}", encoding="utf-8")

    service = ProductionToolService(workspace_root=workspace)
    with pytest.raises(SpecValidationError, match="escaped the approved donor root"):
        service.read_reuse_source(str(project), str(outside))
