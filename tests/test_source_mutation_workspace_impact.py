from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import source_patch as source_patch_module
from minecraft_mod_ai.source_patch import (
    SourcePatchError,
    TransactionalSourcePatcher,
    sha256_bytes,
)


def test_source_patch_sha_mismatch_reports_workspace_drift(tmp_path: Path) -> None:
    target = tmp_path / "src/main/java/Test.java"
    target.parent.mkdir(parents=True)
    target.write_text("class Test {}\n", encoding="utf-8")
    patcher = TransactionalSourcePatcher(tmp_path)

    with pytest.raises(SourcePatchError) as caught:
        patcher.apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/Test.java",
                    "expected_sha256": "sha256:" + "0" * 64,
                    "content": "class Test { int x; }\n",
                }
            ]
        )

    assert caught.value.workspace_impact == "drift"
    assert "[workspace_impact=drift]" in str(caught.value)
    assert target.read_text(encoding="utf-8") == "class Test {}\n"


def test_source_patch_prewrite_rejection_reports_unchanged(tmp_path: Path) -> None:
    patcher = TransactionalSourcePatcher(tmp_path)
    with pytest.raises(SourcePatchError) as caught:
        patcher.apply([])
    assert caught.value.workspace_impact == "unchanged"


def test_source_patch_commit_failure_reports_successful_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "src/main/java/First.java"
    second = tmp_path / "src/main/java/Second.java"
    first.parent.mkdir(parents=True)
    first.write_text("class First {}\n", encoding="utf-8")
    second.write_text("class Second {}\n", encoding="utf-8")
    before_first = first.read_bytes()
    before_second = second.read_bytes()
    original_commit = source_patch_module._commit_staged_path
    calls = 0

    def fail_second(path: Path, after: bytes | None) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic commit failure")
        original_commit(path, after)

    monkeypatch.setenv("MMM_SOURCE_PATCH_WORKERS", "1")
    monkeypatch.setattr(source_patch_module, "_commit_staged_path", fail_second)
    patcher = TransactionalSourcePatcher(tmp_path)
    with pytest.raises(SourcePatchError) as caught:
        patcher.apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/First.java",
                    "expected_sha256": sha256_bytes(before_first),
                    "content": "class First { int x; }\n",
                },
                {
                    "operation": "replace",
                    "path": "src/main/java/Second.java",
                    "expected_sha256": sha256_bytes(before_second),
                    "content": "class Second { int y; }\n",
                },
            ]
        )

    assert caught.value.workspace_impact == "rolled_back"
    assert first.read_bytes() == before_first
    assert second.read_bytes() == before_second


def test_source_patch_failed_rollback_reports_uncertain_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "src/main/java/First.java"
    second = tmp_path / "src/main/java/Second.java"
    first.parent.mkdir(parents=True)
    first.write_text("class First {}\n", encoding="utf-8")
    second.write_text("class Second {}\n", encoding="utf-8")
    before_first = first.read_bytes()
    before_second = second.read_bytes()
    original_commit = source_patch_module._commit_staged_path
    original_write_bytes = Path.write_bytes
    calls = 0

    def fail_second(path: Path, after: bytes | None) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic commit failure")
        original_commit(path, after)

    def fail_first_restore(path: Path, data: bytes) -> int:
        if path == first and data == before_first:
            raise OSError("synthetic rollback failure")
        return original_write_bytes(path, data)

    monkeypatch.setenv("MMM_SOURCE_PATCH_WORKERS", "1")
    monkeypatch.setattr(source_patch_module, "_commit_staged_path", fail_second)
    monkeypatch.setattr(Path, "write_bytes", fail_first_restore)
    patcher = TransactionalSourcePatcher(tmp_path)
    with pytest.raises(SourcePatchError) as caught:
        patcher.apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/First.java",
                    "expected_sha256": sha256_bytes(before_first),
                    "content": "class First { int x; }\n",
                },
                {
                    "operation": "replace",
                    "path": "src/main/java/Second.java",
                    "expected_sha256": sha256_bytes(before_second),
                    "content": "class Second { int y; }\n",
                },
            ]
        )

    assert caught.value.workspace_impact == "uncertain"
    assert "[workspace_impact=uncertain]" in str(caught.value)


def test_materialize_model_source_edit_handles_apply_source_edit_and_aliases(tmp_path: Path) -> None:
    """materialize_model_source_edit accepts new_text/old_text aliases and operation='apply_source_edit'."""
    from minecraft_mod_ai import agent_tool_runtime as runtime_module
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
        materialize_model_source_edit,
    )

    (tmp_path / "build.gradle").write_text("// gradle\n", encoding="utf-8")
    file_path = tmp_path / "src" / "main" / "java" / "Mod.java"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("public class Mod {\n    // placeholder\n}\n", encoding="utf-8")

    payload = {
        "operation": "apply_source_edit",
        "file": "src/main/java/Mod.java",
        "old_text": "// placeholder",
        "new_text": "public static void init() {}",
    }
    result = materialize_model_source_edit(runtime_module, tmp_path, payload)
    assert isinstance(result, dict)
    ops = result.get("operations")
    assert isinstance(ops, list) and len(ops) == 1
    op = ops[0]
    assert op.get("operation") == "edit"
    assert op.get("replacements", [{}])[0].get("new") == "public static void init() {}"


def _make_gradle_project(root: Path, java_rel: str, java_content: str) -> Path:
    """Create a minimal Gradle project layout for testing."""
    (root / "build.gradle").write_text("// gradle\n", encoding="utf-8")
    (root / "src").mkdir(parents=True, exist_ok=True)
    file_path = root / java_rel
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(java_content, encoding="utf-8")
    return file_path


def _run_e2e_patch(workspace_root: Path, project_root: Path, java_rel: str, java_content: str, old_snippet: str, new_snippet: str) -> Path:
    """
    Full pipeline: Qwen-style payload → materialize_model_source_edit
    → MMMToolService.apply_source_patch → TransactionalSourcePatcher → file diff.

    Returns the patched file path so callers can assert on its content.
    """
    import minecraft_mod_ai.agent_tool_runtime as rt_module
    from minecraft_mod_ai.mcp_tools import MMMToolService
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
        materialize_model_source_edit,
    )

    file_path = _make_gradle_project(project_root, java_rel, java_content)

    # Simulated Qwen payload with aliases (operation="apply_source_edit", old_text/new_text)
    payload = {
        "operation": "replace_exact",
        "path": java_rel,
        "old": old_snippet,
        "new": new_snippet,
    }

    # Step 1: materializer (workspace_root = project_root for bound-project case)
    patch = materialize_model_source_edit(rt_module, workspace_root, payload)
    assert isinstance(patch, dict), "materialize_model_source_edit must return a dict"
    assert "project_root" in patch, "patch must contain project_root"
    assert "operations" in patch, "patch must contain operations"

    # Step 2: MCP apply_source_patch (receives project_root from materializer)
    mcp = MMMToolService(workspace_root=workspace_root)
    result = mcp.apply_source_patch(patch["project_root"], patch["operations"])

    # Step 3: verify result
    assert result.get("status") == "APPLIED", f"Expected APPLIED, got: {result}"

    # Step 4: verify actual file on disk
    new_text = file_path.read_text(encoding="utf-8")
    assert new_snippet in new_text, (
        f"Expected new snippet in file after patch:\n{new_text!r}"
    )
    assert old_snippet not in new_text, (
        f"Old snippet must be gone after patch:\n{new_text!r}"
    )
    return file_path


def test_e2e_source_edit_project_is_workspace_root(tmp_path: Path) -> None:
    """
    End-to-end: project root == workspace root (project_root='.' from materializer).
    Full path: Qwen payload → materializer → MMMToolService.apply_source_patch
               → TransactionalSourcePatcher → actual file diff.
    """
    java_content = (
        "package com.example;\n\npublic class Mod {\n    // init placeholder\n}\n"
    )
    _run_e2e_patch(
        workspace_root=tmp_path,
        project_root=tmp_path,
        java_rel="src/main/java/com/example/Mod.java",
        java_content=java_content,
        old_snippet="// init placeholder",
        new_snippet="public static void onInitialize() {}",
    )


def test_e2e_source_edit_project_is_subdir_of_workspace(tmp_path: Path) -> None:
    """
    End-to-end: project root is a subdirectory of workspace.
    Full path: Qwen payload → materializer → MMMToolService.apply_source_patch
               → TransactionalSourcePatcher → actual file diff.
    """
    project_dir = tmp_path / "mymod"
    project_dir.mkdir()
    java_content = (
        "package com.example;\n\npublic class Main {\n    // entry\n}\n"
    )
    _run_e2e_patch(
        workspace_root=tmp_path,
        project_root=project_dir,
        java_rel="src/main/java/com/example/Main.java",
        java_content=java_content,
        old_snippet="// entry",
        new_snippet="public static void main(String[] args) {}",
    )


def test_e2e_source_edit_alias_payload_project_is_workspace_root(tmp_path: Path) -> None:
    """
    End-to-end with Qwen alias fields (operation='apply_source_edit', new_text/old_text):
    project root == workspace root. Verifies alias normalization + actual file diff.
    """
    import minecraft_mod_ai.agent_tool_runtime as rt_module
    from minecraft_mod_ai.mcp_tools import MMMToolService
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
        materialize_model_source_edit,
    )

    java_content = (
        "package ai.test;\n\npublic class SpaceMod {\n    // placeholder\n}\n"
    )
    file_path = _make_gradle_project(
        tmp_path, "src/main/java/ai/test/SpaceMod.java", java_content
    )

    # Qwen emits 'apply_source_edit' as operation + alias field names
    payload = {
        "operation": "apply_source_edit",
        "file": "src/main/java/ai/test/SpaceMod.java",
        "old_text": "// placeholder",
        "new_text": "public static void onInitialize() { LOGGER.info(\"started\"); }",
    }

    patch = materialize_model_source_edit(rt_module, tmp_path, payload)
    assert patch["project_root"] is not None

    mcp = MMMToolService(workspace_root=tmp_path)
    result = mcp.apply_source_patch(patch["project_root"], patch["operations"])
    assert result.get("status") == "APPLIED", f"Expected APPLIED, got: {result}"

    new_text = file_path.read_text(encoding="utf-8")
    assert "onInitialize" in new_text
    assert "// placeholder" not in new_text


def test_qwen_tool_parser_tolerates_non_enum_operations_and_aliases() -> None:
    """Qwen parser parses markup with uncanonical enum/operation strings without crashing."""
    from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA

    schemas = {"apply_source_edit": SOURCE_EDIT_SCHEMA}
    markup = (
        "<tool_call>\n"
        "<function=apply_source_edit>\n"
        "<parameter=operation>create_class</parameter>\n"
        "<parameter=path>src/main/java/ai/test/NewClass.java</parameter>\n"
        "<parameter=package_name>ai.test</parameter>\n"
        "<parameter=declaration>public class NewClass</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    _visible, calls = parse_qwen_tool_markup(markup, schemas)
    assert len(calls) == 1
    assert calls[0].name == "apply_source_edit"
    assert calls[0].arguments["operation"] == "create_class"
    assert calls[0].arguments["path"] == "src/main/java/ai/test/NewClass.java"


def test_repair_engine_hydrates_missing_expected_sha256(tmp_path: Path) -> None:
    """RepairEngine automatically hydrates missing expected_sha256 from disk files."""
    from minecraft_mod_ai.repair_engine import RepairEngine
    from minecraft_mod_ai.source_patch import TransactionalSourcePatcher

    gradle_file = tmp_path / "build.gradle"
    gradle_file.write_text("plugins { id 'fabric-loom' }\n", encoding="utf-8")

    operations = [
        {
            "operation": "replace",
            "path": "build.gradle",
            "content": "plugins { id 'fabric-loom' version '1.7-SNAPSHOT' }\n",
        }
    ]

    # Without hydration, TransactionalSourcePatcher raises SourcePatchError because expected_sha256 is missing
    with pytest.raises(Exception):
        TransactionalSourcePatcher(tmp_path).apply(operations)

    # With RepairEngine._hydrate_repair_preconditions, expected_sha256 is hydrated
    RepairEngine._hydrate_repair_preconditions(tmp_path, operations)
    assert "expected_sha256" in operations[0]
    assert operations[0]["expected_sha256"].startswith("sha256:")

    # Patcher now applies successfully
    receipt = TransactionalSourcePatcher(tmp_path).apply(operations)
    assert receipt["status"] == "APPLIED"
    assert "1.7-SNAPSHOT" in gradle_file.read_text(encoding="utf-8")


def test_qwen_tool_parser_tolerates_invalid_json_in_unknown_or_array_parameter() -> None:
    """Qwen parser handles unescaped JSON strings in parameters without crashing."""
    from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup

    schemas = {}
    markup = (
        "<tool_call>\n"
        "<function=create_java_type>\n"
        "<parameter=top_level_members>[\"public static final Item ITEM = new Item();\"]</parameter>\n"
        "<parameter=invalid_json>[unescaped, 'quotes', {bad json</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    _visible, calls = parse_qwen_tool_markup(markup, schemas)
    assert len(calls) == 1
    assert calls[0].name == "create_java_type"
    assert "top_level_members" in calls[0].arguments
    assert "invalid_json" in calls[0].arguments


def test_repair_engine_handles_delete_and_aliases(tmp_path: Path) -> None:
    """RepairEngine validates and executes delete operations and alias payloads cleanly."""
    from minecraft_mod_ai.repair_engine import RepairEngine
    from minecraft_mod_ai.source_patch import TransactionalSourcePatcher

    junk_file = tmp_path / "junk.java"
    junk_file.write_text("class Junk {}", encoding="utf-8")

    operations = [
        {
            "operation": "delete_file",
            "path": "junk.java",
            "explanation": "delete obsolete file",
        }
    ]

    RepairEngine._hydrate_repair_preconditions(tmp_path, operations)
    RepairEngine._validate_patch_scope(operations)
    assert operations[0]["operation"] == "delete"
    assert "expected_sha256" in operations[0]
    assert "explanation" not in operations[0]

    receipt = TransactionalSourcePatcher(tmp_path).apply(operations)
    assert receipt["status"] == "APPLIED"
    assert not junk_file.exists()







