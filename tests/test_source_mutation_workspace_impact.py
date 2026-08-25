from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import source_patch as source_patch_module
from minecraft_mod_ai.source_patch import SourcePatchError, TransactionalSourcePatcher, sha256_bytes


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
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import materialize_model_source_edit

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


def test_apply_source_patch_and_search_code_rag_allow_workspace_root(tmp_path: Path) -> None:
    """apply_source_patch and search_code_rag accept '.' / workspace root without rejection."""
    from minecraft_mod_ai.mcp_tools import MMMToolService
    from minecraft_mod_ai.production_tools import ProductionToolService

    (tmp_path / "build.gradle").write_text("// gradle\n", encoding="utf-8")
    file_path = tmp_path / "src" / "main" / "java" / "Mod.java"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("public class Mod {}\n", encoding="utf-8")

    mcp_service = MMMToolService(workspace_root=tmp_path)
    # Apply patch targeting project_root="."
    patch_result = mcp_service.apply_source_patch(
        ".",
        [
            {
                "operation": "replace",
                "path": "src/main/java/Mod.java",
                "expected_sha256": sha256_bytes(file_path.read_bytes()),
                "content": "public class Mod { public static void init() {} }\n",
            }
        ],
    )
    assert patch_result.get("status") == "APPLIED"

    prod_service = ProductionToolService(workspace_root=tmp_path)
    # Search code rag targeting index_path="." or directory
    search_result = prod_service.search_code_rag("Mod", index_path=".")
    assert isinstance(search_result, dict)
    assert search_result.get("schema_version") == "mmm/code-rag-result-v1"


