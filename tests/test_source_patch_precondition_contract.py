from __future__ import annotations

import pytest
from types import SimpleNamespace

from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher
from minecraft_mod_ai.source_patch_precondition_contract import (
    SourcePatchPreconditionError,
    bind_source_snapshot_preconditions,
    preflight_source_patch_operations,
)


def _generator_with_index(index: ProjectIndex):
    return SimpleNamespace(_cached_index=index, _cached_root=index.root)


def test_missing_expected_sha_is_bound_before_fabric_mod_replace(tmp_path) -> None:
    root = tmp_path / "project"
    target = root / "src/main/resources/fabric.mod.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"id":"demo","version":"1"}\n', encoding="utf-8")
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "replace",
            "path": "src/main/resources/fabric.mod.json",
            "content": '{"id":"demo","version":"2"}\n',
        }
    ]

    bind_source_snapshot_preconditions(generator, operations)
    preflight_source_patch_operations(generator, operations)

    assert operations[0]["expected_sha256"].startswith("sha256:")
    receipt = TransactionalSourcePatcher(root).apply(operations)
    assert receipt["status"] == "APPLIED"
    assert '"version":"2"' in target.read_text(encoding="utf-8")


def test_existing_create_is_normalized_to_hash_guarded_replace(tmp_path) -> None:
    root = tmp_path / "project"
    target = root / "src/main/resources/fabric.mod.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"id":"demo"}\n', encoding="utf-8")
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "create",
            "path": "src/main/resources/fabric.mod.json",
            "content": '{"id":"demo","name":"Demo"}\n',
        }
    ]

    bind_source_snapshot_preconditions(generator, operations)
    preflight_source_patch_operations(generator, operations)

    assert operations[0]["operation"] == "replace"
    assert operations[0]["expected_sha256"].startswith("sha256:")


def test_model_supplied_sha_cannot_override_snapshot_identity(tmp_path) -> None:
    root = tmp_path / "project"
    target = root / "src/main/resources/fabric.mod.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"id":"demo"}\n', encoding="utf-8")
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "replace",
            "path": "src/main/resources/fabric.mod.json",
            "expected_sha256": "sha256:" + "0" * 64,
            "content": '{"id":"changed"}\n',
        }
    ]

    with pytest.raises(SourcePatchPreconditionError, match="disagrees"):
        bind_source_snapshot_preconditions(generator, operations)


def test_unobserved_replace_without_sha_fails_before_transaction(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "replace",
            "path": "src/main/resources/missing.json",
            "content": "{}\n",
        }
    ]

    with pytest.raises(SourcePatchPreconditionError, match="No observed source SHA"):
        bind_source_snapshot_preconditions(generator, operations)


def test_unobserved_replace_cannot_authorize_itself_with_fabricated_sha(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "replace",
            "path": "src/main/resources/missing.json",
            "expected_sha256": "sha256:" + "1" * 64,
            "content": "{}\n",
        }
    ]

    with pytest.raises(SourcePatchPreconditionError, match="No observed source SHA"):
        bind_source_snapshot_preconditions(generator, operations)


def test_replace_shape_failure_is_reported_before_apply(tmp_path) -> None:
    root = tmp_path / "project"
    target = root / "src/main/resources/fabric.mod.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"id":"demo"}\n', encoding="utf-8")
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "replace",
            "path": "src/main/resources/fabric.mod.json",
        }
    ]

    bind_source_snapshot_preconditions(generator, operations)
    with pytest.raises(SourcePatchPreconditionError, match="Replace content must be text"):
        preflight_source_patch_operations(generator, operations)


def test_edit_replacement_precondition_is_reported_before_apply(tmp_path) -> None:
    root = tmp_path / "project"
    target = root / "src/main/java/Demo.java"
    target.parent.mkdir(parents=True)
    target.write_text("public class Demo {}\n", encoding="utf-8")
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "edit",
            "path": "src/main/java/Demo.java",
            "replacements": [
                {"old": "class Missing", "new": "class Present", "count": 1}
            ],
        }
    ]

    bind_source_snapshot_preconditions(generator, operations)
    with pytest.raises(SourcePatchPreconditionError, match="Replacement precondition failed"):
        preflight_source_patch_operations(generator, operations)


def test_noop_replace_is_reported_before_apply(tmp_path) -> None:
    root = tmp_path / "project"
    target = root / "src/main/resources/fabric.mod.json"
    target.parent.mkdir(parents=True)
    original = '{"id":"demo"}\n'
    target.write_text(original, encoding="utf-8")
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "replace",
            "path": "src/main/resources/fabric.mod.json",
            "content": original,
        }
    ]

    bind_source_snapshot_preconditions(generator, operations)
    with pytest.raises(SourcePatchPreconditionError, match="makes no change"):
        preflight_source_patch_operations(generator, operations)


def test_source_change_after_snapshot_still_fails_transaction(tmp_path) -> None:
    root = tmp_path / "project"
    target = root / "src/main/resources/fabric.mod.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"id":"demo","version":"1"}\n', encoding="utf-8")
    generator = _generator_with_index(ProjectIndex(root))
    operations = [
        {
            "operation": "replace",
            "path": "src/main/resources/fabric.mod.json",
            "content": '{"id":"demo","version":"2"}\n',
        }
    ]
    bind_source_snapshot_preconditions(generator, operations)
    preflight_source_patch_operations(generator, operations)
    target.write_text('{"id":"concurrent","version":"9"}\n', encoding="utf-8")

    with pytest.raises(Exception, match="SHA-256 precondition failed"):
        TransactionalSourcePatcher(root).apply(operations)


def test_precondition_contract_is_explicit_and_wrapper_free() -> None:
    assert callable(bind_source_snapshot_preconditions)
    assert callable(preflight_source_patch_operations)
