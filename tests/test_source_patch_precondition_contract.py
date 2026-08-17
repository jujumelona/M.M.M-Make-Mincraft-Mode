from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher
from minecraft_mod_ai.source_patch_precondition_contract import (
    SourcePatchPreconditionError,
    bind_source_snapshot_preconditions,
)


def _generator_with_index(index: ProjectIndex):
    generator = object.__new__(CustomModuleGenerator)
    generator._cached_index = index
    generator._cached_root = index.root
    return generator


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

    generator._validate_operations(operations)

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

    generator._validate_operations(operations)

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
    generator._validate_operations(operations)
    target.write_text('{"id":"concurrent","version":"9"}\n', encoding="utf-8")

    with pytest.raises(Exception, match="SHA-256 precondition failed"):
        TransactionalSourcePatcher(root).apply(operations)


def test_runtime_bootstrap_installs_precondition_normalizer() -> None:
    assert getattr(
        CustomModuleGenerator._validate_operations,
        "_mmm_source_snapshot_preconditions_v1",
        False,
    ) is True
