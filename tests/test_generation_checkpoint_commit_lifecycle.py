from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.generation_checkpoint import (
    finalize_persisted_generation_checkpoint,
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _checkpoint_result(project_root: Path, identity: str) -> dict:
    target = project_root / "src/main/java/demo/Feature.java"
    return {
        "schema_version": "mmm/custom-module-result-v3",
        "status": "SOURCE_GENERATED",
        "patch_receipt": {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "project_root": str(project_root),
            "changed_paths": ["src/main/java/demo/Feature.java"],
            "operations": [
                {
                    "path": "src/main/java/demo/Feature.java",
                    "operation": "replace",
                    "before_sha256": None,
                    "after_sha256": _sha256(target.read_bytes()),
                }
            ],
        },
        "generation_checkpoint": {
            "schema_version": "mmm/custom-module-checkpoint-v2",
            "status": "AWAITING_LIVE_COMMIT",
            "identity_sha256": identity,
            "cleanup_token": "f" * 64,
        },
    }


def _persist_checkpoint(root: Path, identity: str) -> Path:
    digest = identity.removeprefix("sha256:")
    checkpoint = root / ".mmm-custom-checkpoints" / digest
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": "mmm/custom-module-checkpoint-v2",
                "identity_sha256": identity,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_cached_committed_checkpoint_is_cleaned_only_when_live_patch_matches(tmp_path) -> None:
    project = tmp_path / "project"
    target = project / "src/main/java/demo/Feature.java"
    target.parent.mkdir(parents=True)
    target.write_text("final source\n", encoding="utf-8")
    identity = "sha256:" + "a" * 64
    checkpoint = _persist_checkpoint(tmp_path, identity)
    result = _checkpoint_result(project, identity)

    assert finalize_persisted_generation_checkpoint(
        result,
        project_root=project,
        checkpoint_root=tmp_path / ".mmm-custom-checkpoints",
    )
    assert not checkpoint.exists()
    assert result["generation_checkpoint"]["status"] == "CLEANED_AFTER_LIVE_COMMIT"
    assert "cleanup_token" not in result["generation_checkpoint"]


def test_cached_checkpoint_is_preserved_when_live_patch_digest_drifted(tmp_path) -> None:
    project = tmp_path / "project"
    target = project / "src/main/java/demo/Feature.java"
    target.parent.mkdir(parents=True)
    target.write_text("committed source\n", encoding="utf-8")
    identity = "sha256:" + "b" * 64
    checkpoint = _persist_checkpoint(tmp_path, identity)
    result = _checkpoint_result(project, identity)
    target.write_text("later unrelated mutation\n", encoding="utf-8")

    assert not finalize_persisted_generation_checkpoint(
        result,
        project_root=project,
        checkpoint_root=tmp_path / ".mmm-custom-checkpoints",
    )
    assert checkpoint.is_dir()
    assert result["generation_checkpoint"]["status"] == "AWAITING_LIVE_COMMIT"


class _Ledger:
    def __init__(self, *, fail_commit: bool = False):
        self.state = "pending"
        self.receipt = None
        self.fail_commit = fail_commit

    def cached_receipt(self, _node_id, *, input_hash):
        return None

    def task(self, _node_id):
        return {"state": self.state, "error": ""}

    def retry(self, _node_id):
        self.state = "pending"

    def raise_if_cancelled(self):
        return None

    def begin(self, _node_id, *, worker_id):
        self.state = "running"

    def succeed(self, _node_id, receipt):
        if self.fail_commit:
            raise RuntimeError("ledger commit failed")
        self.receipt = receipt
        self.state = "succeeded"

    def fail(self, _node_id, _error):
        self.state = "failed"


def _node():
    return SimpleNamespace(
        node_id="generate:custom:test",
        stage="generate:custom",
        input_hash="sha256:" + "c" * 64,
        payload={},
        to_dict=lambda: {},
    )


def test_work_node_runs_commit_callback_only_after_durable_succeed() -> None:
    ledger = _Ledger()
    events: list[str] = []

    result = CompleteProductionOrchestrator._run_work_node(
        ledger,
        _node(),
        action=lambda: {"status": "SUCCEEDED"},
        validate_cached=lambda _value: True,
        on_commit=lambda _receipt: events.append("commit"),
        on_abort=lambda _receipt: events.append("abort"),
    )

    assert result == {"status": "SUCCEEDED"}
    assert ledger.state == "succeeded"
    assert events == ["commit"]


def test_work_node_releases_uncommitted_checkpoint_when_ledger_commit_fails() -> None:
    ledger = _Ledger(fail_commit=True)
    events: list[str] = []

    with pytest.raises(RuntimeError, match="ledger commit failed"):
        CompleteProductionOrchestrator._run_work_node(
            ledger,
            _node(),
            action=lambda: {"status": "SUCCEEDED"},
            validate_cached=lambda _value: True,
            on_commit=lambda _receipt: events.append("commit"),
            on_abort=lambda _receipt: events.append("abort"),
        )

    assert ledger.state == "failed"
    assert events == ["abort"]


def test_cached_receipt_accepts_later_repair_when_checkpoint_is_already_gone(tmp_path) -> None:
    project = tmp_path / "project"
    target = project / "src/main/java/demo/Feature.java"
    target.parent.mkdir(parents=True)
    target.write_text("original generated source\n", encoding="utf-8")
    identity = "sha256:" + "d" * 64
    result = _checkpoint_result(project, identity)

    # Simulate a later build-repair mutation after the original checkpoint was
    # successfully cleaned in the previous process.
    target.write_text("repaired source\n", encoding="utf-8")

    assert finalize_persisted_generation_checkpoint(
        result,
        project_root=project,
        checkpoint_root=tmp_path / ".mmm-custom-checkpoints",
    )
    assert result["generation_checkpoint"]["status"] == "CLEANED_AFTER_LIVE_COMMIT"


def test_work_node_runs_abort_callback_when_action_raises_before_receipt() -> None:
    ledger = _Ledger()
    events: list[object] = []

    def fail_before_receipt():
        raise RuntimeError("generation action failed")

    with pytest.raises(RuntimeError, match="generation action failed"):
        CompleteProductionOrchestrator._run_work_node(
            ledger,
            _node(),
            action=fail_before_receipt,
            validate_cached=lambda _value: True,
            on_commit=lambda _receipt: events.append("commit"),
            on_abort=lambda receipt: events.append(("abort", receipt)),
        )

    assert ledger.state == "failed"
    assert events == [("abort", {})]

def test_in_memory_finalize_refuses_cleanup_when_live_patch_drifted(tmp_path) -> None:
    from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator

    project = tmp_path / "project"
    target = project / "src/main/java/demo/Feature.java"
    target.parent.mkdir(parents=True)
    target.write_text("expected generated source\n", encoding="utf-8")
    identity = "sha256:" + "e" * 64
    result = _checkpoint_result(project, identity)
    target.write_text("stale scaffold\n", encoding="utf-8")

    generator = object.__new__(CustomModuleGenerator)
    generator._checkpoint_cleanup_lock = __import__("threading").RLock()
    generator._checkpoint_cleanup_tokens = {}

    assert not generator.finalize_committed_generation_checkpoint(
        result,
        project_root=project,
    )
    assert result["generation_checkpoint"]["status"] == "AWAITING_LIVE_COMMIT"


def test_ensure_live_commit_does_not_replay_retired_checkpoint_delta(tmp_path) -> None:
    import threading
    from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
    from minecraft_mod_ai.generation_checkpoint import (
        _GenerationCheckpointLease,
        _initialize_generation_checkpoint,
    )

    project = tmp_path / "project"
    target = project / "src/main/java/demo/Feature.java"
    target.parent.mkdir(parents=True)
    target.write_text("stale scaffold\n", encoding="utf-8")
    checkpoint_base = tmp_path / "checkpoints"
    checkpoint_base.mkdir()
    identity = "sha256:" + "1" * 64
    checkpoint = checkpoint_base / identity.removeprefix("sha256:")
    staged = _initialize_generation_checkpoint(
        project,
        checkpoint,
        identity_sha256=identity,
    )
    staged_target = staged / "src/main/java/demo/Feature.java"
    staged_target.write_text("retired staged behavior\n", encoding="utf-8")

    token = "a" * 64
    result = _checkpoint_result(project, identity)
    result["generation_checkpoint"]["cleanup_token"] = token
    result["patch_receipt"]["operations"][0]["after_sha256"] = _sha256(
        b"retired staged behavior\n"
    )

    generator = object.__new__(CustomModuleGenerator)
    generator._checkpoint_cleanup_lock = threading.RLock()
    generator._checkpoint_cleanup_tokens = {
        token: (identity, checkpoint, _GenerationCheckpointLease(checkpoint))
    }
    try:
        assert not generator.ensure_generation_live_commit(
            result,
            project_root=project,
        )
        assert target.read_text(encoding="utf-8") == "stale scaffold\n"
    finally:
        owned = generator._checkpoint_cleanup_tokens.pop(token, None)
        if owned is not None:
            owned[2].close()
