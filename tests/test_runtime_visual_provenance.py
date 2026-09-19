from __future__ import annotations

import threading

import pytest

from minecraft_mod_ai.complete_orchestrator import _collect_runtime_screenshot_receipts
from minecraft_mod_ai.runtime_manager import MinecraftRuntimeManager, RuntimePolicyError


class _RunningProcess:
    @staticmethod
    def poll():
        return None


def _manager(tmp_path):
    manager = object.__new__(MinecraftRuntimeManager)
    manager.workspace_root = tmp_path.resolve()
    manager.instance_root = tmp_path / "runtime-instances" / "case"
    manager.instance_root.mkdir(parents=True)
    manager.server_process = _RunningProcess()
    manager.client_process = _RunningProcess()
    manager._server_log = []
    manager._client_log = []
    manager._lock = threading.RLock()
    manager._log_lock = threading.RLock()
    return manager


def test_runtime_screenshot_receipt_is_bound_to_current_client(tmp_path) -> None:
    manager = _manager(tmp_path)
    screenshot = manager.instance_root / "client" / "screenshots" / "proof.png"
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"current-runtime-image")

    receipt = manager.register_screenshot(screenshot)

    assert receipt["schema_version"] == "mmm/runtime-screenshot-v2"
    assert receipt["server_running"] is True
    assert receipt["client_running"] is True
    assert receipt["instance_root"] == str(manager.instance_root)
    assert receipt["sha256"].startswith("sha256:")
    assert len(receipt["sha256"]) == 71


def test_runtime_screenshot_rejects_external_workspace_image(tmp_path) -> None:
    manager = _manager(tmp_path)
    outside = tmp_path / "old-proof.png"
    outside.write_bytes(b"stale-image")

    with pytest.raises(
        RuntimePolicyError,
        match="current disposable client directory",
    ):
        manager.register_screenshot(outside)


def test_runtime_screenshot_is_preserved_before_disposable_cleanup(tmp_path) -> None:
    manager = _manager(tmp_path)
    screenshot = manager.instance_root / "client" / "screenshots" / "proof.png"
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"current-runtime-image")

    receipts = _collect_runtime_screenshot_receipts(manager, ())
    assert len(receipts) == 1
    receipt = receipts[0]
    evidence = __import__("pathlib").Path(receipt["evidence_path"])
    source = __import__("pathlib").Path(receipt["runtime_source_path"])

    assert evidence.is_file()
    assert receipt["path"] == str(evidence)
    assert receipt["sha256"] == manager._sha256_file(evidence)

    source.unlink()
    assert not source.exists()
    assert evidence.is_file()
    assert manager._sha256_file(evidence) == receipt["sha256"]
