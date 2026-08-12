from __future__ import annotations

import io
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.runtime_manager as runtime_module
from minecraft_mod_ai.runtime_manager import (
    MinecraftRuntimeManager,
    RuntimePolicyError,
)


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mod = workspace / "mod.jar"
    launcher = workspace / "server.jar"
    mod.write_bytes(b"mod")
    launcher.write_bytes(b"server")
    return workspace, mod, launcher


def _prepare(
    manager: MinecraftRuntimeManager,
    mod: Path,
    launcher: Path,
    *,
    name: str = "test_instance",
) -> Path:
    result = manager.prepare_instance(
        name,
        mod_jar=mod,
        server_launcher=launcher,
        eula_accepted=True,
    )
    return Path(result["instance_root"])


def test_prepare_validates_inputs_before_reserving_instance_name(
    tmp_path: Path,
) -> None:
    workspace, _mod, launcher = _workspace(tmp_path)
    manager = MinecraftRuntimeManager(workspace)

    with pytest.raises(FileNotFoundError):
        manager.prepare_instance(
            "retryable_instance",
            mod_jar=workspace / "missing.jar",
            server_launcher=launcher,
            eula_accepted=True,
        )

    assert not (workspace / "runtime-instances" / "retryable_instance").exists()

    mod = workspace / "valid.jar"
    mod.write_bytes(b"valid")
    root = _prepare(
        manager,
        mod,
        launcher,
        name="retryable_instance",
    )
    assert root.is_dir()
    manager.cleanup()


def test_prepare_rejects_instance_swap_while_runtime_process_is_live(
    tmp_path: Path,
) -> None:
    workspace, mod, launcher = _workspace(tmp_path)
    manager = MinecraftRuntimeManager(workspace)
    manager.server_process = SimpleNamespace(poll=lambda: None)  # type: ignore[assignment]

    with pytest.raises(RuntimePolicyError, match="Stop the active"):
        manager.prepare_instance(
            "blocked_instance",
            mod_jar=mod,
            server_launcher=launcher,
            eula_accepted=True,
        )

    assert not (workspace / "runtime-instances" / "blocked_instance").exists()


def test_cleanup_failure_preserves_instance_binding_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, mod, launcher = _workspace(tmp_path)
    manager = MinecraftRuntimeManager(workspace)
    root = _prepare(manager, mod, launcher)

    real_rmtree = runtime_module.shutil.rmtree

    def fail_once(path, *args, **kwargs):
        if Path(path) == root:
            raise OSError("simulated delete failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(runtime_module.shutil, "rmtree", fail_once)
    with pytest.raises(OSError, match="simulated delete failure"):
        manager.cleanup()

    assert manager.instance_root == root
    assert root.is_dir()

    monkeypatch.setattr(runtime_module.shutil, "rmtree", real_rmtree)
    manager.cleanup()
    assert manager.instance_root is None
    assert not root.exists()


def test_log_reader_and_tail_use_independent_log_lock(
    tmp_path: Path,
) -> None:
    workspace, _mod, _launcher = _workspace(tmp_path)
    manager = MinecraftRuntimeManager(workspace)
    fake_process = SimpleNamespace(stdout=io.StringIO("one\ntwo\nthree\n"))

    thread = threading.Thread(
        target=manager._read_stream,
        args=(fake_process, manager._server_log),
    )
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()

    assert manager.tail_logs(lines=2)["server"] == ["two", "three"]
