from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import platform_runtime_contract


class _RuntimePolicyError(RuntimeError):
    pass


class _FakeManager:
    def __init__(self, *args, **kwargs):
        raise AssertionError("test bypasses wrapped __init__")

    def prepare_instance(
        self,
        instance_name,
        *,
        mod_jar,
        server_launcher,
        eula_accepted,
        expected_mod_sha256=None,
    ):
        return {
            "instance_name": instance_name,
            "mod_jar": str(mod_jar),
            "server_launcher": str(server_launcher),
        }

    def start_server(self, timeout_seconds=180):
        return {"status": "started", "timeout_seconds": timeout_seconds}

    @staticmethod
    def _process_running(_process):
        return False

    @staticmethod
    def _existing_file(value):
        if value is None:
            raise _RuntimePolicyError("missing file")
        path = Path(value).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise _RuntimePolicyError("missing file")
        return path


def _manager_module():
    return SimpleNamespace(
        MinecraftRuntimeManager=_FakeManager,
        RuntimePolicyError=_RuntimePolicyError,
        RuntimeProfile=lambda **kwargs: SimpleNamespace(**kwargs),
        resolve_config_path=lambda _name: Path("unused"),
    )


def _prepared_manager():
    manager = object.__new__(_FakeManager)
    manager.server_process = None
    manager.client_process = None
    manager._mmm_platform_adapter = SimpleNamespace(
        adapter_id="fabric:test",
        java_version="17",
    )
    manager.profile = SimpleNamespace(server_java_command="java")
    return manager


def test_start_server_does_not_require_prepare_artifact_kwargs(monkeypatch) -> None:
    module = _manager_module()
    monkeypatch.setattr(
        platform_runtime_contract,
        "_validate_java_command",
        lambda *args, **kwargs: None,
    )
    platform_runtime_contract._install_runtime_manager(module)
    manager = _prepared_manager()

    assert manager.start_server(timeout_seconds=7) == {
        "status": "started",
        "timeout_seconds": 7,
    }


def test_prepare_instance_owns_mod_and_launcher_file_validation(tmp_path, monkeypatch) -> None:
    module = _manager_module()
    monkeypatch.setattr(
        platform_runtime_contract,
        "_validate_java_command",
        lambda *args, **kwargs: None,
    )
    platform_runtime_contract._install_runtime_manager(module)
    manager = _prepared_manager()
    mod = tmp_path / "mod.jar"
    launcher = tmp_path / "fabric-server-launch.jar"
    mod.write_bytes(b"mod")
    launcher.write_bytes(b"launcher")

    result = manager.prepare_instance(
        "runtime_case",
        mod_jar=mod,
        server_launcher=launcher,
        eula_accepted=True,
    )

    assert result["mod_jar"] == str(mod)
    assert result["server_launcher"] == str(launcher)
    assert result["platform_adapter"] == "fabric:test"
    assert result["java_version"] == "17"
