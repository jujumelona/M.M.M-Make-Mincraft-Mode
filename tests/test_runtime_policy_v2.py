from pathlib import Path

import pytest

from minecraft_mod_ai.runtime_manager import MinecraftRuntimeManager, RuntimePolicyError


def test_runtime_instance_is_disposable_and_requires_explicit_eula(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mod = workspace / "mod.jar"
    launcher = workspace / "server.jar"
    mod.write_bytes(b"jar")
    launcher.write_bytes(b"server")
    manager = MinecraftRuntimeManager(workspace)
    with pytest.raises(RuntimePolicyError, match="EULA"):
        manager.prepare_instance(
            "test_instance",
            mod_jar=mod,
            server_launcher=launcher,
            eula_accepted=False,
        )
    result = manager.prepare_instance(
        "test_instance",
        mod_jar=mod,
        server_launcher=launcher,
        eula_accepted=True,
    )
    root = Path(result["instance_root"])
    assert result["disposable"] is True
    assert (root / "eula.txt").read_text(encoding="utf-8") == "eula=true\n"
    assert (root / "mods" / "mod.jar").is_file()
    manager.cleanup()
    assert not root.exists()
