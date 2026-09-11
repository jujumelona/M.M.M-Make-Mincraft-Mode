from __future__ import annotations

import zipfile
from pathlib import Path

from setuptools import build_meta


def test_wheel_contains_all_python_packages_and_runtime_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    wheel_name = build_meta.build_wheel(str(tmp_path))
    wheel_path = tmp_path / wheel_name
    assert wheel_path.is_file()

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    package_inits = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "minecraft_mod_ai").rglob("__init__.py")
    }
    missing_packages = sorted(package_inits - names)
    assert not missing_packages, f"wheel omitted Python packages: {missing_packages}"

    required_resources = {
        "download_resources.py",
        "minecraft_mod_ai/config/agent_roles.yaml",
        "minecraft_mod_ai/config/external_mcp_registry.yaml",
        "minecraft_mod_ai/config/model_registry.yaml",
        "minecraft_mod_ai/config/runtime_profiles.yaml",
        "minecraft_mod_ai/integrations/mineflayer/bridge.mjs",
        "minecraft_mod_ai/integrations/mineflayer/package.json",
        "minecraft_mod_ai/templates/prompt/capture.yaml",
    }
    missing_resources = sorted(required_resources - names)
    assert not missing_resources, f"wheel omitted runtime resources: {missing_resources}"

    assert "minecraft_mod_ai/config/__init__.py" not in names
    assert "minecraft_mod_ai/integrations/__init__.py" not in names
    assert "integrations/mineflayer/bridge.mjs" not in names
