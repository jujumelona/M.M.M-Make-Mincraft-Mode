from __future__ import annotations

from pathlib import Path

import pytest

import minecraft_mod_ai.config_paths as config_paths
import minecraft_mod_ai.mineflayer_bridge as mineflayer_bridge


def test_config_path_uses_packaged_canonical_directory() -> None:
    expected = Path(config_paths.__file__).resolve().parent / "config" / "model_registry.yaml"
    assert config_paths.config_path("model_registry.yaml") == expected.resolve()
    assert expected.is_file()


def test_config_path_rejects_parent_escape() -> None:
    with pytest.raises(ValueError, match="escapes package root"):
        config_paths.config_path("../pyproject.toml")


def test_mineflayer_default_resolves_to_packaged_canonical_bridge() -> None:
    packaged = (
        Path(mineflayer_bridge.__file__).resolve().parent
        / "integrations"
        / "mineflayer"
        / "bridge.mjs"
    )
    assert mineflayer_bridge._default_bridge_path().resolve() == packaged.resolve()
    assert packaged.is_file()
    assert (packaged.parent / "package.json").is_file()
