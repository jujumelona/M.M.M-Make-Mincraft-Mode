from __future__ import annotations

from minecraft_mod_ai.capabilities import capability_names
from minecraft_mod_ai.capability_plugins import buildable_plugin_ids
from minecraft_mod_ai.mod_development_methods import resolve_mod_development_methods


def test_baseline_mod_methods_exclude_standalone_map() -> None:
    result = resolve_mod_development_methods(
        "간단한 음식 아이템을 추가하는 모드를 만들어줘"
    )

    assert result["standalone_map_generation"] is False
    assert "fabric_project_contract" in result["method_ids"]
    assert "registry_and_datagen" in result["method_ids"]
    assert "content_registry" in result["method_ids"]
    assert "fabric_worldgen" not in result["method_ids"]


def test_worldgen_is_mod_owned_and_explicit() -> None:
    result = resolve_mod_development_methods(
        "새 바이옴과 구조물을 추가하는 Fabric 모드를 만들어줘"
    )

    assert "fabric_worldgen" in result["method_ids"]
    assert result["standalone_map_generation"] is False


def test_standalone_map_request_is_flagged_not_selected() -> None:
    result = resolve_mod_development_methods(
        "월드 ZIP과 litematic 맵 파일을 만들어줘"
    )

    assert result["standalone_map_requested"] is True
    assert result["standalone_map_generation"] is False
    assert "fabric_worldgen" not in result["method_ids"]


def test_capabilities_have_no_map_compiler() -> None:
    names = capability_names()

    assert "world.ir.generate" not in names
    assert "world.compile" not in names
    assert "fabric.worldgen.generate" in names


def test_plugins_have_no_standalone_map_builder() -> None:
    plugin_ids = buildable_plugin_ids()

    assert "worldgen-map" not in plugin_ids
    assert "worldgen-arena" not in plugin_ids
    assert "fabric-worldgen" in plugin_ids
    assert "mod-development-methods" in plugin_ids
