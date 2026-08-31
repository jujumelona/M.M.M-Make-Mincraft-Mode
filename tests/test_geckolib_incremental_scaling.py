import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from minecraft_mod_ai import platform_specialized_generator_contract as specialized
from minecraft_mod_ai.geckolib_generator import (
    _entity_index_text,
    _registration_unit_files,
    _root_client,
    _root_reg,
    generate_geckolib_entity_assets,
    iter_geckolib_entity_records,
)
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.platform_catalog import adapter_from_project
from minecraft_mod_ai.production_hardener import harden_generated_project
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec


def _project(root: Path) -> Path:
    spec = ModSpec(
        mod_id="gecko_scale",
        mod_name="Gecko Scale",
        package_name="ai.minecraft.gecko_scale",
        version="1.0.0",
        summary="incremental GeckoLib scaling test",
        contents=(
            ContentSpec(
                content_id="bootstrap_item",
                kind=ContentKind.ITEM,
                display_name_en="Bootstrap Item",
                display_name_ko="Bootstrap Item",
            ),
        ),
    )
    FabricProjectGenerator().generate(spec, root)
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_paths(project: Path, entity_id: str) -> list[Path]:
    class_name = "".join(part.capitalize() for part in entity_id.split("_"))
    package = project / "src/main/java/ai/minecraft/gecko_scale"
    resources = project / "src/main/resources"
    return [
        project / f".minecraft_ai/geckolib-entities/{entity_id}.json",
        resources / f"data/gecko_scale/mmm_geckolib/entities/{entity_id}.json",
        resources / f"assets/gecko_scale/geo/{entity_id}.geo.json",
        resources / f"assets/gecko_scale/animations/{entity_id}.animation.json",
        resources / f"assets/gecko_scale/textures/entity/{entity_id}.png",
        package / f"entity/{class_name}Entity.java",
        package / f"client/geckolib/{class_name}GeoModel.java",
        package / f"client/geckolib/{class_name}GeoRenderer.java",
        package / f"geckolib/entry/{class_name}GeckoRegistration.java",
        package / f"client/geckolib/entry/{class_name}GeckoClientRegistration.java",
    ]


def test_entity_generation_is_incremental_and_does_not_replay_prior_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path / "project")
    adapter = adapter_from_project(project)
    approved = replace(
        adapter,
        deterministic_module_kinds=frozenset(
            {
                *adapter.deterministic_module_kinds,
                "entity",
                "geckolib:entity",
                "geckolib:version:4.8.2",
            }
        ),
    )
    monkeypatch.setattr(specialized, "adapter_from_project", lambda _root: approved)

    generate_geckolib_entity_assets(
        project_root=project,
        mod_id="gecko_scale",
        package_name="ai.minecraft.gecko_scale",
        entity_id="frost_guard",
    )
    first_paths = _unit_paths(project, "frost_guard")
    assert all(path.is_file() for path in first_paths)
    first_hashes = {path: _sha256(path) for path in first_paths}
    server_root = project / "src/main/java/ai/minecraft/gecko_scale/geckolib/GeneratedGeckoEntities.java"
    client_root = project / "src/main/java/ai/minecraft/gecko_scale/client/geckolib/GeneratedGeckoClient.java"
    root_hashes = (_sha256(server_root), _sha256(client_root))

    second = generate_geckolib_entity_assets(
        project_root=project,
        mod_id="gecko_scale",
        package_name="ai.minecraft.gecko_scale",
        entity_id="ember_guard",
    )
    assert {path: _sha256(path) for path in first_paths} == first_hashes
    assert (_sha256(server_root), _sha256(client_root)) == root_hashes
    second_operations = second["receipts"]["files"]["operations"]
    assert all(
        "frost_guard" not in operation["path"] and "FrostGuard" not in operation["path"]
        for operation in second_operations
    )

    second_paths = _unit_paths(project, "ember_guard")
    second_hashes = {path: _sha256(path) for path in second_paths}
    revised = generate_geckolib_entity_assets(
        project_root=project,
        mod_id="gecko_scale",
        package_name="ai.minecraft.gecko_scale",
        entity_id="frost_guard",
        max_health=120.0,
    )
    assert {path: _sha256(path) for path in second_paths} == second_hashes
    assert (_sha256(server_root), _sha256(client_root)) == root_hashes
    assert all(
        "ember_guard" not in operation["path"] and "EmberGuard" not in operation["path"]
        for operation in revised["receipts"]["files"]["operations"]
    )

    index = json.loads((project / ".minecraft_ai/geckolib-entities.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "mmm/geckolib-entity-index-v1"
    assert "entities" not in index
    assert [record["entity_id"] for record in iter_geckolib_entity_records(project)] == ["ember_guard", "frost_guard"]
    hardened = harden_generated_project(project, policy=ScalePolicy(java_shard_size=8))
    assert hardened["registry_definition_count"] == 2


def _entry(index: int) -> dict:
    entity_id = f"entity_{index:05d}"
    class_name = f"Entity{index:05d}"
    return {
        "entity_id": entity_id,
        "class_name": class_name,
        "entity_class": class_name + "Entity",
        "max_health": 80.0,
        "attack_damage": 8.0,
        "movement_speed": 0.27,
        "follow_range": 40.0,
        "entity_width": 0.8,
        "entity_height": 2.0,
        "archetype": "biped",
        "behavior": "hostile_melee",
        "spawn_group": "monster",
    }


def test_thousands_of_registration_units_do_not_grow_root_or_unit_files() -> None:
    package = "ai.minecraft.gecko_scale"
    mod_id = "gecko_scale"
    base = "src/main/java/ai/minecraft/gecko_scale"
    server_root = _root_reg(package, mod_id)
    client_root = _root_client(package)
    index = _entity_index_text(mod_id)

    small = _registration_unit_files(package, mod_id, base, _entry(0))
    small_max = max(len(value.encode("utf-8")) for value in small.values())
    large_max = 0
    for number in range(2_000):
        unit = _registration_unit_files(package, mod_id, base, _entry(number))
        assert len(unit) == 4
        large_max = max(large_max, *(len(value.encode("utf-8")) for value in unit.values()))

    assert large_max <= small_max + 256
    assert _root_reg(package, mod_id) == server_root
    assert _root_client(package) == client_root
    assert _entity_index_text(mod_id) == index
    assert "FabricDefaultAttributeRegistry.register" in next(
        value for path, value in small.items() if path.endswith("GeckoRegistration.java")
    )
    assert "EntityRendererRegistry.register" in next(
        value for path, value in small.items() if path.endswith("GeckoClientRegistration.java")
    )


def test_legacy_geckolib_manifest_remains_readable(tmp_path: Path) -> None:
    metadata = tmp_path / ".minecraft_ai"
    metadata.mkdir()
    record = _entry(7)
    (metadata / "geckolib-entities.json").write_text(
        json.dumps(
            {
                "schema_version": "mmm/geckolib-entities-v2",
                "shard_size": 24,
                "entities": [record],
            }
        ),
        encoding="utf-8",
    )

    assert list(iter_geckolib_entity_records(tmp_path)) == [record]
