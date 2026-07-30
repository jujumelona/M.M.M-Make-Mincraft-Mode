from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.complete_orchestrator_services import run_playtest
from minecraft_mod_ai.complete_orchestrator_support import (
    CompleteProductionError,
    _handled_module_ids,
    _normalize_modules,
    _system_groups,
)
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.extended_content_generator import generate_extended_content
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.production_hardener import harden_generated_project
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.system_pack_generator import generate_system_pack


def _spec() -> ModSpec:
    return ModSpec(
        mod_id="closed_test",
        mod_name="Closed Test",
        package_name="ai.minecraft.closed_test",
        version="1.0.0",
        summary="fail closed production tests",
        contents=(
            ContentSpec(
                content_id="bootstrap_item",
                kind=ContentKind.ITEM,
                display_name_en="Bootstrap Item",
                display_name_ko="부트스트랩 아이템",
            ),
        ),
    )


def _project(root: Path) -> Path:
    FabricProjectGenerator().generate(_spec(), root)
    return root


def test_empty_or_observation_only_playtest_cannot_pass() -> None:
    with pytest.raises(CompleteProductionError, match="explicit playtest"):
        run_playtest(())
    with pytest.raises(CompleteProductionError, match="gameplay interaction"):
        run_playtest(({"action": "status", "params": {}},))
    with pytest.raises(CompleteProductionError, match="include wait_for"):
        run_playtest(
            (
                {
                    "action": "chat",
                    "params": {"message": "/mmmquest list"},
                },
            )
        )


def test_builtin_quest_rejects_unimplemented_objective(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    with pytest.raises(ValueError, match="use custom_java"):
        generate_system_pack(
            project_root=project,
            pack_id="quest-system",
            mod_id="closed_test",
            package_name="ai.minecraft.closed_test",
            config={
                "modules": [
                    {
                        "module_id": "escort_merchant",
                        "kind": "quest",
                        "config": {
                            "objective": "escort",
                            "target": "closed_test:merchant",
                        },
                        "depends_on": [],
                        "required_gates": [],
                    }
                ]
            },
        )


def test_builtin_shop_requires_server_owned_catalog(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    with pytest.raises(ValueError, match="non-empty entries"):
        generate_system_pack(
            project_root=project,
            pack_id="economy-shop",
            mod_id="closed_test",
            package_name="ai.minecraft.closed_test",
            config={
                "modules": [
                    {
                        "module_id": "bad_shop",
                        "kind": "shop",
                        "config": {},
                        "depends_on": [],
                        "required_gates": [],
                    }
                ]
            },
        )


def test_explicit_custom_module_is_normalized_before_builtin_generators() -> None:
    module = ProductionModule(
        module_id="escort_system",
        kind="quest",
        config={
            "implementation": "custom",
            "objective": "escort",
        },
    )
    normalized, receipts = _normalize_modules((module,), _spec())
    assert len(normalized) == 1
    assert normalized[0].kind == "custom_java"
    assert normalized[0].config["requested_kind"] == "quest"
    assert "implementation" not in normalized[0].config
    assert _system_groups(normalized) == {}
    assert normalized[0].module_id not in _handled_module_ids(normalized)
    assert receipts[0]["status"] == "ROUTED_CUSTOM"


def test_gui_networking_generation_is_read_only_and_replay_safe(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    result = generate_system_pack(
        project_root=project,
        pack_id="gui-networking",
        mod_id="closed_test",
        package_name="ai.minecraft.closed_test",
        config={
            "modules": [
                {
                    "module_id": "guide_menu",
                    "kind": "gui",
                    "config": {
                        "template": "read_only_menu",
                        "title": "Guide",
                        "rows": 1,
                        "entries": [
                            {
                                "slot": 0,
                                "item": "minecraft:book",
                                "count": 1,
                            }
                        ],
                    },
                    "depends_on": [],
                    "required_gates": [],
                },
                {
                    "module_id": "guide_actions",
                    "kind": "networking",
                    "config": {
                        "template": "validated_action_channel",
                        "actions": [
                            {
                                "id": "claim_book",
                                "type": "grant_item",
                                "item": "minecraft:book",
                                "count": 1,
                            }
                        ],
                    },
                    "depends_on": [],
                    "required_gates": [],
                },
            ]
        },
    )
    assert result["status"] == "fabric_binding_generated"
    java = next(
        path
        for path in map(Path, result["files"])
        if path.name == "GuiNetworkingSystem.java"
    ).read_text(encoding="utf-8")
    assert "ReadOnlyMenuHandler" in java
    assert "canTakeItems" in java and "return false" in java
    assert "NETWORK_COOLDOWN_TICKS" in java
    assert 'namespace("network_actions")' in java
    assert '"grant_item".equals' in java


def test_manual_quest_progress_targets_only_requested_id(tmp_path: Path) -> None:
    project = _project(tmp_path / "project")
    result = generate_system_pack(
        project_root=project,
        pack_id="quest-system",
        mod_id="closed_test",
        package_name="ai.minecraft.closed_test",
        config={
            "modules": [
                {
                    "module_id": "first_manual",
                    "kind": "quest",
                    "config": {"objective": "manual", "required": 2},
                    "depends_on": [],
                    "required_gates": [],
                },
                {
                    "module_id": "second_manual",
                    "kind": "quest",
                    "config": {"objective": "manual", "required": 2},
                    "depends_on": [],
                    "required_gates": [],
                },
            ]
        },
    )
    java = next(
        path
        for path in map(Path, result["files"])
        if path.name == "QuestSystem.java"
    ).read_text(encoding="utf-8")
    assert 'if (!definition.objective().equals(objective)) continue;' in java
    assert 'if (!definition.id().equals(target)) continue;' in java


def test_hardener_adds_machine_model_and_registry_gametest(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "project")
    generate_extended_content(
        project_root=project,
        mod_id="closed_test",
        package_name="ai.minecraft.closed_test",
        modules=(
            ProductionModule(
                module_id="processor",
                kind="machine",
                config={
                    "input_item": "minecraft:iron_ingot",
                    "output_item": "minecraft:gold_ingot",
                    "processing_ticks": 20,
                },
            ),
        ),
        policy=ScalePolicy(java_shard_size=8),
    )
    result = harden_generated_project(
        project,
        policy=ScalePolicy(java_shard_size=8),
    )
    assert result["status"] == "HARDENED"
    root_java = project / (
        "src/main/java/ai/minecraft/closed_test/extended/"
        "GeneratedExtendedContent.java"
    )
    text = root_java.read_text(encoding="utf-8")
    assert "BlockRenderType.MODEL" in text
    tests = sorted(project.rglob("GeneratedRegistryGameTest*.java"))
    assert tests
    assert "processor" in tests[0].read_text(encoding="utf-8")
    metadata = json.loads(
        (project / "src/main/resources/fabric.mod.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(
        "GeneratedRegistryGameTest" in entry
        for entry in metadata["entrypoints"]["fabric-gametest"]
    )
