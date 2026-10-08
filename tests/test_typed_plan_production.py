from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.typed_plan_production import generate_typed_plan_module
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.work_graph import _is_typed_host_module, _module_shards, _node



def _project(root: Path) -> Path:
    spec = ModSpec(
        mod_id="typedtest",
        mod_name="Typed Test",
        package_name="ai.minecraft.typedtest",
        version="1.0.0",
        summary="typed plan test",
        contents=(
            ContentSpec(
                content_id="marker_item",
                kind=ContentKind.ITEM,
                display_name_en="Marker",
                display_name_ko="마커",
            ),
        ),
    )
    FabricProjectGenerator().generate(spec, root)
    return root


def _plan(text: str = "typed behavior") -> dict:
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "functions": [
            {
                "id": "compute",
                "parameters": [],
                "return_type": "int",
                "body": [
                    {
                        "op": "return",
                        "value": {"op": "literal", "type": "int", "value": 42},
                    }
                ],
                "covers": [],
            }
        ],
        "initialize": [
            {
                "op": "expr",
                "value": {
                    "op": "call",
                    "function": "compute",
                    "args": [],
                },
            }
        ],
    }


def test_typed_plan_backend_generates_compileable_java_without_router(tmp_path: Path) -> None:
    root = _project(tmp_path)
    module = ProductionModule(
        module_id="authored_typed_plan",
        kind="typed_host",
        config={
            "typed_plan_ir": _plan(),
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {},
        },
        required_gates=("target_compile",),
    )

    receipt = generate_typed_plan_module(
        root,
        module=module,
    )

    assert receipt["generation_verification"]["model_calls"] == 0
    source_path = (
        root / "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "// MMM:TYPED_PLAN_OWNER" in source
    assert "return 42;" in source

    javac = shutil.which("javac")
    if not javac:
        pytest.skip("JDK required")
    classes = root / ".typed-test-classes"
    classes.mkdir()
    compiled = subprocess.run(
        [javac, "-encoding", "UTF-8", "-d", str(classes), str(source_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


def test_typed_state_backend_generates_state_owner_without_router(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    text = "stateful behavior"
    typed = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "functions": [
            {
                "id": "readCoins",
                "parameters": [],
                "return_type": "int",
                "body": [
                    {
                        "op": "return",
                        "value": {
                            "op": "state_get",
                            "key": {
                                "op": "literal",
                                "type": "string",
                                "value": "coins",
                            },
                            "type": "int",
                            "context": {"op": "map", "entries": []},
                        },
                    }
                ],
                "covers": [],
            }
        ],
        "initialize": [],
    }
    state = {
        "specification": {
            "variables": [
                {
                    "name": "coins",
                    "owner": "player",
                    "type": "long",
                    "unit": "credits",
                    "default": "0",
                    "domain": "integer >= 0",
                }
            ],
            "transitions": [],
            "invariants": [],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
            "inapplicable_concerns": [],
        },
        "constraint_evidence_refs": [],
    }
    module = ProductionModule(
        module_id="authored_typed_state",
        kind="typed_host",
        config={
            "typed_plan_ir": typed,
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {},
            "typed_plan_state_section": state,
        },
        required_gates=("target_compile",),
    )

    receipt = generate_typed_plan_module(
        root,
        module=module,
    )

    assert receipt["generation_verification"]["model_calls"] == 0
    package_root = root / "src/main/java/ai/minecraft/typedtest"
    program = package_root / "AuthoredProgram.java"
    state_owner = package_root / "AuthoredStateModel.java"
    assert program.is_file()
    assert state_owner.is_file()
    assert "// MMM:TYPED_PLAN_STATE_OWNER" in state_owner.read_text(encoding="utf-8")

    javac = shutil.which("javac")
    if not javac:
        pytest.skip("JDK required")
    classes = root / ".typed-state-test-classes"
    classes.mkdir()
    compiled = subprocess.run(
        [
            javac,
            "-encoding",
            "UTF-8",
            "-d",
            str(classes),
            str(state_owner),
            str(program),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


def test_typed_state_operations_fail_closed_without_canonical_state_authority() -> None:
    text = "stateful behavior"
    typed = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "functions": [
            {
                "id": "readCoins",
                "parameters": [],
                "return_type": "int",
                "body": [
                    {
                        "op": "return",
                        "value": {
                            "op": "state_get",
                            "key": {
                                "op": "literal",
                                "type": "string",
                                "value": "coins",
                            },
                            "type": "int",
                            "context": {
                                "op": "map",
                                "entries": [],
                            },
                        },
                    }
                ],
                "covers": [],
            }
        ],
        "initialize": [
            {
                "op": "expr",
                "value": {
                    "op": "call",
                    "function": "readCoins",
                    "args": [],
                },
            }
        ],
    }
    plan = AuthoredPlan(
        requested_prompt="stateful mod",
        text=text,
        typed_plan_ir=typed,
    )

    with pytest.raises(ValueError, match="TYPED_PLAN_STATE_AUTHORITY_REQUIRED"):
        _compile_new_authored_modules(
            plan,
            mod_id="typedtest",
            package_name="ai.minecraft.typedtest",
            target={},
            production_state_section={},
        )

def test_typed_plan_backend_refuses_unowned_existing_source(tmp_path: Path) -> None:
    root = _project(tmp_path)
    source_path = (
        root / "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
    )
    source_path.write_text(
        "package ai.minecraft.typedtest; public final class AuthoredProgram {}\n",
        encoding="utf-8",
    )
    module = ProductionModule(
        module_id="authored_typed_plan",
        kind="typed_host",
        config={
            "typed_plan_ir": _plan(),
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {},
        },
        required_gates=("target_compile",),
    )

    with pytest.raises(ValueError, match="TYPED_PLAN_OWNERSHIP_CONFLICT"):
        generate_typed_plan_module(
            root,
            module=module,
        )


def test_typed_plan_host_work_is_isolated_from_llm_shards() -> None:
    typed = ProductionModule(
        module_id="typed",
        kind="typed_host",
        config={
            "typed_plan_ir": _plan(),
        },
    )
    legacy = ProductionModule(
        module_id="legacy",
        kind="item",
        config={},
    )

    assert _is_typed_host_module(typed) is True
    shards = list(
        _module_shards(
            (typed, legacy),
            policy=ScalePolicy.from_environment(),
        )
    )
    assert len(shards) == 2
    typed_stage, typed_members = shards[0]
    assert typed_stage == "host"
    assert [member.module_id for member in typed_members] == ["typed"]

    node = _node(
        "typed-host",
        "generate:host",
        (),
        {
            "kind": "module-shard",
            "generation_stage": "host",
            "members": [
                {
                    "module_id": "typed",
                    "kind": "typed_host",
                    "config": {"typed_plan_ir": _plan()},
                }
            ],
        },
    )
    assert node.resource_class == "cpu_io"

def test_typed_platform_modules_lower_to_deterministic_production_modules() -> None:
    text = "item behavior"
    typed = {
        **_plan(text),
        "platform_modules": [
            {
                "module_id": "marker_token",
                "kind": "item",
                "config": {"display_name_en": "Marker Token"},
                "covers": ["resources_and_ui.registries"],
            }
        ],
    }
    plan = AuthoredPlan(
        requested_prompt="add marker token",
        text=text,
        structured_sections=_structured_section(
            "resources_and_ui",
            "registries",
            [
                {
                    "purpose": "item",
                    "identifier": "marker_token",
                    "binding_requirement": "register the marker token",
                }
            ],
        ),
        typed_plan_ir=typed,
    )

    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="typedtest",
        package_name="ai.minecraft.typedtest",
        target={},
        production_state_section={},
    )

    assert [module.module_id for module in modules] == [
        "authored_typed_plan",
        "marker_token",
    ]
    assert modules[1].kind == "item"
    assert modules[1].config["display_name_en"] == "Marker Token"
    assert manifest["platform_modules"] == [
        {
            "module_id": "marker_token",
            "kind": "item",
            "covers": ["resources_and_ui.registries"],
        }
    ]


def test_typed_state_store_generates_state_owner_and_persistence_bridge(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    state = {
        "specification": {
            "variables": [
                {
                    "name": "coins",
                    "owner": "player",
                    "type": "long",
                    "unit": "credits",
                    "default": "0",
                    "domain": "integer >= 0",
                }
            ],
            "transitions": [],
            "invariants": [],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
            "inapplicable_concerns": [],
        },
        "constraint_evidence_refs": [],
    }
    module = ProductionModule(
        module_id="authored_typed_plan",
        kind="typed_host",
        config={
            "typed_plan_ir": {
                **_plan(),
                "platform_modules": [
                    {
                        "module_id": "persistent_state",
                        "kind": "state_store",
                        "config": {"namespace": "player_state"},
                        "covers": ["persistence.stored_state"],
                    }
                ],
            },
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {},
            "typed_plan_state_section": state,
            "typed_state_store": {"namespace": "player_state"},
        },
        required_gates=("target_compile",),
    )

    receipt = generate_typed_plan_module(
        root,
        module=module,
    )

    package_root = root / "src/main/java/ai/minecraft/typedtest"
    state_owner = package_root / "AuthoredStateModel.java"
    bridge = package_root / "AuthoredStatePersistence.java"
    store = package_root / "system/MmmPersistentStore.java"
    assert receipt["generation_verification"]["model_calls"] == 0
    assert state_owner.is_file()
    assert bridge.is_file()
    assert store.is_file()
    bridge_text = bridge.read_text(encoding="utf-8")
    assert "// MMM:TYPED_STATE_PERSISTENCE_OWNER" in bridge_text
    assert 'MmmPersistentStore.namespace("player_state")' in bridge_text
    assert 'AuthoredStateModel.getState("coins")' in bridge_text
    assert 'AuthoredStateModel.setState("coins", data.get("coins"))' in bridge_text

    main_source = next(
        path
        for path in (root / "src/main/java/ai/minecraft/typedtest").glob("*.java")
        if "implements ModInitializer" in path.read_text(encoding="utf-8")
    )
    main_text = main_source.read_text(encoding="utf-8")
    assert "AuthoredStatePersistence.register();" in main_text

def _structured_section(section: str, concern: str, rows: list[dict]) -> dict:
    specification = {
        name: []
        for name in DETAIL_RECORDS[section]
    }
    specification[concern] = rows
    specification["inapplicable_concerns"] = []
    return {
        section: {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }


def test_typed_platform_content_lowers_to_deterministic_module() -> None:
    text = "typed platform content"
    typed = _plan(text)
    typed["platform_modules"] = [
        {
            "module_id": "marker_item",
            "kind": "item",
            "config": {"display_name_en": "Marker"},
            "covers": ["resources_and_ui.registries"],
        }
    ]
    structured = _structured_section(
        "resources_and_ui",
        "registries",
        [
            {
                "purpose": "item",
                "identifier": "marker_item",
                "binding_requirement": "register the marker item",
            }
        ],
    )
    plan = AuthoredPlan(
        requested_prompt="marker item",
        text=text,
        structured_sections=structured,
        typed_plan_ir=typed,
    )

    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="typedtest",
        package_name="ai.minecraft.typedtest",
        target={},
        production_state_section={},
    )

    assert [(module.module_id, module.kind) for module in modules] == [
        ("authored_typed_plan", "typed_host"),
        ("marker_item", "item"),
    ]
    assert modules[1].config == {"display_name_en": "Marker"}
    assert manifest["policy"] == "host_typed_plan_ir"
    assert manifest["platform_modules"] == [
        {
            "module_id": "marker_item",
            "kind": "item",
            "covers": ["resources_and_ui.registries"],
        }
    ]


def test_typed_state_store_materializes_persistence_without_router(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    text = "persistent typed state"
    typed = _plan(text)
    typed["platform_modules"] = [
        {
            "module_id": "persistent_state",
            "kind": "state_store",
            "config": {"namespace": "player_state"},
            "covers": ["persistence.stored_state"],
        }
    ]
    state_section = _structured_section(
        "state_model",
        "variables",
        [
            {
                "name": "coins",
                "owner": "player",
                "type": "long",
                "unit": "credits",
                "default": "0",
                "domain": "integer >= 0",
            }
        ],
    )["state_model"]
    persistence_section = _structured_section(
        "persistence",
        "stored_state",
        [
            {
                "state": "coins",
                "owner": "player",
                "scope": "world",
            }
        ],
    )["persistence"]
    plan = AuthoredPlan(
        requested_prompt="persistent state",
        text=text,
        structured_sections={
            "state_model": state_section,
            "persistence": persistence_section,
        },
        typed_plan_ir=typed,
    )

    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="typedtest",
        package_name="ai.minecraft.typedtest",
        target={},
        production_state_section=state_section,
    )

    assert len(modules) == 1
    module = modules[0]
    assert module.module_id == "authored_typed_plan"
    state_store = module.config["typed_state_store"]
    assert state_store["namespace"] == "player_state"
    assert state_store["schema_version"].startswith("state-")
    assert state_store["migrations"] == []
    assert state_store["malformed_policy"] == "backup_and_reset"
    assert state_store["transfer_on_respawn"] is False
    assert manifest["typed_program"]["state_store"] is True

    receipt = generate_typed_plan_module(
        root,
        module=module,
    )

    assert receipt["generation_verification"]["model_calls"] == 0
    package_root = root / "src/main/java/ai/minecraft/typedtest"
    assert (package_root / "AuthoredStateModel.java").is_file()
    bridge = package_root / "AuthoredStatePersistence.java"
    assert bridge.is_file()
    assert "// MMM:TYPED_STATE_PERSISTENCE_OWNER" in bridge.read_text(
        encoding="utf-8"
    )
    assert (package_root / "system/MmmPersistentStore.java").is_file()
    main_text = (
        package_root / "TypedtestMod.java"
    ).read_text(encoding="utf-8")
    assert "AuthoredStatePersistence.register();" in main_text

def test_typed_host_capability_generates_owned_java_without_router(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    text = "notify player"
    typed = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "functions": [
            {
                "id": "notify",
                "parameters": [{"name": "player", "type": "object"}],
                "return_type": "void",
                "body": [
                    {
                        "op": "expr",
                        "value": {
                            "op": "capability",
                            "id": "player.send_message",
                            "args": [
                                {"op": "ref", "name": "player"},
                                {
                                    "op": "literal",
                                    "type": "string",
                                    "value": "hello",
                                },
                            ],
                        },
                    },
                    {"op": "return"},
                ],
                "covers": ["behavior_contract.outputs"],
            }
        ],
        "initialize": [],
    }
    contracts = typed_host_capability_contracts()
    module = ProductionModule(
        module_id="authored_typed_plan",
        kind="typed_host",
        config={
            "typed_plan_ir": typed,
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {
                "player.send_message": contracts["player.send_message"],
            },
        },
        required_gates=("target_compile",),
    )

    receipt = generate_typed_plan_module(
        root,
        module=module,
    )

    assert receipt["generation_verification"]["model_calls"] == 0
    package_root = root / "src/main/java/ai/minecraft/typedtest"
    program = (package_root / "AuthoredProgram.java").read_text(
        encoding="utf-8"
    )
    capability_source = (
        package_root / "AuthoredHostCapabilities.java"
    ).read_text(encoding="utf-8")
    assert "AuthoredHostCapabilities.sendMessage" in program
    assert "// MMM:TYPED_HOST_CAPABILITIES_OWNER" in capability_source
    assert "public static void sendMessage" in capability_source



def test_host_capabilities_select_mojang_names_for_java25_target() -> None:
    # 2026-10-08: 26.1 is unobfuscated. Emitting Yarn names must fail the
    # version-specific renderer test before the real Gradle gate is scheduled.
    from minecraft_mod_ai.typed_host_capabilities import (
        render_typed_host_capabilities_java,
    )

    modern = render_typed_host_capabilities_java(
        "example.mod", minecraft_version="26.1.2"
    )
    legacy = render_typed_host_capabilities_java(
        "example.mod", minecraft_version="1.21.5"
    )
    assert "net.minecraft.world.effect.MobEffectInstance" in modern
    assert "net.minecraft.server.level.ServerPlayer" in modern
    assert "net.minecraft.resources.Identifier" in modern
    assert "BuiltInRegistries.MOB_EFFECT.get(ResourceKey.create(Registries.MOB_EFFECT, id))" in modern
    assert "MOB_EFFECT.getHolder(id)" not in modern
    assert "Identifier.parse(" in modern
    assert "net.minecraft.entity.effect.StatusEffectInstance" not in modern
    assert "net.minecraft.server.network.ServerPlayerEntity" not in modern
    assert "hasPermissionLevel(" not in modern
    assert "net.minecraft.entity.effect.StatusEffectInstance" in legacy
    assert "net.minecraft.server.network.ServerPlayerEntity" in legacy
