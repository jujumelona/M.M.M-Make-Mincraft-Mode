from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .generator import make_texture_png
from .project_edit import (
    ensure_client_entrypoint,
    ensure_dependency,
    ensure_main_initializer_call,
    inspect_fabric_project,
    write_text_files,
)
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher, sha256_file

_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_ARCHETYPES = {"biped", "quadruped", "flying", "serpentine", "construct", "custom"}
_BEHAVIORS = {"hostile_melee", "neutral_melee", "passive", "npc"}
_SPAWN = {
    "monster": "MONSTER",
    "creature": "CREATURE",
    "ambient": "AMBIENT",
    "water_creature": "WATER_CREATURE",
    "misc": "MISC",
}


class GeckoLibGenerationError(ValueError):
    pass


def generate_geckolib_entity_assets(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    entity_id: str,
    texture_width: int = 64,
    texture_height: int = 64,
    geckolib_version: str = "4.8.2",
    max_health: float = 80.0,
    attack_damage: float = 8.0,
    movement_speed: float = 0.27,
    archetype: str = "biped",
    behavior: str = "hostile_melee",
    entity_width: float = 0.8,
    entity_height: float = 2.0,
    follow_range: float = 40.0,
    spawn_group: str | None = None,
    custom_bones: list[dict[str, Any]] | None = None,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    if not _ID.fullmatch(mod_id) or not _ID.fullmatch(entity_id):
        raise GeckoLibGenerationError("Invalid mod or entity id.")
    if not _PACKAGE.fullmatch(package_name):
        raise GeckoLibGenerationError("Invalid Java package.")
    if (
        type(texture_width) is not int
        or type(texture_height) is not int
        or not (1 <= texture_width <= policy.max_texture_dimension)
        or not (1 <= texture_height <= policy.max_texture_dimension)
    ):
        raise GeckoLibGenerationError(
            "Texture dimensions exceed configured resource policy."
        )
    for name, value in {
        "max_health": max_health,
        "attack_damage": attack_damage,
        "movement_speed": movement_speed,
        "entity_width": entity_width,
        "entity_height": entity_height,
        "follow_range": follow_range,
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise GeckoLibGenerationError(f"{name} must be a positive finite number.")
    if archetype not in _ARCHETYPES or (archetype == "custom" and not custom_bones):
        raise GeckoLibGenerationError("Unknown or incomplete entity archetype.")
    if behavior not in _BEHAVIORS:
        raise GeckoLibGenerationError("Unknown entity behavior profile.")
    spawn_group = spawn_group or (
        "monster" if behavior == "hostile_melee" else "creature"
    )
    if spawn_group not in _SPAWN:
        raise GeckoLibGenerationError("Unknown spawn group.")

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise GeckoLibGenerationError("GeckoLib target does not match fabric.mod.json.")
    cls = "".join(part.capitalize() for part in entity_id.split("_"))
    entity_cls = cls + "Entity"
    package_path = package_name.replace(".", "/")
    base = f"src/main/java/{package_path}"
    assets = f"src/main/resources/assets/{mod_id}"
    manifest_path = info.root / ".minecraft_ai/geckolib-entities.json"
    legacy_entries: list[dict[str, Any]] = []
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("entities"), list):
            legacy_entries = [_legacy_entity_record(value) for value in raw["entities"]]
    entry = {
        "entity_id": entity_id,
        "class_name": cls,
        "entity_class": entity_cls,
        "max_health": float(max_health),
        "attack_damage": float(attack_damage),
        "movement_speed": float(movement_speed),
        "follow_range": float(follow_range),
        "entity_width": float(entity_width),
        "entity_height": float(entity_height),
        "archetype": archetype,
        "behavior": behavior,
        "spawn_group": spawn_group,
    }
    bones = custom_bones if custom_bones is not None else _bones(archetype)
    files = {
        f"{assets}/geo/{entity_id}.geo.json": json.dumps(
            _geometry(mod_id, entity_id, texture_width, texture_height, bones),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        f"{assets}/animations/{entity_id}.animation.json": json.dumps(
            _animations(mod_id, entity_id, archetype), indent=2, sort_keys=True
        )
        + "\n",
        f"{base}/entity/{entity_cls}.java": _entity_java(
            package_name,
            mod_id,
            entity_id,
            entity_cls,
            float(max_health),
            float(attack_damage),
            float(movement_speed),
            float(follow_range),
            behavior,
        ),
        f"{base}/client/geckolib/{cls}GeoModel.java": _model_java(
            package_name, mod_id, entity_id, cls, entity_cls
        ),
        f"{base}/client/geckolib/{cls}GeoRenderer.java": _renderer_java(
            package_name, cls, entity_cls, float(entity_width)
        ),
        ".minecraft_ai/geckolib-entities.json": _entity_index_text(mod_id),
        f"{base}/geckolib/GeneratedGeckoEntities.java": _root_reg(
            package_name,
            mod_id,
        ),
        f"{base}/client/geckolib/GeneratedGeckoClient.java": _root_client(
            package_name,
        ),
    }
    for legacy in legacy_entries:
        files.update(
            _registration_unit_files(
                package_name,
                mod_id,
                base,
                legacy,
            )
        )
    files.update(
        _registration_unit_files(
            package_name,
            mod_id,
            base,
            entry,
        )
    )
    write_receipt = write_text_files(info, files, replace_existing=True)
    texture = info.root / f"{assets}/textures/entity/{entity_id}.png"
    texture.parent.mkdir(parents=True, exist_ok=True)
    if not texture.exists():
        texture.write_bytes(
            make_texture_png(
                "#5ba6d8",
                entity_id,
                kind="entity",
                size=max(texture_width, texture_height),
            )
        )
    dependency = ensure_dependency(
        info,
        repository_block="""maven {\n    name = 'GeckoLib'\n    url = 'https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/'\n}""",
        dependency_line=f'modImplementation("software.bernie.geckolib:geckolib-fabric-1.20.1:{geckolib_version}")',
        marker="geckolib",
    )
    main = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.geckolib.GeneratedGeckoEntities",
        call_line="GeneratedGeckoEntities.register()",
        marker="geckolib:entities",
    )
    client = ensure_client_entrypoint(
        info, entrypoint=f"{package_name}.client.geckolib.GeneratedGeckoClient"
    )
    metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
    depends = metadata.setdefault("depends", {})
    if not isinstance(depends, dict):
        raise GeckoLibGenerationError("fabric.mod.json depends must be an object.")
    if depends.get("geckolib") != f">={geckolib_version}":
        depends["geckolib"] = f">={geckolib_version}"
        meta_receipt: dict[str, Any] = TransactionalSourcePatcher(info.root).apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/resources/fabric.mod.json",
                    "expected_sha256": sha256_file(info.fabric_mod_json),
                    "content": json.dumps(metadata, indent=2) + "\n",
                }
            ]
        )
    else:
        meta_receipt = {"status": "UNCHANGED"}
    record_root = info.root / ".minecraft_ai/geckolib-entities"
    entity_count = sum(
        1
        for path in record_root.glob("*.json")
        if path.is_file() and not path.is_symlink()
    )
    return {
        "schema_version": "mmm/geckolib-generation-v3",
        "entity_id": entity_id,
        "archetype": archetype,
        "behavior": behavior,
        "entity_count": entity_count,
        "registrar_shards": entity_count,
        "files": [str(info.root / p) for p in files] + [str(texture)],
        "status": "fabric_binding_generated",
        "receipts": {
            "files": write_receipt,
            "dependency": dependency,
            "main": main,
            "client": client,
            "metadata": meta_receipt,
        },
        "required_gates": [
            "Blockbench UV and bone hierarchy review",
            "Gradle clean build",
            "GameTest spawn and attributes",
            "runtime animation review",
        ],
    }


def _bones(kind: str) -> list[dict[str, Any]]:
    root = [{"name": "root", "pivot": [0, 0, 0]}]
    if kind == "quadruped":
        return (
            root
            + [
                {
                    "name": "body",
                    "parent": "root",
                    "pivot": [0, 10, 0],
                    "cubes": [
                        {"origin": [-5, 6, -8], "size": [10, 8, 16], "uv": [0, 0]}
                    ],
                },
                {
                    "name": "head",
                    "parent": "body",
                    "pivot": [0, 12, -8],
                    "cubes": [
                        {"origin": [-4, 8, -14], "size": [8, 8, 8], "uv": [0, 24]}
                    ],
                },
            ]
            + [
                {
                    "name": name,
                    "parent": "root",
                    "pivot": [x, 7, z],
                    "cubes": [
                        {
                            "origin": [x - 2, -1, z - 2],
                            "size": [4, 8, 4],
                            "uv": [32, 24],
                        }
                    ],
                }
                for name, x, z in (
                    ("front_left_leg", 4, -5),
                    ("front_right_leg", -4, -5),
                    ("back_left_leg", 4, 5),
                    ("back_right_leg", -4, 5),
                )
            ]
        )
    if kind == "flying":
        return root + [
            {
                "name": "body",
                "parent": "root",
                "pivot": [0, 12, 0],
                "cubes": [{"origin": [-4, 8, -4], "size": [8, 8, 8], "uv": [0, 0]}],
            },
            {
                "name": "left_wing",
                "parent": "body",
                "pivot": [4, 13, 0],
                "cubes": [{"origin": [4, 12, -2], "size": [12, 1, 8], "uv": [0, 20]}],
            },
            {
                "name": "right_wing",
                "parent": "body",
                "pivot": [-4, 13, 0],
                "cubes": [{"origin": [-16, 12, -2], "size": [12, 1, 8], "uv": [0, 20]}],
            },
        ]
    if kind == "serpentine":
        bones = root
        parent = "root"
        for i in range(8):
            name = f"segment_{i}"
            bones.append(
                {
                    "name": name,
                    "parent": parent,
                    "pivot": [0, 3, i * 4],
                    "cubes": [
                        {"origin": [-3, 0, i * 4], "size": [6, 6, 5], "uv": [i * 6, 0]}
                    ],
                }
            )
            parent = name
        return bones
    if kind == "construct":
        return root + [
            {
                "name": "core",
                "parent": "root",
                "pivot": [0, 12, 0],
                "cubes": [{"origin": [-6, 6, -6], "size": [12, 12, 12], "uv": [0, 0]}],
            },
            {
                "name": "left_arm",
                "parent": "core",
                "pivot": [7, 15, 0],
                "cubes": [{"origin": [6, 5, -3], "size": [5, 12, 6], "uv": [32, 24]}],
            },
            {
                "name": "right_arm",
                "parent": "core",
                "pivot": [-7, 15, 0],
                "cubes": [{"origin": [-11, 5, -3], "size": [5, 12, 6], "uv": [32, 24]}],
            },
        ]
    return (
        root
        + [
            {
                "name": "body",
                "parent": "root",
                "pivot": [0, 12, 0],
                "cubes": [{"origin": [-4, 6, -2], "size": [8, 12, 4], "uv": [0, 16]}],
            },
            {
                "name": "head",
                "parent": "body",
                "pivot": [0, 18, 0],
                "cubes": [{"origin": [-4, 18, -4], "size": [8, 8, 8], "uv": [0, 0]}],
            },
        ]
        + [
            {
                "name": name,
                "parent": parent,
                "pivot": pivot,
                "cubes": [{"origin": origin, "size": [4, 12, 4], "uv": uv}],
            }
            for name, parent, pivot, origin, uv in (
                ("right_arm", "body", [-5, 16, 0], [-8, 6, -2], [24, 16]),
                ("left_arm", "body", [5, 16, 0], [4, 6, -2], [40, 16]),
                ("right_leg", "root", [-2, 6, 0], [-4, -6, -2], [0, 32]),
                ("left_leg", "root", [2, 6, 0], [0, -6, -2], [16, 32]),
            )
        ]
    )


def _geometry(
    mod_id: str, entity_id: str, width: int, height: int, bones: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "format_version": "1.12.0",
        "minecraft:geometry": [
            {
                "description": {
                    "identifier": f"geometry.{mod_id}.{entity_id}",
                    "texture_width": width,
                    "texture_height": height,
                    "visible_bounds_width": 8.0,
                    "visible_bounds_height": 8.0,
                    "visible_bounds_offset": [0, 2, 0],
                },
                "bones": bones,
            }
        ],
    }


def _animations(mod_id: str, entity_id: str, archetype: str) -> dict[str, Any]:
    prefix = f"animation.{mod_id}.{entity_id}"
    bones = _bones(archetype)
    moving = {}
    for bone in bones:
        name = bone["name"]
        if any(token in name for token in ("leg", "wing", "segment", "arm")):
            phase = 20 if sum(name.encode("utf-8")) % 2 else -20
            moving[name] = {
                "rotation": {
                    "0.0": [phase, 0, 0],
                    "0.5": [-phase, 0, 0],
                    "1.0": [phase, 0, 0],
                }
            }
    attack = next(
        (
            b["name"]
            for b in bones
            if "arm" in b["name"] or b["name"] in {"head", "core", "segment_0"}
        ),
        "root",
    )
    return {
        "format_version": "1.8.0",
        "animations": {
            f"{prefix}.idle": {
                "loop": True,
                "animation_length": 2.0,
                "bones": {
                    "root": {
                        "rotation": {
                            "0.0": [0, 0, 0],
                            "1.0": [1.5, 0, 0],
                            "2.0": [0, 0, 0],
                        }
                    }
                },
            },
            f"{prefix}.walk": {"loop": True, "animation_length": 1.0, "bones": moving},
            f"{prefix}.attack": {
                "loop": False,
                "animation_length": 0.5,
                "bones": {
                    attack: {
                        "rotation": {
                            "0.0": [0, 0, 0],
                            "0.2": [-75, 0, 0],
                            "0.5": [0, 0, 0],
                        }
                    }
                },
            },
        },
    }


def _entity_java(
    package: str,
    mod_id: str,
    entity_id: str,
    cls: str,
    health: float,
    damage: float,
    speed: float,
    follow: float,
    behavior: str,
) -> str:
    melee = (
        "        this.goalSelector.add(2, new MeleeAttackGoal(this, 1.05, false));\n"
        if behavior in {"hostile_melee", "neutral_melee"}
        else ""
    )
    target = (
        "        this.targetSelector.add(2, new ActiveTargetGoal<>(this, PlayerEntity.class, true));\n"
        if behavior == "hostile_melee"
        else ""
    )
    interaction = (
        """\n    @Override protected ActionResult interactMob(PlayerEntity player, Hand hand) {\n        if (!getWorld().isClient) player.sendMessage(Text.literal("NPC: %s"), false);\n        return ActionResult.success(getWorld().isClient);\n    }\n"""
        % entity_id
        if behavior == "npc"
        else ""
    )
    extra = (
        "import net.minecraft.util.ActionResult;\nimport net.minecraft.util.Hand;\n"
        if behavior == "npc"
        else ""
    )
    return f"""package {package}.entity;\n\nimport net.minecraft.entity.EntityType;\nimport net.minecraft.entity.ai.goal.*;\nimport net.minecraft.entity.attribute.*;\nimport net.minecraft.entity.mob.MobEntity;\nimport net.minecraft.entity.mob.PathAwareEntity;\nimport net.minecraft.entity.player.PlayerEntity;\nimport net.minecraft.text.Text;\nimport net.minecraft.world.World;\n{extra}import software.bernie.geckolib.animatable.GeoEntity;\nimport software.bernie.geckolib.core.animatable.instance.AnimatableInstanceCache;\nimport software.bernie.geckolib.core.animation.*;\nimport software.bernie.geckolib.core.object.PlayState;\nimport software.bernie.geckolib.util.GeckoLibUtil;\n\npublic final class {cls} extends PathAwareEntity implements GeoEntity {{\n    private static final RawAnimation IDLE=RawAnimation.begin().thenLoop("animation.{mod_id}.{entity_id}.idle");\n    private static final RawAnimation WALK=RawAnimation.begin().thenLoop("animation.{mod_id}.{entity_id}.walk");\n    private static final RawAnimation ATTACK=RawAnimation.begin().thenPlay("animation.{mod_id}.{entity_id}.attack");\n    private final AnimatableInstanceCache cache=GeckoLibUtil.createInstanceCache(this);\n    public {cls}(EntityType<? extends PathAwareEntity> type, World world) {{ super(type,world); this.experiencePoints=30; }}\n    public static DefaultAttributeContainer.Builder createAttributes() {{ return MobEntity.createMobAttributes().add(EntityAttributes.GENERIC_MAX_HEALTH,{health:.3f}).add(EntityAttributes.GENERIC_ATTACK_DAMAGE,{damage:.3f}).add(EntityAttributes.GENERIC_MOVEMENT_SPEED,{speed:.5f}).add(EntityAttributes.GENERIC_FOLLOW_RANGE,{follow:.3f}); }}\n    @Override protected void initGoals() {{ this.goalSelector.add(1,new SwimGoal(this));\n{melee}        this.goalSelector.add(7,new WanderAroundFarGoal(this,0.85)); this.goalSelector.add(8,new LookAroundGoal(this));\n{target}    }}\n{interaction}    @Override public void registerControllers(AnimatableManager.ControllerRegistrar controllers) {{ controllers.add(new AnimationController<>(this,"movement",4,state->state.setAndContinue(state.isMoving()?WALK:IDLE))); controllers.add(new AnimationController<>(this,"attack",0,state->this.handSwinging?state.setAndContinue(ATTACK):PlayState.STOP)); }}\n    @Override public AnimatableInstanceCache getAnimatableInstanceCache() {{ return cache; }}\n}}\n"""


def _model_java(package: str, mod_id: str, entity_id: str, name: str, cls: str) -> str:
    return f'''package {package}.client.geckolib;\nimport {package}.entity.{cls};\nimport net.minecraft.util.Identifier;\nimport software.bernie.geckolib.model.GeoModel;\npublic final class {name}GeoModel extends GeoModel<{cls}> {{\n @Override public Identifier getModelResource({cls} e){{return new Identifier("{mod_id}","geo/{entity_id}.geo.json");}}\n @Override public Identifier getTextureResource({cls} e){{return new Identifier("{mod_id}","textures/entity/{entity_id}.png");}}\n @Override public Identifier getAnimationResource({cls} e){{return new Identifier("{mod_id}","animations/{entity_id}.animation.json");}}\n}}\n'''


def _renderer_java(package: str, name: str, cls: str, width: float) -> str:
    return f"""package {package}.client.geckolib;\nimport {package}.entity.{cls};\nimport net.minecraft.client.render.entity.EntityRendererFactory;\nimport software.bernie.geckolib.renderer.GeoEntityRenderer;\npublic final class {name}GeoRenderer extends GeoEntityRenderer<{cls}> {{ public {name}GeoRenderer(EntityRendererFactory.Context c){{super(c,new {name}GeoModel());this.shadowRadius={max(0.1, min(4.0, width * 0.55)):.3f}f;}} }}\n"""


def iter_geckolib_entity_records(
    project_root: str | Path,
):
    """Yield legacy or per-entity metadata without an aggregate manifest."""

    root = Path(project_root).expanduser().resolve()
    manifest = root / ".minecraft_ai/geckolib-entities.json"
    if not manifest.is_file() or manifest.is_symlink():
        return
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise GeckoLibGenerationError("GeckoLib entity index must be an object.")
    legacy = raw.get("entities")
    if isinstance(legacy, list):
        for value in legacy:
            yield _legacy_entity_record(value)
        return
    if raw.get("schema_version") != "mmm/geckolib-entity-index-v1":
        raise GeckoLibGenerationError("Unsupported GeckoLib entity index.")
    relative = raw.get("record_directory")
    if relative != ".minecraft_ai/geckolib-entities":
        raise GeckoLibGenerationError("GeckoLib entity record directory is invalid.")
    records = root / relative
    if not records.exists():
        return
    if not records.is_dir() or records.is_symlink():
        raise GeckoLibGenerationError("GeckoLib entity record directory is unsafe.")
    for path in sorted(records.glob("*.json")):
        if not path.is_file() or path.is_symlink():
            raise GeckoLibGenerationError("GeckoLib entity record is unsafe.")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(loaded, dict)
            and loaded.get("schema_version") == "mmm/geckolib-entity-record-v1"
        ):
            value = _legacy_entity_record(loaded.get("entity"))
        else:
            value = _legacy_entity_record(loaded)
        if path.stem != value["entity_id"]:
            raise GeckoLibGenerationError(
                "GeckoLib entity record name does not match its entity ID."
            )
        yield value


def _legacy_entity_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GeckoLibGenerationError("GeckoLib entity record must be an object.")
    required = {
        "entity_id",
        "class_name",
        "entity_class",
        "max_health",
        "attack_damage",
        "movement_speed",
        "follow_range",
        "entity_width",
        "entity_height",
        "archetype",
        "behavior",
        "spawn_group",
    }
    if set(value) != required:
        raise GeckoLibGenerationError("GeckoLib entity record fields are invalid.")
    entity_id = str(value["entity_id"])
    if not _ID.fullmatch(entity_id):
        raise GeckoLibGenerationError("GeckoLib entity record ID is invalid.")
    return dict(value)


def _registration_unit_files(
    package: str,
    mod_id: str,
    base: str,
    entry: dict[str, Any],
) -> dict[str, str]:
    entity_id = str(entry["entity_id"])
    class_name = str(entry["class_name"])
    server_name = f"{class_name}GeckoRegistration"
    client_name = f"{class_name}GeckoClientRegistration"
    descriptor = {
        "schema_version": "mmm/geckolib-registration-unit-v1",
        "entity_id": entity_id,
        "server_registrar": f"{package}.geckolib.entry.{server_name}",
        "client_registrar": (f"{package}.client.geckolib.entry.{client_name}"),
    }
    return {
        f".minecraft_ai/geckolib-entities/{entity_id}.json": _json_text(
            {
                "schema_version": "mmm/geckolib-entity-record-v1",
                "entity": entry,
            }
        ),
        (
            f"src/main/resources/data/{mod_id}/mmm_geckolib/entities/{entity_id}.json"
        ): _json_text(descriptor),
        f"{base}/geckolib/entry/{server_name}.java": _server_unit_java(
            package,
            mod_id,
            server_name,
            entry,
        ),
        (f"{base}/client/geckolib/entry/{client_name}.java"): _client_unit_java(
            package,
            client_name,
            server_name,
            entry,
        ),
    }


def _server_unit_java(
    package: str,
    mod_id: str,
    name: str,
    entry: dict[str, Any],
) -> str:
    entity_class = str(entry["entity_class"])
    constant = str(entry["entity_id"]).upper()
    return f'''package {package}.geckolib.entry;

import {package}.entity.{entity_class};
import {package}.geckolib.GeneratedGeckoEntities;
import net.fabricmc.fabric.api.object.builder.v1.entity.FabricDefaultAttributeRegistry;
import net.fabricmc.fabric.api.object.builder.v1.entity.FabricEntityTypeBuilder;
import net.minecraft.entity.EntityDimensions;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.SpawnGroup;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;

public final class {name} implements GeneratedGeckoEntities.Registration {{
    public static EntityType<{entity_class}> {constant};
    private static boolean registered;

    @Override
    public synchronized void register() {{
        if (registered) return;
        registered = true;
        {constant} = Registry.register(
            Registries.ENTITY_TYPE,
            new Identifier("{mod_id}", "{entry["entity_id"]}"),
            FabricEntityTypeBuilder.create(
                SpawnGroup.{_SPAWN[str(entry["spawn_group"])]},
                {entity_class}::new
            ).dimensions(
                EntityDimensions.fixed(
                    {float(entry["entity_width"]):.4f}f,
                    {float(entry["entity_height"]):.4f}f
                )
            ).trackRangeBlocks(10).build()
        );
        FabricDefaultAttributeRegistry.register(
            {constant},
            {entity_class}.createAttributes()
        );
    }}
}}
'''


def _client_unit_java(
    package: str,
    name: str,
    server_name: str,
    entry: dict[str, Any],
) -> str:
    return f"""package {package}.client.geckolib.entry;

import {package}.client.geckolib.GeneratedGeckoClient;
import {package}.client.geckolib.{entry["class_name"]}GeoRenderer;
import {package}.geckolib.entry.{server_name};
import net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;

public final class {name} implements GeneratedGeckoClient.Registration {{
    @Override
    public void register() {{
        EntityRendererRegistry.register(
            {server_name}.{str(entry["entity_id"]).upper()},
            {entry["class_name"]}GeoRenderer::new
        );
    }}
}}
"""


def _root_reg(package: str, mod_id: str) -> str:
    return f'''package {package}.geckolib;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import net.fabricmc.loader.api.FabricLoader;
import net.fabricmc.loader.api.ModContainer;
import software.bernie.geckolib.GeckoLib;

import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;
import java.util.stream.Stream;

public final class GeneratedGeckoEntities {{
    private static final Gson GSON = new Gson();
    private static final String SERVER_PREFIX =
        "{package}.geckolib.entry.";
    private static boolean registered;

    public interface Registration {{
        void register();
    }}

    private GeneratedGeckoEntities() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        GeckoLib.initialize();
        forEachDescriptor(GeneratedGeckoEntities::registerDescriptor);
    }}

    public static void forEachDescriptor(Consumer<JsonObject> visitor) {{
        ModContainer container = FabricLoader.getInstance()
            .getModContainer("{mod_id}")
            .orElseThrow(() -> new IllegalStateException(
                "Missing Fabric mod container: {mod_id}"
            ));
        Path directory = container.findPath(
            "data/{mod_id}/mmm_geckolib/entities"
        ).orElseThrow(() -> new IllegalStateException(
            "Missing GeckoLib entity descriptor directory"
        ));
        try (Stream<Path> paths = Files.list(directory)) {{
            paths.filter(path ->
                    path.getFileName().toString().endsWith(".json")
                )
                .sorted()
                .forEach(path -> readDescriptor(path, visitor));
        }} catch (Exception exception) {{
            throw new IllegalStateException(
                "Could not scan GeckoLib entity descriptors",
                exception
            );
        }}
    }}

    private static void readDescriptor(
        Path path,
        Consumer<JsonObject> visitor
    ) {{
        try (Reader reader = Files.newBufferedReader(
            path,
            StandardCharsets.UTF_8
        )) {{
            JsonObject descriptor = GSON.fromJson(reader, JsonObject.class);
            if (!"mmm/geckolib-registration-unit-v1".equals(
                descriptor.get("schema_version").getAsString()
            )) {{
                throw new IllegalStateException(
                    "Unsupported GeckoLib descriptor: " + path
                );
            }}
            visitor.accept(descriptor);
        }} catch (Exception exception) {{
            throw new IllegalStateException(
                "Could not load GeckoLib descriptor: " + path,
                exception
            );
        }}
    }}

    private static void registerDescriptor(JsonObject descriptor) {{
        String className = descriptor.get("server_registrar").getAsString();
        if (!className.startsWith(SERVER_PREFIX)) {{
            throw new IllegalStateException(
                "Untrusted GeckoLib server registrar: " + className
            );
        }}
        try {{
            Object value = Class.forName(className)
                .getDeclaredConstructor()
                .newInstance();
            if (!(value instanceof Registration registration)) {{
                throw new IllegalStateException(
                    "Invalid GeckoLib server registrar: " + className
                );
            }}
            registration.register();
        }} catch (ReflectiveOperationException exception) {{
            throw new IllegalStateException(
                "Could not initialize GeckoLib server registrar: "
                    + className,
                exception
            );
        }}
    }}
}}
'''


def _root_client(package: str) -> str:
    return f'''package {package}.client.geckolib;

import com.google.gson.JsonObject;
import {package}.geckolib.GeneratedGeckoEntities;
import net.fabricmc.api.ClientModInitializer;

public final class GeneratedGeckoClient implements ClientModInitializer {{
    private static final String CLIENT_PREFIX =
        "{package}.client.geckolib.entry.";

    public interface Registration {{
        void register();
    }}

    @Override
    public void onInitializeClient() {{
        GeneratedGeckoEntities.forEachDescriptor(
            GeneratedGeckoClient::registerDescriptor
        );
    }}

    private static void registerDescriptor(JsonObject descriptor) {{
        String className = descriptor.get("client_registrar").getAsString();
        if (!className.startsWith(CLIENT_PREFIX)) {{
            throw new IllegalStateException(
                "Untrusted GeckoLib client registrar: " + className
            );
        }}
        try {{
            Object value = Class.forName(className)
                .getDeclaredConstructor()
                .newInstance();
            if (!(value instanceof Registration registration)) {{
                throw new IllegalStateException(
                    "Invalid GeckoLib client registrar: " + className
                );
            }}
            registration.register();
        }} catch (ReflectiveOperationException exception) {{
            throw new IllegalStateException(
                "Could not initialize GeckoLib client registrar: "
                    + className,
                exception
            );
        }}
    }}
}}
'''


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _entity_index_text(mod_id: str) -> str:
    return _json_text(
        {
            "schema_version": "mmm/geckolib-entity-index-v1",
            "record_directory": ".minecraft_ai/geckolib-entities",
            "runtime_record_directory": (f"data/{mod_id}/mmm_geckolib/entities"),
        }
    )
