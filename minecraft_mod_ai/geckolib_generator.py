from __future__ import annotations

import json
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
from .source_patch import TransactionalSourcePatcher, sha256_file


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


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
) -> dict[str, Any]:
    """Generate and bind a complete GeckoLib entity implementation.

    The generator writes the entity, renderer, model, registrar, client entrypoint,
    geometry, three animation states, texture, dependency and fabric.mod.json binding.
    Build, GameTest, Blockbench UV review and runtime animation evidence remain gates.
    """

    if not _ID.fullmatch(mod_id) or not _ID.fullmatch(entity_id):
        raise GeckoLibGenerationError("Invalid mod or entity id.")
    if not _PACKAGE.fullmatch(package_name):
        raise GeckoLibGenerationError("Invalid Java package.")
    if texture_width not in {16, 32, 64, 128, 256} or texture_height not in {
        16,
        32,
        64,
        128,
        256,
    }:
        raise GeckoLibGenerationError("Unsupported GeckoLib texture size.")
    if not 1.0 <= float(max_health) <= 2048.0:
        raise GeckoLibGenerationError("max_health is outside the reviewed range.")
    if not 0.1 <= float(attack_damage) <= 100.0:
        raise GeckoLibGenerationError("attack_damage is outside the reviewed range.")
    if not 0.05 <= float(movement_speed) <= 1.0:
        raise GeckoLibGenerationError("movement_speed is outside the reviewed range.")

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise GeckoLibGenerationError("GeckoLib target does not match fabric.mod.json.")
    class_name = "".join(part.capitalize() for part in entity_id.split("_"))
    entity_class = f"{class_name}Entity"
    package_path = package_name.replace(".", "/")
    base_java = f"src/main/java/{package_path}"
    assets = f"src/main/resources/assets/{mod_id}"
    manifest_path = info.root / ".minecraft_ai/geckolib-entities.json"
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {str(item["entity_id"]): dict(item) for item in raw_manifest.get("entities", [])}
    entries[entity_id] = {
        "entity_id": entity_id,
        "class_name": class_name,
        "entity_class": entity_class,
        "max_health": float(max_health),
        "attack_damage": float(attack_damage),
        "movement_speed": float(movement_speed),
    }
    ordered_entries = [entries[key] for key in sorted(entries)]

    files = {
        f"{assets}/geo/{entity_id}.geo.json": json.dumps(
            _geometry(mod_id, entity_id, texture_width, texture_height),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        f"{assets}/animations/{entity_id}.animation.json": json.dumps(
            _animations(mod_id, entity_id), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        f"{base_java}/entity/{entity_class}.java": _entity_java(
            package_name,
            mod_id,
            entity_id,
            entity_class,
            max_health=max_health,
            attack_damage=attack_damage,
            movement_speed=movement_speed,
        ),
        f"{base_java}/client/geckolib/{class_name}GeoModel.java": _model_java(
            package_name, mod_id, entity_id, class_name, entity_class
        ),
        f"{base_java}/client/geckolib/{class_name}GeoRenderer.java": _renderer_java(
            package_name, class_name, entity_class
        ),
        f"{base_java}/geckolib/GeneratedGeckoEntities.java": _registrar_java_all(
            package_name, mod_id, ordered_entries
        ),
        f"{base_java}/client/geckolib/GeneratedGeckoClient.java": _client_java_all(
            package_name, ordered_entries
        ),
        ".minecraft_ai/geckolib-entities.json": json.dumps(
            {"schema_version": "mmm/geckolib-entities-v1", "entities": ordered_entries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
    }
    write_receipt = write_text_files(info, files, replace_existing=True)
    texture = info.root / f"{assets}/textures/entity/{entity_id}.png"
    texture.parent.mkdir(parents=True, exist_ok=True)
    if not texture.exists():
        texture.write_bytes(make_texture_png("#5ba6d8", entity_id, kind="entity", size=max(texture_width, texture_height)))

    dependency_receipt = ensure_dependency(
        info,
        repository_block='''maven {
    name = 'GeckoLib'
    url = 'https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/'
    content {
        includeGroupByRegex("software\\.bernie.*")
        includeGroup("com.eliotlash.mclib")
    }
}''',
        dependency_line=f'modImplementation("software.bernie.geckolib:geckolib-fabric-1.20.1:{geckolib_version}")',
        marker="geckolib",
    )
    main_binding = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.geckolib.GeneratedGeckoEntities",
        call_line="GeneratedGeckoEntities.register()",
        marker="geckolib:entities",
    )
    client_entrypoint = f"{package_name}.client.geckolib.GeneratedGeckoClient"
    client_binding = ensure_client_entrypoint(info, entrypoint=client_entrypoint)
    metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8"))
    depends = metadata.setdefault("depends", {})
    if not isinstance(depends, dict):
        raise GeckoLibGenerationError("fabric.mod.json depends must be an object.")
    metadata_changed = False
    if depends.get("geckolib") != f">={geckolib_version}":
        depends["geckolib"] = f">={geckolib_version}"
        metadata_changed = True
    metadata_receipt: dict[str, Any]
    if metadata_changed:
        metadata_receipt = TransactionalSourcePatcher(info.root).apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/resources/fabric.mod.json",
                    "expected_sha256": sha256_file(info.fabric_mod_json),
                    "content": json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                }
            ]
        )
    else:
        metadata_receipt = {"status": "UNCHANGED"}

    return {
        "schema_version": "mmm/geckolib-generation-v2",
        "entity_id": entity_id,
        "geckolib_version": geckolib_version,
        "files": [str(info.root / path) for path in files] + [str(texture)],
        "status": "fabric_binding_generated",
        "receipts": {
            "files": write_receipt,
            "dependency": dependency_receipt,
            "main": main_binding,
            "client": client_binding,
            "metadata": metadata_receipt,
        },
        "required_gates": [
            "Blockbench UV and bone hierarchy review",
            "Gradle clean build",
            "GameTest spawn and attributes",
            "runtime idle/walk/attack animation screenshot review",
        ],
    }


def _geometry(mod_id: str, entity_id: str, width: int, height: int) -> dict[str, Any]:
    bones = [
        {"name": "root", "pivot": [0, 0, 0]},
        {"name": "body", "parent": "root", "pivot": [0, 12, 0], "cubes": [{"origin": [-4, 6, -2], "size": [8, 12, 4], "uv": [0, 16]}]},
        {"name": "head", "parent": "body", "pivot": [0, 18, 0], "cubes": [{"origin": [-4, 18, -4], "size": [8, 8, 8], "uv": [0, 0]}]},
        {"name": "right_arm", "parent": "body", "pivot": [-5, 16, 0], "cubes": [{"origin": [-8, 6, -2], "size": [4, 12, 4], "uv": [24, 16]}]},
        {"name": "left_arm", "parent": "body", "pivot": [5, 16, 0], "cubes": [{"origin": [4, 6, -2], "size": [4, 12, 4], "uv": [40, 16]}]},
        {"name": "right_leg", "parent": "root", "pivot": [-2, 6, 0], "cubes": [{"origin": [-4, -6, -2], "size": [4, 12, 4], "uv": [0, 32]}]},
        {"name": "left_leg", "parent": "root", "pivot": [2, 6, 0], "cubes": [{"origin": [0, -6, -2], "size": [4, 12, 4], "uv": [16, 32]}]},
    ]
    return {
        "format_version": "1.12.0",
        "minecraft:geometry": [
            {
                "description": {
                    "identifier": f"geometry.{mod_id}.{entity_id}",
                    "texture_width": width,
                    "texture_height": height,
                    "visible_bounds_width": 2.5,
                    "visible_bounds_height": 3.5,
                    "visible_bounds_offset": [0, 1.25, 0],
                },
                "bones": bones,
            }
        ],
    }


def _animations(mod_id: str, entity_id: str) -> dict[str, Any]:
    prefix = f"animation.{mod_id}.{entity_id}"
    return {
        "format_version": "1.8.0",
        "animations": {
            f"{prefix}.idle": {
                "loop": True,
                "animation_length": 2.0,
                "bones": {"body": {"rotation": {"0.0": [0, 0, 0], "1.0": [1.5, 0, 0], "2.0": [0, 0, 0]}}},
            },
            f"{prefix}.walk": {
                "loop": True,
                "animation_length": 1.0,
                "bones": {
                    "right_leg": {"rotation": {"0.0": [30, 0, 0], "0.5": [-30, 0, 0], "1.0": [30, 0, 0]}},
                    "left_leg": {"rotation": {"0.0": [-30, 0, 0], "0.5": [30, 0, 0], "1.0": [-30, 0, 0]}},
                    "right_arm": {"rotation": {"0.0": [-20, 0, 0], "0.5": [20, 0, 0], "1.0": [-20, 0, 0]}},
                    "left_arm": {"rotation": {"0.0": [20, 0, 0], "0.5": [-20, 0, 0], "1.0": [20, 0, 0]}},
                },
            },
            f"{prefix}.attack": {
                "loop": False,
                "animation_length": 0.5,
                "bones": {"right_arm": {"rotation": {"0.0": [0, 0, 0], "0.2": [-100, 0, 0], "0.5": [0, 0, 0]}}},
            },
        },
    }


def _entity_java(
    package_name: str,
    mod_id: str,
    entity_id: str,
    entity_class: str,
    *,
    max_health: float,
    attack_damage: float,
    movement_speed: float,
) -> str:
    return f'''package {package_name}.entity;

import net.minecraft.entity.EntityType;
import net.minecraft.entity.ai.goal.ActiveTargetGoal;
import net.minecraft.entity.ai.goal.LookAroundGoal;
import net.minecraft.entity.ai.goal.MeleeAttackGoal;
import net.minecraft.entity.ai.goal.SwimGoal;
import net.minecraft.entity.ai.goal.WanderAroundFarGoal;
import net.minecraft.entity.attribute.DefaultAttributeContainer;
import net.minecraft.entity.attribute.EntityAttributes;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.world.World;
import software.bernie.geckolib.animatable.GeoEntity;
import software.bernie.geckolib.core.animatable.instance.AnimatableInstanceCache;
import software.bernie.geckolib.core.animation.AnimatableManager;
import software.bernie.geckolib.core.animation.AnimationController;
import software.bernie.geckolib.core.animation.RawAnimation;
import software.bernie.geckolib.util.GeckoLibUtil;

public final class {entity_class} extends HostileEntity implements GeoEntity {{
    private static final RawAnimation IDLE = RawAnimation.begin().thenLoop("animation.{mod_id}.{entity_id}.idle");
    private static final RawAnimation WALK = RawAnimation.begin().thenLoop("animation.{mod_id}.{entity_id}.walk");
    private static final RawAnimation ATTACK = RawAnimation.begin().thenPlay("animation.{mod_id}.{entity_id}.attack");
    private final AnimatableInstanceCache cache = GeckoLibUtil.createInstanceCache(this);

    public {entity_class}(EntityType<? extends HostileEntity> type, World world) {{
        super(type, world);
        this.experiencePoints = 30;
    }}

    public static DefaultAttributeContainer.Builder createAttributes() {{
        return HostileEntity.createHostileAttributes()
            .add(EntityAttributes.GENERIC_MAX_HEALTH, {float(max_health):.2f})
            .add(EntityAttributes.GENERIC_ATTACK_DAMAGE, {float(attack_damage):.2f})
            .add(EntityAttributes.GENERIC_MOVEMENT_SPEED, {float(movement_speed):.3f})
            .add(EntityAttributes.GENERIC_FOLLOW_RANGE, 40.0);
    }}

    @Override
    protected void initGoals() {{
        this.goalSelector.add(1, new SwimGoal(this));
        this.goalSelector.add(2, new MeleeAttackGoal(this, 1.05, false));
        this.goalSelector.add(7, new WanderAroundFarGoal(this, 0.85));
        this.goalSelector.add(8, new LookAroundGoal(this));
        this.targetSelector.add(2, new ActiveTargetGoal<>(this, PlayerEntity.class, true));
    }}

    @Override
    public void registerControllers(AnimatableManager.ControllerRegistrar controllers) {{
        controllers.add(new AnimationController<>(this, "movement", 4, state ->
            state.setAndContinue(state.isMoving() ? WALK : IDLE)));
        controllers.add(new AnimationController<>(this, "attack", 0, state -> {{
            if (this.handSwinging) return state.setAndContinue(ATTACK);
            return state.stop();
        }}));
    }}

    @Override
    public AnimatableInstanceCache getAnimatableInstanceCache() {{
        return this.cache;
    }}
}}
'''


def _model_java(package_name: str, mod_id: str, entity_id: str, class_name: str, entity_class: str) -> str:
    return f'''package {package_name}.client.geckolib;

import {package_name}.entity.{entity_class};
import net.minecraft.util.Identifier;
import software.bernie.geckolib.model.GeoModel;

public final class {class_name}GeoModel extends GeoModel<{entity_class}> {{
    @Override
    public Identifier getModelResource({entity_class} entity) {{
        return new Identifier("{mod_id}", "geo/{entity_id}.geo.json");
    }}

    @Override
    public Identifier getTextureResource({entity_class} entity) {{
        return new Identifier("{mod_id}", "textures/entity/{entity_id}.png");
    }}

    @Override
    public Identifier getAnimationResource({entity_class} entity) {{
        return new Identifier("{mod_id}", "animations/{entity_id}.animation.json");
    }}
}}
'''


def _renderer_java(package_name: str, class_name: str, entity_class: str) -> str:
    return f'''package {package_name}.client.geckolib;

import {package_name}.entity.{entity_class};
import net.minecraft.client.render.entity.EntityRendererFactory;
import software.bernie.geckolib.renderer.GeoEntityRenderer;

public final class {class_name}GeoRenderer extends GeoEntityRenderer<{entity_class}> {{
    public {class_name}GeoRenderer(EntityRendererFactory.Context context) {{
        super(context, new {class_name}GeoModel());
        this.shadowRadius = 0.55f;
    }}
}}
'''


def _registrar_java_all(package_name: str, mod_id: str, entries: list[dict[str, Any]]) -> str:
    imports = "\n".join(
        f"import {package_name}.entity.{item['entity_class']};" for item in entries
    )
    declarations = "\n".join(
        f"    public static EntityType<{item['entity_class']}> {item['entity_id'].upper()};"
        for item in entries
    )
    registrations: list[str] = []
    for item in entries:
        constant = item["entity_id"].upper()
        entity_class = item["entity_class"]
        entity_id = item["entity_id"]
        registrations.append(
            f'''        {constant} = Registry.register(
            Registries.ENTITY_TYPE,
            new Identifier("{mod_id}", "{entity_id}"),
            FabricEntityTypeBuilder.create(SpawnGroup.MONSTER, {entity_class}::new)
                .dimensions(EntityDimensions.fixed(0.8f, 2.0f))
                .trackRangeBlocks(10)
                .build()
        );
        FabricDefaultAttributeRegistry.register({constant}, {entity_class}.createAttributes());'''
        )
    return f'''package {package_name}.geckolib;

{imports}
import net.fabricmc.fabric.api.object.builder.v1.entity.FabricDefaultAttributeRegistry;
import net.fabricmc.fabric.api.object.builder.v1.entity.FabricEntityTypeBuilder;
import net.minecraft.entity.EntityDimensions;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.SpawnGroup;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;
import software.bernie.geckolib.GeckoLib;

public final class GeneratedGeckoEntities {{
{declarations}
    private static boolean registered;

    private GeneratedGeckoEntities() {{}}

    public static synchronized void register() {{
        if (registered) return;
        registered = true;
        GeckoLib.initialize();
{chr(10).join(registrations)}
    }}
}}
'''


def _client_java_all(package_name: str, entries: list[dict[str, Any]]) -> str:
    registrations = "\n".join(
        f"        EntityRendererRegistry.register(GeneratedGeckoEntities.{item['entity_id'].upper()}, {item['class_name']}GeoRenderer::new);"
        for item in entries
    )
    return f'''package {package_name}.client.geckolib;

import {package_name}.geckolib.GeneratedGeckoEntities;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;

public final class GeneratedGeckoClient implements ClientModInitializer {{
    @Override
    public void onInitializeClient() {{
{registrations}
    }}
}}
'''
