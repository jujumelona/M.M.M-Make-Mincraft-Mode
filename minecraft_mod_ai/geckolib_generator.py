from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .generator import make_texture_png
from .project_edit import ensure_client_entrypoint, ensure_dependency, ensure_main_initializer_call, inspect_fabric_project, write_text_files
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher, sha256_file

_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_ARCHETYPES = {"biped", "quadruped", "flying", "serpentine", "construct", "custom"}
_BEHAVIORS = {"hostile_melee", "neutral_melee", "passive", "npc"}
_SPAWN = {"monster": "MONSTER", "creature": "CREATURE", "ambient": "AMBIENT", "water_creature": "WATER_CREATURE", "misc": "MISC"}

class GeckoLibGenerationError(ValueError):
    pass


def generate_geckolib_entity_assets(
    *, project_root: str | Path, mod_id: str, package_name: str, entity_id: str,
    texture_width: int = 64, texture_height: int = 64, geckolib_version: str = "4.8.2",
    max_health: float = 80.0, attack_damage: float = 8.0, movement_speed: float = 0.27,
    archetype: str = "biped", behavior: str = "hostile_melee", entity_width: float = 0.8,
    entity_height: float = 2.0, follow_range: float = 40.0, spawn_group: str | None = None,
    custom_bones: list[dict[str, Any]] | None = None, policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment(); policy.validate()
    if not _ID.fullmatch(mod_id) or not _ID.fullmatch(entity_id): raise GeckoLibGenerationError("Invalid mod or entity id.")
    if not _PACKAGE.fullmatch(package_name): raise GeckoLibGenerationError("Invalid Java package.")
    if type(texture_width) is not int or type(texture_height) is not int or not (1 <= texture_width <= policy.max_texture_dimension) or not (1 <= texture_height <= policy.max_texture_dimension):
        raise GeckoLibGenerationError("Texture dimensions exceed configured resource policy.")
    for name, value in {"max_health": max_health, "attack_damage": attack_damage, "movement_speed": movement_speed, "entity_width": entity_width, "entity_height": entity_height, "follow_range": follow_range}.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise GeckoLibGenerationError(f"{name} must be a positive finite number.")
    if archetype not in _ARCHETYPES or (archetype == "custom" and not custom_bones): raise GeckoLibGenerationError("Unknown or incomplete entity archetype.")
    if behavior not in _BEHAVIORS: raise GeckoLibGenerationError("Unknown entity behavior profile.")
    spawn_group = spawn_group or ("monster" if behavior == "hostile_melee" else "creature")
    if spawn_group not in _SPAWN: raise GeckoLibGenerationError("Unknown spawn group.")

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name: raise GeckoLibGenerationError("GeckoLib target does not match fabric.mod.json.")
    cls = "".join(part.capitalize() for part in entity_id.split("_")); entity_cls = cls + "Entity"
    package_path = package_name.replace(".", "/"); base = f"src/main/java/{package_path}"; assets = f"src/main/resources/assets/{mod_id}"
    manifest_path = info.root / ".minecraft_ai/geckolib-entities.json"
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8")); entries = {str(x["entity_id"]): dict(x) for x in raw.get("entities", [])}
    entries[entity_id] = {"entity_id": entity_id, "class_name": cls, "entity_class": entity_cls, "max_health": float(max_health), "attack_damage": float(attack_damage), "movement_speed": float(movement_speed), "follow_range": float(follow_range), "entity_width": float(entity_width), "entity_height": float(entity_height), "archetype": archetype, "behavior": behavior, "spawn_group": spawn_group}
    ordered = [entries[key] for key in sorted(entries)]
    bones = custom_bones if custom_bones is not None else _bones(archetype)
    files = {
        f"{assets}/geo/{entity_id}.geo.json": json.dumps(_geometry(mod_id, entity_id, texture_width, texture_height, bones), indent=2, sort_keys=True) + "\n",
        f"{assets}/animations/{entity_id}.animation.json": json.dumps(_animations(mod_id, entity_id, archetype), indent=2, sort_keys=True) + "\n",
        f"{base}/entity/{entity_cls}.java": _entity_java(package_name, mod_id, entity_id, entity_cls, float(max_health), float(attack_damage), float(movement_speed), float(follow_range), behavior),
        f"{base}/client/geckolib/{cls}GeoModel.java": _model_java(package_name, mod_id, entity_id, cls, entity_cls),
        f"{base}/client/geckolib/{cls}GeoRenderer.java": _renderer_java(package_name, cls, entity_cls, float(entity_width)),
        ".minecraft_ai/geckolib-entities.json": json.dumps({"schema_version": "mmm/geckolib-entities-v2", "shard_size": policy.entity_shard_size, "entities": ordered}, indent=2, sort_keys=True) + "\n",
    }
    files.update(_registrars(package_name, mod_id, base, ordered, policy.entity_shard_size))
    write_receipt = write_text_files(info, files, replace_existing=True)
    texture = info.root / f"{assets}/textures/entity/{entity_id}.png"; texture.parent.mkdir(parents=True, exist_ok=True)
    if not texture.exists(): texture.write_bytes(make_texture_png("#5ba6d8", entity_id, kind="entity", size=max(texture_width, texture_height)))
    dependency = ensure_dependency(info, repository_block='''maven {\n    name = 'GeckoLib'\n    url = 'https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/'\n}''', dependency_line=f'modImplementation("software.bernie.geckolib:geckolib-fabric-1.20.1:{geckolib_version}")', marker="geckolib")
    main = ensure_main_initializer_call(info, import_line=f"import {package_name}.geckolib.GeneratedGeckoEntities", call_line="GeneratedGeckoEntities.register()", marker="geckolib:entities")
    client = ensure_client_entrypoint(info, entrypoint=f"{package_name}.client.geckolib.GeneratedGeckoClient")
    metadata = json.loads(info.fabric_mod_json.read_text(encoding="utf-8")); depends = metadata.setdefault("depends", {})
    if not isinstance(depends, dict): raise GeckoLibGenerationError("fabric.mod.json depends must be an object.")
    if depends.get("geckolib") != f">={geckolib_version}":
        depends["geckolib"] = f">={geckolib_version}"
        meta_receipt: dict[str, Any] = TransactionalSourcePatcher(info.root).apply([{"operation": "replace", "path": "src/main/resources/fabric.mod.json", "expected_sha256": sha256_file(info.fabric_mod_json), "content": json.dumps(metadata, indent=2) + "\n"}])
    else: meta_receipt = {"status": "UNCHANGED"}
    return {"schema_version": "mmm/geckolib-generation-v3", "entity_id": entity_id, "archetype": archetype, "behavior": behavior, "entity_count": len(ordered), "registrar_shards": max(1, math.ceil(len(ordered) / policy.entity_shard_size)), "files": [str(info.root / p) for p in files] + [str(texture)], "status": "fabric_binding_generated", "receipts": {"files": write_receipt, "dependency": dependency, "main": main, "client": client, "metadata": meta_receipt}, "required_gates": ["Blockbench UV and bone hierarchy review", "Gradle clean build", "GameTest spawn and attributes", "runtime animation review"]}


def _bones(kind: str) -> list[dict[str, Any]]:
    root = [{"name": "root", "pivot": [0, 0, 0]}]
    if kind == "quadruped":
        return root + [{"name":"body","parent":"root","pivot":[0,10,0],"cubes":[{"origin":[-5,6,-8],"size":[10,8,16],"uv":[0,0]}]},{"name":"head","parent":"body","pivot":[0,12,-8],"cubes":[{"origin":[-4,8,-14],"size":[8,8,8],"uv":[0,24]}]}] + [{"name":name,"parent":"root","pivot":[x,7,z],"cubes":[{"origin":[x-2,-1,z-2],"size":[4,8,4],"uv":[32,24]}]} for name,x,z in (("front_left_leg",4,-5),("front_right_leg",-4,-5),("back_left_leg",4,5),("back_right_leg",-4,5))]
    if kind == "flying":
        return root + [{"name":"body","parent":"root","pivot":[0,12,0],"cubes":[{"origin":[-4,8,-4],"size":[8,8,8],"uv":[0,0]}]},{"name":"left_wing","parent":"body","pivot":[4,13,0],"cubes":[{"origin":[4,12,-2],"size":[12,1,8],"uv":[0,20]}]},{"name":"right_wing","parent":"body","pivot":[-4,13,0],"cubes":[{"origin":[-16,12,-2],"size":[12,1,8],"uv":[0,20]}]}]
    if kind == "serpentine":
        bones = root; parent = "root"
        for i in range(8):
            name=f"segment_{i}"; bones.append({"name":name,"parent":parent,"pivot":[0,3,i*4],"cubes":[{"origin":[-3,0,i*4],"size":[6,6,5],"uv":[i*6,0]}]}); parent=name
        return bones
    if kind == "construct":
        return root + [{"name":"core","parent":"root","pivot":[0,12,0],"cubes":[{"origin":[-6,6,-6],"size":[12,12,12],"uv":[0,0]}]},{"name":"left_arm","parent":"core","pivot":[7,15,0],"cubes":[{"origin":[6,5,-3],"size":[5,12,6],"uv":[32,24]}]},{"name":"right_arm","parent":"core","pivot":[-7,15,0],"cubes":[{"origin":[-11,5,-3],"size":[5,12,6],"uv":[32,24]}]}]
    return root + [{"name":"body","parent":"root","pivot":[0,12,0],"cubes":[{"origin":[-4,6,-2],"size":[8,12,4],"uv":[0,16]}]},{"name":"head","parent":"body","pivot":[0,18,0],"cubes":[{"origin":[-4,18,-4],"size":[8,8,8],"uv":[0,0]}]}] + [{"name":name,"parent":parent,"pivot":pivot,"cubes":[{"origin":origin,"size":[4,12,4],"uv":uv}]} for name,parent,pivot,origin,uv in (("right_arm","body",[-5,16,0],[-8,6,-2],[24,16]),("left_arm","body",[5,16,0],[4,6,-2],[40,16]),("right_leg","root",[-2,6,0],[-4,-6,-2],[0,32]),("left_leg","root",[2,6,0],[0,-6,-2],[16,32]))]


def _geometry(mod_id: str, entity_id: str, width: int, height: int, bones: list[dict[str, Any]]) -> dict[str, Any]:
    return {"format_version":"1.12.0","minecraft:geometry":[{"description":{"identifier":f"geometry.{mod_id}.{entity_id}","texture_width":width,"texture_height":height,"visible_bounds_width":8.0,"visible_bounds_height":8.0,"visible_bounds_offset":[0,2,0]},"bones":bones}]}


def _animations(mod_id: str, entity_id: str, archetype: str) -> dict[str, Any]:
    prefix=f"animation.{mod_id}.{entity_id}"; bones=_bones(archetype); moving={}
    for bone in bones:
        name=bone["name"]
        if any(token in name for token in ("leg","wing","segment","arm")):
            phase = 20 if sum(name.encode("utf-8")) % 2 else -20
            moving[name]={"rotation":{"0.0":[phase,0,0],"0.5":[-phase,0,0],"1.0":[phase,0,0]}}
    attack=next((b["name"] for b in bones if "arm" in b["name"] or b["name"] in {"head","core","segment_0"}),"root")
    return {"format_version":"1.8.0","animations":{f"{prefix}.idle":{"loop":True,"animation_length":2.0,"bones":{"root":{"rotation":{"0.0":[0,0,0],"1.0":[1.5,0,0],"2.0":[0,0,0]}}}},f"{prefix}.walk":{"loop":True,"animation_length":1.0,"bones":moving},f"{prefix}.attack":{"loop":False,"animation_length":0.5,"bones":{attack:{"rotation":{"0.0":[0,0,0],"0.2":[-75,0,0],"0.5":[0,0,0]}}}}}}


def _entity_java(package: str, mod_id: str, entity_id: str, cls: str, health: float, damage: float, speed: float, follow: float, behavior: str) -> str:
    melee='        this.goalSelector.add(2, new MeleeAttackGoal(this, 1.05, false));\n' if behavior in {"hostile_melee","neutral_melee"} else ''
    target='        this.targetSelector.add(2, new ActiveTargetGoal<>(this, PlayerEntity.class, true));\n' if behavior == "hostile_melee" else ''
    interaction='''\n    @Override protected ActionResult interactMob(PlayerEntity player, Hand hand) {\n        if (!getWorld().isClient) player.sendMessage(Text.literal("NPC: %s"), false);\n        return ActionResult.success(getWorld().isClient);\n    }\n''' % entity_id if behavior == "npc" else ''
    extra='import net.minecraft.util.ActionResult;\nimport net.minecraft.util.Hand;\n' if behavior == "npc" else ''
    return f'''package {package}.entity;\n\nimport net.minecraft.entity.EntityType;\nimport net.minecraft.entity.ai.goal.*;\nimport net.minecraft.entity.attribute.*;\nimport net.minecraft.entity.mob.MobEntity;\nimport net.minecraft.entity.mob.PathAwareEntity;\nimport net.minecraft.entity.player.PlayerEntity;\nimport net.minecraft.text.Text;\nimport net.minecraft.world.World;\n{extra}import software.bernie.geckolib.animatable.GeoEntity;\nimport software.bernie.geckolib.core.animatable.instance.AnimatableInstanceCache;\nimport software.bernie.geckolib.core.animation.*;\nimport software.bernie.geckolib.util.GeckoLibUtil;\n\npublic final class {cls} extends PathAwareEntity implements GeoEntity {{\n    private static final RawAnimation IDLE=RawAnimation.begin().thenLoop("animation.{mod_id}.{entity_id}.idle");\n    private static final RawAnimation WALK=RawAnimation.begin().thenLoop("animation.{mod_id}.{entity_id}.walk");\n    private static final RawAnimation ATTACK=RawAnimation.begin().thenPlay("animation.{mod_id}.{entity_id}.attack");\n    private final AnimatableInstanceCache cache=GeckoLibUtil.createInstanceCache(this);\n    public {cls}(EntityType<? extends PathAwareEntity> type, World world) {{ super(type,world); this.experiencePoints=30; }}\n    public static DefaultAttributeContainer.Builder createAttributes() {{ return MobEntity.createMobAttributes().add(EntityAttributes.GENERIC_MAX_HEALTH,{health:.3f}).add(EntityAttributes.GENERIC_ATTACK_DAMAGE,{damage:.3f}).add(EntityAttributes.GENERIC_MOVEMENT_SPEED,{speed:.5f}).add(EntityAttributes.GENERIC_FOLLOW_RANGE,{follow:.3f}); }}\n    @Override protected void initGoals() {{ this.goalSelector.add(1,new SwimGoal(this));\n{melee}        this.goalSelector.add(7,new WanderAroundFarGoal(this,0.85)); this.goalSelector.add(8,new LookAroundGoal(this));\n{target}    }}\n{interaction}    @Override public void registerControllers(AnimatableManager.ControllerRegistrar controllers) {{ controllers.add(new AnimationController<>(this,"movement",4,state->state.setAndContinue(state.isMoving()?WALK:IDLE))); controllers.add(new AnimationController<>(this,"attack",0,state->this.handSwinging?state.setAndContinue(ATTACK):state.stop())); }}\n    @Override public AnimatableInstanceCache getAnimatableInstanceCache() {{ return cache; }}\n}}\n'''


def _model_java(package: str, mod_id: str, entity_id: str, name: str, cls: str) -> str:
    return f'''package {package}.client.geckolib;\nimport {package}.entity.{cls};\nimport net.minecraft.util.Identifier;\nimport software.bernie.geckolib.model.GeoModel;\npublic final class {name}GeoModel extends GeoModel<{cls}> {{\n @Override public Identifier getModelResource({cls} e){{return new Identifier("{mod_id}","geo/{entity_id}.geo.json");}}\n @Override public Identifier getTextureResource({cls} e){{return new Identifier("{mod_id}","textures/entity/{entity_id}.png");}}\n @Override public Identifier getAnimationResource({cls} e){{return new Identifier("{mod_id}","animations/{entity_id}.animation.json");}}\n}}\n'''


def _renderer_java(package: str, name: str, cls: str, width: float) -> str:
    return f'''package {package}.client.geckolib;\nimport {package}.entity.{cls};\nimport net.minecraft.client.render.entity.EntityRendererFactory;\nimport software.bernie.geckolib.renderer.GeoEntityRenderer;\npublic final class {name}GeoRenderer extends GeoEntityRenderer<{cls}> {{ public {name}GeoRenderer(EntityRendererFactory.Context c){{super(c,new {name}GeoModel());this.shadowRadius={max(.1,min(4.,width*.55)):.3f}f;}} }}\n'''


def _registrars(package: str, mod_id: str, base: str, entries: list[dict[str, Any]], size: int) -> dict[str,str]:
    shards=[entries[i:i+size] for i in range(0,len(entries),size)]
    if len(shards)==1:
        return {f"{base}/geckolib/GeneratedGeckoEntities.java":_server_reg(package,mod_id,"GeneratedGeckoEntities",shards[0]),f"{base}/client/geckolib/GeneratedGeckoClient.java":_client_reg(package,"GeneratedGeckoClient","GeneratedGeckoEntities",shards[0],True)}
    files={}; server_names=[]; client_names=[]
    for i,shard in enumerate(shards):
        sn=f"GeneratedGeckoEntityShard{i:04d}"; cn=f"GeneratedGeckoClientShard{i:04d}"; server_names.append(sn); client_names.append(cn)
        files[f"{base}/geckolib/{sn}.java"]=_server_reg(package,mod_id,sn,shard); files[f"{base}/client/geckolib/{cn}.java"]=_client_reg(package,cn,sn,shard,False)
    files[f"{base}/geckolib/GeneratedGeckoEntities.java"]=_root_reg(package,server_names)
    files[f"{base}/client/geckolib/GeneratedGeckoClient.java"]=_root_client(package,client_names)
    return files


def _server_reg(package: str, mod_id: str, name: str, entries: list[dict[str, Any]]) -> str:
    imports='\n'.join(f'import {package}.entity.{e["entity_class"]};' for e in entries); fields='\n'.join(f' public static EntityType<{e["entity_class"]}> {e["entity_id"].upper()};' for e in entries); regs=[]
    for e in entries:
        c=e['entity_id'].upper(); regs.append(f'''  {c}=Registry.register(Registries.ENTITY_TYPE,new Identifier("{mod_id}","{e['entity_id']}"),FabricEntityTypeBuilder.create(SpawnGroup.{_SPAWN[e['spawn_group']]},{e['entity_class']}::new).dimensions(EntityDimensions.fixed({e['entity_width']:.4f}f,{e['entity_height']:.4f}f)).trackRangeBlocks(10).build());\n  FabricDefaultAttributeRegistry.register({c},{e['entity_class']}.createAttributes());''')
    return f'''package {package}.geckolib;\n{imports}\nimport net.fabricmc.fabric.api.object.builder.v1.entity.*;\nimport net.minecraft.entity.*;\nimport net.minecraft.registry.*;\nimport net.minecraft.util.Identifier;\nimport software.bernie.geckolib.GeckoLib;\npublic final class {name} {{\n{fields}\n private static boolean registered; private {name}(){{}}\n public static synchronized void register(){{if(registered)return;registered=true;GeckoLib.initialize();\n{chr(10).join(regs)}\n }}\n}}\n'''


def _client_reg(package: str, name: str, server: str, entries: list[dict[str, Any]], root: bool) -> str:
    regs='\n'.join(f'  EntityRendererRegistry.register({server}.{e["entity_id"].upper()},{e["class_name"]}GeoRenderer::new);' for e in entries); interface=' implements ClientModInitializer' if root else ''; override=' @Override' if root else ''; method='onInitializeClient' if root else 'register'
    return f'''package {package}.client.geckolib;\nimport {package}.geckolib.{server};\nimport net.fabricmc.api.ClientModInitializer;\nimport net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry;\npublic final class {name}{interface} {{{override} public {'void' if root else 'static void'} {method}(){{\n{regs}\n }} }}\n'''


def _root_reg(package: str, names: list[str]) -> str:
    calls=''.join(f'{n}.register();' for n in names); return f'package {package}.geckolib;\npublic final class GeneratedGeckoEntities{{private static boolean r;public static synchronized void register(){{if(r)return;r=true;{calls}}}}}\n'

def _root_client(package: str, names: list[str]) -> str:
    calls=''.join(f'{n}.register();' for n in names); return f'package {package}.client.geckolib;\nimport net.fabricmc.api.ClientModInitializer;\npublic final class GeneratedGeckoClient implements ClientModInitializer{{@Override public void onInitializeClient(){{{calls}}}}}\n'
