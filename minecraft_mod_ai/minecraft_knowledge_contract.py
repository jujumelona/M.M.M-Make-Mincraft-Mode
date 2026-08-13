from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import wraps
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "mmm/minecraft-knowledge-plan-v2"
COVERAGE_SCHEMA_VERSION = "mmm/minecraft-knowledge-route-coverage-v2"
_MARKER = "_mmm_minecraft_knowledge_contract_v2"


@dataclass(frozen=True)
class KnowledgeNode:
    id: str
    domain: str
    objective: str
    deps: tuple[str, ...] = ()
    mcp: tuple[str, ...] = ()
    checks: tuple[str, ...] = ("gradle_build",)
    kinds: tuple[str, ...] = ("minecraft_api", "source_code", "testing")
    providers: tuple[str, ...] = ("official_docs", "project_rag", "github")
    side: str = "common"

    @property
    def query(self) -> str:
        return (
            "Fabric Minecraft requested existing host-resolved target mappings "
            f"{self.id.replace('.', ' ')}: {self.objective}"
        )


def _n(
    id: str,
    domain: str,
    objective: str,
    *,
    deps: Sequence[str] = (),
    mcp: Sequence[str] = (),
    checks: Sequence[str] = ("gradle_build",),
    kinds: Sequence[str] = ("minecraft_api", "source_code", "testing"),
    providers: Sequence[str] = ("official_docs", "project_rag", "github"),
    side: str = "common",
) -> KnowledgeNode:
    return KnowledgeNode(
        id, domain, objective, tuple(deps), tuple(mcp), tuple(checks),
        tuple(kinds), tuple(providers), side,
    )


# Host-owned ontology: model output may select neither nodes nor dependencies.
NODES = {
    node.id: node
    for node in (
        _n("platform.fabric_target", "platform", "Lock the exact host-resolved Minecraft version, Fabric Loader/API, Loom, mappings, Java and Gradle profile.", mcp=("version_diff", "official_mod_docs"), checks=("platform_lock", "gradle_configuration"), kinds=("minecraft_api", "dependency", "compatibility", "testing")),
        _n("platform.mappings", "platform", "Resolve exact mappings names/signatures instead of guessing symbols.", deps=("platform.fabric_target",), mcp=("mapping_resolution", "source_search"), checks=("mapping_target_match",), kinds=("minecraft_api", "source_code", "compatibility")),
        _n("project.structure", "project", "Reuse current namespace, entrypoints, source sets, helpers and dependency boundaries.", deps=("platform.fabric_target",), mcp=("workspace_validation",), checks=("project_index_current",), kinds=("local_project", "source_code", "dependency", "testing"), providers=("project_rag", "github")),
        _n("registry.content", "registry", "Use target-correct registries and unique identifiers.", deps=("platform.mappings",), mcp=("registry_lookup", "source_search"), checks=("registry_ids_unique",)),
        _n("lifecycle.common", "lifecycle", "Keep common/server/client/world lifecycle and ticks on the correct logical side.", deps=("project.structure", "platform.mappings"), mcp=("official_mod_docs", "source_search"), checks=("client_server_boundary", "lifecycle_smoke"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("events.fabric", "lifecycle", "Prefer supported Fabric events/callbacks before invasive hooks.", deps=("lifecycle.common",), mcp=("official_mod_docs", "source_search"), checks=("event_hook_valid",)),
        _n("data.persistence", "data", "Persist authoritative state with target-correct NBT/codecs and save/load boundaries.", deps=("lifecycle.common",), mcp=("source_search", "mod_examples"), checks=("save_reload", "state_roundtrip"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("resources.lang", "resources", "Provide translation keys/language resources for user-facing content.", deps=("project.structure",), checks=("resource_json", "translation_key_resolution"), kinds=("minecraft_api", "local_project", "testing"), side="client"),
        _n("resources.model_texture", "resources", "Bind models/textures to exact identifiers and resource paths.", deps=("project.structure",), mcp=("official_mod_docs",), checks=("resource_exists", "resource_identifier_match"), kinds=("minecraft_api", "local_project", "testing"), side="client"),
        _n("resources.sound", "resources", "Resolve vanilla-vs-custom sound; register assets only when custom and play on the correct side.", deps=("registry.content", "lifecycle.common"), mcp=("registry_lookup", "source_search", "mod_examples"), checks=("sound_registry", "sound_runtime"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("resources.particle", "resources", "Resolve vanilla-vs-custom particle; register client factories only when required.", deps=("registry.content", "lifecycle.common"), mcp=("registry_lookup", "source_search", "mod_examples"), checks=("particle_registry", "client_server_boundary", "particle_runtime"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("item.registration", "item_block", "Register requested custom items with target-correct settings.", deps=("registry.content",), mcp=("registry_lookup", "mod_examples"), checks=("item_registry", "item_runtime")),
        _n("item.group", "item_block", "Place requested items in the correct creative item group.", deps=("item.registration",), mcp=("official_mod_docs", "source_search"), checks=("item_group_runtime",)),
        _n("block.registration", "item_block", "Register requested blocks and optional block items.", deps=("registry.content",), mcp=("registry_lookup", "mod_examples"), checks=("block_registry", "block_runtime")),
        _n("block_entity.lifecycle", "item_block", "Register block entities and bind ticking/persistence to block lifecycle.", deps=("block.registration", "data.persistence"), mcp=("source_search", "mod_examples"), checks=("block_entity_load_save", "server_tick"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("inventory.container", "ui", "Keep inventory/container mutation server-owned with valid slot rules.", deps=("block_entity.lifecycle",), mcp=("source_search", "mod_examples"), checks=("inventory_roundtrip", "slot_validation"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("rendering.client_boundary", "rendering", "Keep renderer/model/HUD/key classes out of dedicated-server classloading.", deps=("lifecycle.common",), mcp=("source_search",), checks=("dedicated_server_smoke", "client_server_boundary"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing"), side="client"),
        _n("rendering.entity", "rendering", "Register entity renderer/model/texture client-side against the exact EntityType.", deps=("entity.registration", "rendering.client_boundary", "resources.model_texture"), mcp=("source_search", "mod_examples"), checks=("entity_render_runtime", "dedicated_server_smoke"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing"), side="client"),
        _n("rendering.block_entity", "rendering", "Register BlockEntityRenderer entirely client-side against the exact type.", deps=("block_entity.lifecycle", "rendering.client_boundary", "resources.model_texture"), mcp=("source_search", "mod_examples"), checks=("block_entity_render_runtime", "dedicated_server_smoke"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing"), side="client"),
        _n("animation.entity", "rendering", "Choose vanilla animation hooks or a compatible library only when requested.", deps=("rendering.entity",), mcp=("source_search", "mod_examples", "mod_jar_analysis"), checks=("animation_runtime", "dedicated_server_smoke"), kinds=("minecraft_api", "source_code", "dependency", "compatibility", "testing"), side="client"),
        _n("ui.client_screen", "ui", "Implement display/input client-side without making UI authoritative.", deps=("rendering.client_boundary",), mcp=("official_mod_docs", "mod_examples"), checks=("client_only_classloading", "gui_runtime"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing"), side="client"),
        _n("ui.hud", "ui", "Render HUD client-side while sourcing gameplay state from authoritative state.", deps=("rendering.client_boundary",), mcp=("official_mod_docs", "source_search"), checks=("hud_runtime", "client_only_classloading"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing"), side="client"),
        _n("ui.screen_handler", "ui", "Use ScreenHandler for server-authoritative container/menu synchronization.", deps=("inventory.container",), mcp=("source_search", "mod_examples"), checks=("screen_handler_sync",), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("networking.payloads", "networking", "Use target-correct C2S/S2C registration, validation and server authority.", deps=("lifecycle.common", "platform.mappings"), mcp=("official_mod_docs", "source_search", "mapping_resolution"), checks=("packet_schema", "malformed_packet", "server_authority"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("networking.state_sync", "networking", "Synchronize authoritative custom state for joins/reconnects.", deps=("networking.payloads",), mcp=("source_search", "mod_examples"), checks=("multiplayer_sync", "late_join_sync"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("entity.registration", "entity", "Register EntityType, dimensions, factory and vanilla tracking parameters.", deps=("registry.content",), mcp=("registry_lookup", "source_search", "mapping_resolution"), checks=("entity_spawn_runtime", "entity_registry")),
        _n("entity.attributes", "entity", "Register living-entity attributes before spawn.", deps=("entity.registration",), mcp=("source_search", "mod_examples"), checks=("entity_attributes",)),
        _n("entity.ai_goals", "entity", "Implement goals/navigation/targeting without blocking server ticks.", deps=("entity.attributes",), mcp=("source_search", "mapping_resolution"), checks=("ai_tick_runtime", "no_tick_blocking"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("entity.tracked_data", "entity", "Use vanilla tracked data; do not invent custom packets unless needed.", deps=("entity.registration", "lifecycle.common"), mcp=("source_search", "mapping_resolution"), checks=("entity_tracking", "late_join_entity_state"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("entity.spawn", "entity", "Implement only the requested natural/explicit/spawn-egg path.", deps=("entity.registration",), mcp=("source_search", "registry_lookup"), checks=("spawn_conditions",)),
        _n("boss.bossbar", "entity", "Bind boss-bar players/health/lifecycle to the authoritative boss entity.", deps=("entity.registration", "entity.tracked_data"), mcp=("source_search", "mapping_resolution"), checks=("bossbar_health_sync", "bossbar_join_leave"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("combat.damage", "combat", "Use target-correct damage types/sources and server-authoritative combat.", deps=("lifecycle.common", "platform.mappings"), mcp=("source_search", "mapping_resolution"), checks=("combat_runtime", "server_authority"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("combat.projectile", "combat", "Implement projectile ownership/spawn/collision/damage on the server.", deps=("entity.registration", "entity.tracked_data", "combat.damage"), mcp=("source_search", "mapping_resolution"), checks=("projectile_collision", "multiplayer_sync"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("attributes.custom", "combat", "Register custom attributes only when the mechanic truly needs one.", deps=("registry.content", "lifecycle.common"), mcp=("registry_lookup", "source_search", "mapping_resolution"), checks=("attribute_registry", "attribute_runtime"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("effects.status", "combat", "Register/apply/remove requested status effects server-side.", deps=("registry.content", "lifecycle.common"), mcp=("registry_lookup", "source_search", "mod_examples"), checks=("status_effect_registry", "status_effect_runtime"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("enchantment.custom", "combat", "Implement custom enchantment registration/applicability/effect hooks.", deps=("registry.content", "events.fabric"), mcp=("registry_lookup", "source_search", "mapping_resolution", "mod_examples"), checks=("enchantment_registry", "enchantment_runtime"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("datagen.recipe", "datagen", "Generate/validate recipes against registered identifiers.", deps=("registry.content",), mcp=("official_mod_docs",), checks=("datagen", "recipe_load")),
        _n("datagen.loot", "datagen", "Generate/validate block/entity loot and requested drops.", deps=("registry.content",), mcp=("official_mod_docs",), checks=("datagen", "loot_runtime")),
        _n("datagen.tags_advancements", "datagen", "Generate tags/advancements and validate every referenced id.", deps=("registry.content",), mcp=("official_mod_docs",), checks=("datagen", "resource_json")),
        _n("worldgen.structure", "worldgen", "Use native structure/worldgen registration and placement, not map editing.", deps=("registry.content",), mcp=("source_search", "registry_lookup", "mod_examples"), checks=("worldgen_runtime", "new_world_smoke"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("worldgen.feature", "worldgen", "Register configured/placed features or ore generation correctly.", deps=("registry.content",), mcp=("source_search", "registry_lookup", "mod_examples"), checks=("worldgen_runtime", "new_world_smoke"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("worldgen.biome", "worldgen", "Use target-version biome registry/modification hooks.", deps=("registry.content",), mcp=("source_search", "registry_lookup"), checks=("worldgen_runtime",), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("worldgen.dimension", "worldgen", "Use data-driven dimension contracts and server-world lifecycle.", deps=("registry.content", "data.persistence"), mcp=("source_search", "registry_lookup"), checks=("dimension_load", "save_reload"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("worldgen.portal", "worldgen", "Validate custom portal/dimension travel server-side.", deps=("worldgen.dimension", "lifecycle.common"), mcp=("source_search", "mapping_resolution", "mod_examples"), checks=("portal_travel_runtime", "dimension_load"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("villager.trade", "gameplay", "Modify villager trades through supported target-version hooks.", deps=("events.fabric", "registry.content"), mcp=("source_search", "mod_examples", "registry_lookup"), checks=("villager_trade_runtime",), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("rules.gamerule", "gameplay", "Register/integrate server game rules and persist authoritative values.", deps=("lifecycle.common", "data.persistence"), mcp=("source_search", "mapping_resolution"), checks=("gamerule_runtime", "save_reload"), kinds=("minecraft_api", "source_code", "runtime_behavior", "testing")),
        _n("commands.brigadier", "commands", "Register Brigadier commands with permission/argument validation.", deps=("lifecycle.common",), mcp=("official_mod_docs", "source_search"), checks=("command_permissions", "command_runtime")),
        _n("mixin.injection", "mixin", "Use Mixins/access wideners only when supported APIs are insufficient; validate exact targets.", deps=("platform.mappings", "project.structure"), mcp=("mixin_validation", "access_widener_validation", "source_search", "mapping_resolution"), checks=("mixin_target_validation", "dedicated_server_smoke"), kinds=("minecraft_api", "source_code", "compatibility", "testing")),
        _n("config.persistence", "data", "Persist mod config separately from authoritative game state.", deps=("project.structure",), checks=("config_roundtrip",), kinds=("local_project", "source_code", "testing"), providers=("project_rag", "github")),
        _n("input.keybinding", "ui", "Register keybindings client-side; server-validate gameplay mutation.", deps=("rendering.client_boundary",), mcp=("official_mod_docs",), checks=("client_only_classloading",), side="client"),
        _n("library.integration", "library", "Pin external libraries to compatible artifacts and isolate their API boundary.", deps=("platform.fabric_target",), mcp=("mod_jar_analysis",), checks=("dependency_resolution", "license_origin"), kinds=("dependency", "compatibility", "license", "source_code")),
        _n("quality.compile", "quality", "Compile the exact project with the host-resolved Java version and locked Fabric toolchain.", deps=("platform.fabric_target", "project.structure"), mcp=("workspace_validation",), checks=("gradle_build",), kinds=("testing", "local_project", "compatibility"), providers=("official_docs", "project_rag", "runtime")),
        _n("quality.gametest", "quality", "Turn observable gameplay requirements into GameTest/equivalent assertions.", deps=("quality.compile",), mcp=("official_mod_docs",), checks=("gametest",), kinds=("testing", "runtime_behavior"), providers=("official_docs", "project_rag", "runtime")),
        _n("quality.runtime", "quality", "Run disposable dedicated-server/client smoke tests and inspect behavior.", deps=("quality.compile",), mcp=("runtime_inspection", "runtime_server_status", "runtime_visual"), checks=("dedicated_server_smoke", "client_smoke"), kinds=("testing", "runtime_behavior", "local_project"), providers=("official_docs", "project_rag", "runtime")),
        _n("release.packaging", "release", "Validate fabric.mod.json, dependency bounds, resources and remapped JAR.", deps=("quality.compile",), mcp=("mod_jar_analysis", "workspace_validation"), checks=("jar_contents", "metadata_validation"), kinds=("release", "dependency", "compatibility", "testing")),
        _n("custom.source_extension", "custom", "For unusual mechanics inspect exact source/mappings and nearest project patterns before custom Java.", deps=("platform.mappings", "project.structure", "quality.compile"), mcp=("source_search", "mapping_resolution", "mod_examples"), checks=("gradle_build", "runtime_acceptance"), kinds=("minecraft_api", "source_code", "local_project", "testing")),
    )
}


FEATURE_ROOTS: dict[str, tuple[str, ...]] = {
    "base_mod": ("platform.fabric_target", "platform.mappings", "project.structure", "quality.compile", "release.packaging"),
    "custom_item": ("item.registration", "resources.model_texture", "resources.lang"),
    "item_group": ("item.group", "resources.lang"),
    "custom_block": ("block.registration", "resources.model_texture", "resources.lang", "datagen.loot"),
    "machine_with_gui": ("block.registration", "block_entity.lifecycle", "inventory.container", "ui.screen_handler", "ui.client_screen", "networking.payloads", "networking.state_sync", "resources.model_texture", "resources.lang", "datagen.loot", "quality.gametest"),
    "custom_entity": ("entity.registration", "entity.tracked_data", "rendering.entity", "resources.lang", "quality.gametest"),
    "custom_mob": ("entity.registration", "entity.attributes", "entity.ai_goals", "entity.tracked_data", "entity.spawn", "rendering.entity", "resources.lang", "quality.gametest"),
    "custom_boss": ("entity.registration", "entity.attributes", "entity.ai_goals", "entity.tracked_data", "entity.spawn", "rendering.entity", "combat.damage", "boss.bossbar", "datagen.loot", "resources.sound", "resources.particle", "custom.source_extension", "quality.gametest", "quality.runtime"),
    "projectile": ("combat.projectile", "rendering.entity", "quality.gametest"),
    "status_effect": ("effects.status", "resources.lang", "quality.gametest"),
    "enchantment": ("enchantment.custom", "resources.lang", "quality.gametest"),
    "custom_attribute": ("attributes.custom", "quality.gametest"),
    "recipe": ("datagen.recipe",), "loot": ("datagen.loot",), "advancement": ("datagen.tags_advancements", "resources.lang"),
    "sound": ("resources.sound",), "particle": ("resources.particle",), "gui": ("ui.client_screen",), "hud": ("ui.hud",),
    "networking": ("networking.payloads", "networking.state_sync", "quality.runtime"), "persistence": ("data.persistence",),
    "world_structure": ("worldgen.structure", "quality.runtime"), "world_feature": ("worldgen.feature", "quality.runtime"),
    "biome": ("worldgen.biome", "quality.runtime"), "dimension": ("worldgen.dimension", "quality.runtime"), "custom_portal": ("worldgen.portal", "quality.runtime"),
    "villager_trade": ("villager.trade", "quality.gametest"), "gamerule": ("rules.gamerule", "quality.gametest"),
    "command": ("commands.brigadier",), "mixin": ("mixin.injection",), "config": ("config.persistence",),
    "keybinding": ("input.keybinding",), "library": ("library.integration",), "events": ("events.fabric",),
    "block_entity_renderer": ("rendering.block_entity",), "entity_animation": ("animation.entity",),
    "custom_java": ("custom.source_extension", "quality.gametest", "quality.runtime"),
}

FEATURE_TERMS: dict[str, tuple[str, ...]] = {
    "custom_boss": ("boss", "보스"), "custom_mob": ("mob", "monster", "npc", "몹", "몬스터"), "custom_entity": ("entity", "엔티티"),
    "custom_item": ("weapon", "armor", "tool", "food", "sword", "pickaxe", "axe", "bow", "무기", "갑옷", "도구", "음식", "검", "곡괭이", "도끼", "활"),
    "item_group": ("creative tab", "item group", "크리에이티브 탭", "아이템 그룹"), "custom_block": ("block", "블록"),
    "projectile": ("projectile", "bullet", "arrow", "missile", "투사체", "발사체", "총알", "미사일"),
    "status_effect": ("status effect", "potion effect", "상태 효과", "상태효과", "포션 효과", "버프", "디버프"),
    "enchantment": ("enchantment", "enchant", "마법부여", "인챈트"), "custom_attribute": ("custom attribute", "속성 추가", "커스텀 속성"),
    "recipe": ("recipe", "crafting", "조합법", "레시피", "제작법"), "loot": ("loot", "drop", "드롭", "전리품"),
    "advancement": ("advancement", "achievement", "발전과제", "도전과제", "업적"), "sound": ("sound", "music", "audio", "사운드", "효과음", "소리", "음악"),
    "particle": ("particle", "파티클", "입자"), "gui": ("gui", "screen", "menu", "화면", "메뉴", "인터페이스"),
    "hud": ("hud", "stamina bar", "스태미나 바", "스태미너 바"), "networking": ("network", "packet", "multiplayer", "네트워크", "패킷", "멀티플레이", "멀티"),
    "persistence": ("persist", "save", "load", "nbt", "저장", "불러오기", "영속"),
    "world_structure": ("structure", "worldgen", "world generation", "구조물", "월드젠", "월드 생성"),
    "world_feature": ("ore generation", "placed feature", "configured feature", "광물 생성", "광석 생성", "배치 피처", "월드 피처"),
    "biome": ("biome", "바이옴"), "dimension": ("dimension", "차원"), "custom_portal": ("portal", "포탈", "포털"),
    "villager_trade": ("villager trade", "trading", "주민 거래", "거래 목록"), "gamerule": ("gamerule", "game rule", "게임룰", "게임 규칙"),
    "command": ("command", "brigadier", "명령어", "커맨드"), "mixin": ("mixin", "access widener", "믹스인", "액세스 위드너"),
    "config": ("configuration", "config file", "설정 파일", "구성 파일"), "keybinding": ("keybind", "key binding", "단축키", "키바인딩"),
    "library": ("geckolib", "library", "dependency", "라이브러리", "외부 의존성"), "events": ("event", "callback", "이벤트", "콜백"),
    "block_entity_renderer": ("block entity renderer", "블록 엔티티 렌더"), "entity_animation": ("animation", "geckolib", "애니메이션"),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: str | Any) -> str:
    text = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _contains(text: str, term: str) -> bool:
    folded, needle = text.casefold(), term.casefold()
    if needle.isascii() and re.fullmatch(r"[a-z0-9_ ]+", needle):
        return re.search(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])", folded) is not None
    return needle in folded


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_flatten(v) for k, v in value.items() if not str(k).startswith("_"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten(v) for v in value)
    return ""


def detect_features(prompt: str, game_design: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    text = f"{prompt}\n{_flatten(game_design or {})}"
    features = {"base_mod"}
    for feature, terms in FEATURE_TERMS.items():
        if any(_contains(text, term) for term in terms):
            features.add(feature)
    folded = text.casefold()
    if (
        re.search(r"\b(?:add|create|make|new|custom)\b.{0,40}\bitem\b", folded)
        or re.search(r"\bitem\b.{0,40}\b(?:add|create|make|register)\b", folded)
        or re.search(r"(?:새|신규|커스텀).{0,20}아이템", text)
        or re.search(r"아이템(?:을|를)?\s*(?:추가|등록|만들|제작)", text)
    ):
        features.add("custom_item")
    if any(_contains(text, x) for x in ("machine", "기계", "장치", "설비")) and (
        any(_contains(text, x) for x in ("inventory", "container", "인벤토리", "보관함"))
        or "gui" in features or "networking" in features
    ):
        features.add("machine_with_gui")
    if "custom_boss" in features:
        features.update(("custom_mob", "custom_entity"))
    elif "custom_mob" in features:
        features.add("custom_entity")
    if len(features) == 1:
        features.add("custom_java")
    return tuple(sorted(features))


def _expand(roots: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    visiting: set[str] = set()
    out: list[str] = []

    def visit(node_id: str) -> None:
        if node_id in seen:
            return
        if node_id in visiting or node_id not in NODES:
            raise RuntimeError(f"Invalid Minecraft knowledge dependency: {node_id}")
        visiting.add(node_id)
        for dep in NODES[node_id].deps:
            visit(dep)
        visiting.remove(node_id)
        seen.add(node_id)
        out.append(node_id)

    for root in sorted(set(roots)):
        visit(root)
    return tuple(out)


def expand_features(features: Sequence[str]) -> tuple[str, ...]:
    return _expand([root for feature in features for root in FEATURE_ROOTS.get(feature, ())])


def compile_minecraft_knowledge_plan(prompt: str, game_design: Mapping[str, Any] | None = None) -> dict[str, Any]:
    features = detect_features(prompt, game_design)
    order = expand_features(features)
    closures = {f: set(expand_features((f,))) for f in features if f in FEATURE_ROOTS}
    requirements = [
        {
            "requirement_id": f"mk:{node_id}",
            "knowledge_id": node_id,
            "domain": NODES[node_id].domain,
            "objective": NODES[node_id].objective,
            "depends_on": [f"mk:{dep}" for dep in NODES[node_id].deps],
            "feature_refs": sorted(f for f, closure in closures.items() if node_id in closure),
            "side": NODES[node_id].side,
            "evidence": {
                "rag_queries": [NODES[node_id].query],
                "mcp_capabilities": list(NODES[node_id].mcp),
                "providers": list(NODES[node_id].providers),
                "evidence_kinds": list(NODES[node_id].kinds),
            },
            "validations": list(NODES[node_id].checks),
        }
        for node_id in order
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in requirements:
        grouped.setdefault(str(item["domain"]), []).append(item)
    research_domains = []
    for domain, items in sorted(grouped.items()):
        mcp = sorted({cap for item in items for cap in item["evidence"]["mcp_capabilities"]})
        research_domains.append(
            {
                "domain_id": "mk_" + re.sub(r"[^a-z0-9_]+", "_", domain.casefold()),
                "objective": f"Resolve mandatory Minecraft/Fabric {domain} knowledge. "
                + (f"If RAG is insufficient use exact research MCP: {', '.join(mcp)}." if mcp else "Do not guess APIs."),
                "requirements": [str(item["requirement_id"]) for item in items],
                "evidence_kinds": list(dict.fromkeys(k for item in items for k in item["evidence"]["evidence_kinds"])),
                "queries": list(dict.fromkeys(q for item in items for q in item["evidence"]["rag_queries"])),
                "providers": list(dict.fromkeys(p for item in items for p in item["evidence"]["providers"])),
                "depends_on": [],
            }
        )
    all_mcp = sorted({cap for item in requirements for cap in item["evidence"]["mcp_capabilities"]})
    plan = {
        "schema_version": SCHEMA_VERSION,
        "features": list(features),
        "knowledge_order": list(order),
        "requirements": requirements,
        "research_domains": research_domains,
        "mcp_requirements": [
            {
                "capability": cap,
                "requirement_refs": sorted(
                    str(item["requirement_id"]) for item in requirements if cap in item["evidence"]["mcp_capabilities"]
                ),
            }
            for cap in all_mcp
        ],
        "policy": {
            "dependency_expansion_owner": "host",
            "model_may_remove_requirements": False,
            "versioned_rag_required_before_design": True,
            "mcp_capability_selection_owner": "host",
            "mcp_execution": "research_agent_or_downstream_specialist_when_exact_lookup_is_needed",
            "validation_selection_owner": "host",
        },
        "plan_sha256": "",
    }
    plan["plan_sha256"] = _sha({**plan, "plan_sha256": ""})
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Minecraft knowledge plan schema.")
    items = plan.get("requirements")
    if not isinstance(items, list) or not items:
        raise ValueError("Minecraft knowledge plan has no requirements.")
    ids = [str(item.get("requirement_id", "")) for item in items if isinstance(item, Mapping)]
    if len(ids) != len(items) or len(ids) != len(set(ids)):
        raise ValueError("Invalid/duplicate Minecraft knowledge requirement ids.")
    known = set(ids)
    for item in items:
        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping) or not evidence.get("rag_queries") or not item.get("validations"):
            raise ValueError(f"Incomplete Minecraft knowledge contract: {item.get('requirement_id')}")
        if any(dep not in known for dep in item.get("depends_on", [])):
            raise ValueError(f"Unknown Minecraft knowledge dependency: {item.get('requirement_id')}")
    if plan.get("plan_sha256") != _sha({**dict(plan), "plan_sha256": ""}):
        raise ValueError("Minecraft knowledge plan hash mismatch.")


def _augment_brief(normalize: Any, prompt: str, design: Mapping[str, Any], plan: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    existing = [dict(x) for x in base.get("domains", []) if isinstance(x, Mapping)]
    ids = {str(x.get("domain_id", "")) for x in existing}
    candidate = {
        "summary": str(base.get("summary", "")) + " Host-expanded Minecraft feature dependencies are mandatory.",
        "domains": existing + [dict(x) for x in plan["research_domains"] if x["domain_id"] not in ids],
        "unresolved_questions": list(base.get("unresolved_questions", [])),
    }
    return normalize(prompt, dict(design), candidate)


def evaluate_route_coverage(plan: Mapping[str, Any], research: Mapping[str, Any]) -> dict[str, Any]:
    validate_plan(plan)
    expected = {str(x["domain_id"]): x for x in plan["research_domains"]}
    brief_ids = {
        str(x.get("domain_id", ""))
        for x in research.get("research_brief", {}).get("domains", [])
        if isinstance(x, Mapping)
    }
    deterministic = research.get("deterministic")
    forced = deterministic.get("forced_project_rag") if isinstance(deterministic, Mapping) else None
    forced_map = {
        str(x.get("domain_id", "")): x
        for x in (forced.get("domains", []) if isinstance(forced, Mapping) else [])
        if isinstance(x, Mapping)
    }
    notes = {
        str(x.get("domain_id", "")): x
        for x in research.get("domain_notes", [])
        if isinstance(x, Mapping)
    }
    receipts, blocking, deferred = [], [], []
    for domain_id, domain in expected.items():
        executed = forced_map.get(domain_id)
        raw_queries = executed.get("queries", []) if isinstance(executed, Mapping) else []
        got = {str(x.get("query_sha256", "")) for x in raw_queries if isinstance(x, Mapping)}
        missing = sorted({_sha(str(q)) for q in domain["queries"]} - got)
        note = notes.get(domain_id)
        if domain_id not in brief_ids:
            status = "MISSING_RESEARCH_DOMAIN"
        elif executed is None:
            status = "MISSING_FORCED_RAG_RECEIPT"
        elif missing:
            status = "MISSING_FORCED_RAG_QUERY"
        elif note is None:
            status = "MISSING_RESEARCH_AGENT_NOTE"
        elif bool(note.get("sufficient")):
            status = "ROUTES_EXECUTED"
        elif bool(note.get("fixed_point")):
            # A fixed point is a terminal research outcome, not evidence that the
            # required route failed to execute. Preserve the gaps for downstream
            # exact lookup/validation instead of deadlocking pre-design planning.
            status = "ROUTES_EXECUTED_WITH_GAPS"
        else:
            status = "RESEARCH_UNRESOLVED"
        if status in {
            "MISSING_RESEARCH_DOMAIN",
            "MISSING_FORCED_RAG_RECEIPT",
            "MISSING_FORCED_RAG_QUERY",
            "MISSING_RESEARCH_AGENT_NOTE",
            "RESEARCH_UNRESOLVED",
        }:
            blocking.extend(str(ref) for ref in domain["requirements"])
        elif status == "ROUTES_EXECUTED_WITH_GAPS":
            deferred.extend(str(ref) for ref in domain["requirements"])
        receipts.append(
            {
                "domain_id": domain_id,
                "status": status,
                "query_count": len(domain["queries"]),
                "forced_query_count": len(raw_queries),
                "missing_query_sha256": missing,
                "research_agent_sufficient": bool(note.get("sufficient")) if isinstance(note, Mapping) else False,
                "research_agent_fixed_point": bool(note.get("fixed_point")) if isinstance(note, Mapping) else False,
            }
        )
    result = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "status": "PASS" if not blocking else "BLOCK",
        "blocking_requirement_refs": sorted(set(blocking)),
        "deferred_requirement_refs": sorted(set(deferred)),
        "domains": receipts,
        "semantics": (
            "PASS proves every host-required domain entered the brief, every deterministic forced-RAG query "
            "has an execution receipt, and every domain research agent produced a terminal note. A terminal "
            "fixed point may retain explicit deferred gaps for downstream exact lookup/validation; PASS does "
            "not claim those gaps are resolved or that optional MCP lookups ran."
        ),
        "coverage_sha256": "",
    }
    result["coverage_sha256"] = _sha({**result, "coverage_sha256": ""})
    return result


def compact_plan(plan: Mapping[str, Any], coverage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": plan["schema_version"],
        "features": plan["features"],
        "knowledge_order": plan["knowledge_order"],
        "requirements": [
            {
                "requirement_id": x["requirement_id"], "knowledge_id": x["knowledge_id"],
                "objective": x["objective"], "depends_on": x["depends_on"], "side": x["side"],
                "mcp_capabilities": x["evidence"]["mcp_capabilities"], "validations": x["validations"],
            }
            for x in plan["requirements"]
        ],
        "mcp_requirements": plan["mcp_requirements"],
        "route_coverage": dict(coverage or {}),
        "policy": plan["policy"],
        "plan_sha256": plan["plan_sha256"],
    }


def install(agentic_module: Any, complete_planner_module: Any | None = None) -> None:
    """Make deterministic Minecraft knowledge expansion outermost around research/design."""

    current_normalize = agentic_module.normalize_research_brief
    if not getattr(current_normalize, _MARKER, False):
        original = current_normalize

        @wraps(original)
        def normalize(prompt: str, design: dict[str, Any], candidate: Any | None = None):
            base = original(prompt, design, candidate)
            return _augment_brief(original, prompt, design, compile_minecraft_knowledge_plan(prompt, design), base)

        setattr(normalize, _MARKER, True)
        normalize.__wrapped__ = original  # type: ignore[attr-defined]
        agentic_module.normalize_research_brief = normalize

    current_collect = agentic_module.collect_pre_design_research
    if not getattr(current_collect, _MARKER, False):

        @wraps(current_collect)
        def collect(router: Any, prompt: str, *, trace_metadata=None):
            result = dict(current_collect(router, prompt, trace_metadata=trace_metadata))
            plan = compile_minecraft_knowledge_plan(prompt)
            coverage = evaluate_route_coverage(plan, result)
            if coverage["status"] != "PASS":
                error = getattr(agentic_module, "SpecValidationError", RuntimeError)
                blocked_domains = [
                    item
                    for item in coverage.get("domains", [])
                    if isinstance(item, Mapping)
                    and str(item.get("status", "")) not in {
                        "ROUTES_EXECUTED",
                        "ROUTES_EXECUTED_WITH_GAPS",
                    }
                ]
                domain_detail = "; ".join(
                    f"{item.get('domain_id', 'unknown')}={item.get('status', 'unknown')}"
                    for item in blocked_domains[:12]
                )
                notes = {
                    str(item.get("domain_id", "")): item
                    for item in result.get("domain_notes", [])
                    if isinstance(item, Mapping)
                }
                failures = []
                for item in blocked_domains[:4]:
                    note = notes.get(str(item.get("domain_id", "")))
                    if not isinstance(note, Mapping) or not note.get("worker_error"):
                        continue
                    failure = str(note.get("retry_error") or note.get("parallel_error") or "").strip()
                    if failure:
                        failures.append(
                            f"{item.get('domain_id', 'unknown')}:{failure[:400]}"
                        )
                message = (
                    "Minecraft knowledge route coverage is incomplete: "
                    + ", ".join(coverage["blocking_requirement_refs"][:16])
                )
                if domain_detail:
                    message += "; domains: " + domain_detail
                if failures:
                    message += "; research_errors: " + " | ".join(failures)
                raise error(message)
            result["minecraft_knowledge_plan"] = plan
            result["minecraft_knowledge_route_coverage"] = coverage
            result["research_sha256"] = agentic_module._json_sha256(result)
            return result

        setattr(collect, _MARKER, True)
        collect.__wrapped__ = current_collect  # type: ignore[attr-defined]
        agentic_module.collect_pre_design_research = collect

    current_compact = agentic_module._compact_research_for_design
    if not getattr(current_compact, _MARKER, False):

        @wraps(current_compact)
        def compact(research: Mapping[str, Any]):
            value = dict(current_compact(research))
            plan = research.get("minecraft_knowledge_plan")
            if isinstance(plan, Mapping):
                coverage = research.get("minecraft_knowledge_route_coverage")
                value["minecraft_knowledge"] = compact_plan(plan, coverage if isinstance(coverage, Mapping) else None)
            return value

        setattr(compact, _MARKER, True)
        compact.__wrapped__ = current_compact  # type: ignore[attr-defined]
        agentic_module._compact_research_for_design = compact

    if complete_planner_module is not None:
        current_outline = complete_planner_module._implementation_research_outline
        if not getattr(current_outline, _MARKER, False):

            @wraps(current_outline)
            def outline(game_design: dict[str, Any]):
                value = dict(current_outline(game_design))
                pre = game_design.get("_pre_design_research")
                if isinstance(pre, Mapping) and isinstance(pre.get("minecraft_knowledge_plan"), Mapping):
                    coverage = pre.get("minecraft_knowledge_route_coverage")
                    value["minecraft_knowledge"] = compact_plan(
                        pre["minecraft_knowledge_plan"],
                        coverage if isinstance(coverage, Mapping) else None,
                    )
                return value

            setattr(outline, _MARKER, True)
            outline.__wrapped__ = current_outline  # type: ignore[attr-defined]
            complete_planner_module._implementation_research_outline = outline


__all__ = [
    "FEATURE_ROOTS", "NODES", "SCHEMA_VERSION", "COVERAGE_SCHEMA_VERSION",
    "compile_minecraft_knowledge_plan", "detect_features", "evaluate_route_coverage",
    "expand_features", "install", "validate_plan",
]
