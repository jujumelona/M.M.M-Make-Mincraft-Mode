from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModDevelopmentMethod:
    method_id: str
    purpose: str
    outputs: tuple[str, ...]
    required_evidence: tuple[str, ...]
    release_gates: tuple[str, ...]


BASELINE_METHODS: tuple[ModDevelopmentMethod, ...] = (
    ModDevelopmentMethod(
        "fabric_project_contract",
        "Lock Minecraft, Fabric Loader, Fabric API, Yarn, Loom, Gradle and Java versions before generation.",
        (
            "settings.gradle",
            "build.gradle",
            "gradle.properties",
            "fabric.mod.json",
            "version-lock receipt",
        ),
        (
            "official version metadata",
            "dependency license",
            "immutable artifact hashes",
        ),
        ("Gradle dependency resolution", "Java 17 compile"),
    ),
    ModDevelopmentMethod(
        "client_server_boundary",
        "Separate dedicated-server-safe common code from client rendering, screens, keybinds and model registration.",
        (
            "common initializer",
            "client initializer when requested",
            "environment-safe source layout",
        ),
        ("Fabric environment rules", "dedicated server API evidence"),
        ("dedicated-server classloading check", "client startup when requested"),
    ),
    ModDevelopmentMethod(
        "registry_and_datagen",
        "Register requested content through typed registries and generate data/resources instead of hand-maintaining repeated JSON.",
        (
            "typed registrars",
            "recipes",
            "loot tables",
            "tags",
            "models",
            "blockstates",
            "language files",
        ),
        ("Fabric registry API", "Fabric data generation API"),
        ("resource validation", "registry GameTests"),
    ),
    ModDevelopmentMethod(
        "validation_and_release",
        "Validate source, build outputs and distributable metadata before packaging.",
        (
            "static validation report",
            "Gradle report",
            "GameTest report",
            "JAR inspection",
            "SBOM",
            "provenance receipt",
            "release ZIP",
        ),
        ("requested acceptance criteria", "license and dependency receipts"),
        ("all requested gates",),
    ),
)

OPTIONAL_METHODS: tuple[tuple[tuple[str, ...], ModDevelopmentMethod], ...] = (
    (
        (
            "item",
            "아이템",
            "block",
            "블록",
            "crop",
            "작물",
            "food",
            "요리",
            "tool",
            "도구",
            "weapon",
            "무기",
            "armor",
            "방어구",
            "machine",
            "기계",
        ),
        ModDevelopmentMethod(
            "content_registry",
            "Generate requested items, blocks, recipes, loot, tools, armor, food, crops and machines through typed registry modules.",
            ("Java registry modules", "resource assets", "content GameTests"),
            ("exact Minecraft/Fabric API evidence",),
            ("compile", "registry presence", "recipe/loot validation"),
        ),
    ),
    (
        (
            "event",
            "이벤트",
            "hook",
            "훅",
            "mixin",
            "믹스인",
            "vanilla",
            "바닐라",
            "accessor",
            "access widener",
        ),
        ModDevelopmentMethod(
            "events_mixins_access",
            "Prefer Fabric events; use Mixins or access wideners only where the public API cannot implement the requested behavior.",
            (
                "event handlers",
                "bounded Mixins",
                "mixin config",
                "access widener when necessary",
            ),
            ("target method descriptors", "mapping/version evidence"),
            (
                "Mixin target validation",
                "dedicated server check",
                "behavior GameTest",
            ),
        ),
    ),
    (
        (
            "gui",
            "화면",
            "메뉴",
            "screen",
            "hud",
            "overlay",
            "network",
            "패킷",
            "packet",
        ),
        ModDevelopmentMethod(
            "gui_and_networking",
            "Keep server authority for state changes and validate every client-to-server action.",
            (
                "screen handler",
                "client screen",
                "typed packets",
                "server validation",
                "rate limits",
            ),
            ("Fabric networking API", "threading rules"),
            (
                "packet decode tests",
                "permission tests",
                "replay/rate-limit tests",
            ),
        ),
    ),
    (
        (
            "save",
            "저장",
            "persistent",
            "영속",
            "quest",
            "퀘스트",
            "economy",
            "경제",
            "class",
            "직업",
            "skill",
            "스킬",
            "party",
            "guild",
        ),
        ModDevelopmentMethod(
            "persistent_game_state",
            "Store authoritative state on the server with schema versions, atomic writes and migration handling.",
            (
                "persistent state",
                "schema version",
                "migration code",
                "restart tests",
            ),
            ("Minecraft persistence lifecycle", "serialization format evidence"),
            (
                "restart persistence",
                "corruption fallback",
                "multiplayer authority",
            ),
        ),
    ),
    (
        ("config", "설정", "option", "옵션", "gamerule", "게임룰"),
        ModDevelopmentMethod(
            "configuration",
            "Expose bounded configuration with defaults, validation and server/client ownership rules.",
            ("config schema", "validated loader", "defaults", "migration"),
            ("selected config library compatibility",),
            ("invalid-config fallback", "server ownership test"),
        ),
    ),
    (
        (
            "entity",
            "엔티티",
            "mob",
            "몹",
            "boss",
            "보스",
            "animation",
            "애니메이션",
            "geckolib",
        ),
        ModDevelopmentMethod(
            "entity_rendering_animation",
            "Generate entity logic, attributes, goals, renderer/model bindings and optional GeckoLib animation assets.",
            (
                "entity type",
                "attributes",
                "goals",
                "renderer",
                "model",
                "animation",
                "spawn rules when requested",
            ),
            ("entity API", "renderer API", "GeckoLib compatibility when selected"),
            (
                "dedicated server compile",
                "spawn GameTest",
                "runtime animation review",
            ),
        ),
    ),
    (
        (
            "worldgen",
            "월드젠",
            "biome",
            "바이옴",
            "dimension",
            "차원",
            "structure",
            "구조물",
            "arena",
            "아레나",
            "dungeon",
            "던전",
            "template pool",
            "jigsaw",
            "ore generation",
            "광석 생성",
            "placed feature",
            "configured feature",
        ),
        ModDevelopmentMethod(
            "fabric_worldgen",
            "Generate mod-owned worldgen resources and registration code only when the mod explicitly needs structures, biomes, dimensions or features.",
            (
                "configured/placed features",
                "structure or biome data",
                "dimension resources",
                "bootstrap/registration code",
                "fresh-world tests",
            ),
            ("Fabric/Minecraft worldgen codecs", "datapack schema evidence"),
            (
                "fresh-world generation",
                "upgrade compatibility",
                "no standalone world save",
            ),
        ),
    ),
    (
        (
            "command",
            "명령어",
            "permission",
            "권한",
            "server",
            "서버",
            "multiplayer",
            "멀티플레이",
        ),
        ModDevelopmentMethod(
            "commands_permissions_multiplayer",
            "Implement commands and multiplayer operations with explicit permissions, server authority and concurrency-safe state changes.",
            (
                "Brigadier commands",
                "permission checks",
                "server-side handlers",
                "concurrency tests",
            ),
            ("command API", "server lifecycle evidence"),
            ("permission tests", "two-client tests", "restart tests"),
        ),
    ),
    (
        (
            "existing",
            "기존 모드",
            "수정",
            "patch",
            "포팅",
            "port",
            "upgrade",
            "업데이트",
        ),
        ModDevelopmentMethod(
            "existing_project_patch",
            "Inspect and patch an owned source project transactionally without treating a JAR as editable source.",
            (
                "project index",
                "SHA-256 guarded patch",
                "rollback receipt",
                "migration report",
            ),
            ("source ownership/permission", "current project version graph"),
            ("compile before/after", "regression tests", "rollback test"),
        ),
    ),
)

_STANDALONE_MAP_TERMS = (
    "standalone map",
    "map file",
    "world file",
    "world zip",
    "맵 파일",
    "월드 파일",
    "월드 zip",
    "세이브 파일",
    "schematic",
    "litematic",
    "도시를 그대로",
    "서울시를 그대로",
)


def resolve_mod_development_methods(
    request: str,
    *,
    existing_project: bool = False,
) -> dict[str, Any]:
    if not isinstance(request, str) or not request.strip():
        raise ValueError("request must not be empty")

    folded = request.casefold()
    selected: list[ModDevelopmentMethod] = list(BASELINE_METHODS)
    for triggers, method in OPTIONAL_METHODS:
        if any(trigger.casefold() in folded for trigger in triggers):
            selected.append(method)

    if existing_project and not any(
        method.method_id == "existing_project_patch" for method in selected
    ):
        selected.append(
            next(
                method
                for _, method in OPTIONAL_METHODS
                if method.method_id == "existing_project_patch"
            )
        )

    standalone_map_requested = any(
        term in folded for term in _STANDALONE_MAP_TERMS
    )
    unique = {method.method_id: method for method in selected}
    ordered = [unique[key] for key in sorted(unique)]

    return {
        "schema_version": "mmm/mod-development-methods-v1",
        "scope": "MINECRAFT_FABRIC_MOD_PROJECT",
        "standalone_map_generation": False,
        "standalone_map_requested": standalone_map_requested,
        "scope_note": (
            "Standalone world saves, map ZIPs, schematics and Builder block-delta "
            "handoffs are outside this product. Mod-owned worldgen code and datapack "
            "resources remain available only when explicitly required by the mod."
        ),
        "methods": [asdict(method) for method in ordered],
        "method_ids": [method.method_id for method in ordered],
    }


def mod_development_method_catalog() -> dict[str, Any]:
    methods = {
        method.method_id: method
        for method in (
            *BASELINE_METHODS,
            *(method for _, method in OPTIONAL_METHODS),
        )
    }
    return {
        "schema_version": "mmm/mod-development-method-catalog-v1",
        "scope": "MINECRAFT_FABRIC_MOD_PROJECT",
        "standalone_map_generation": False,
        "methods": [asdict(methods[key]) for key in sorted(methods)],
    }
