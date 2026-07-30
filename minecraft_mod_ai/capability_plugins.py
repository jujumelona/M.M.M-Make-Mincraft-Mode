from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PluginStatus:
    plugin_id: str
    status: str
    implementation: str
    outputs: tuple[str, ...]
    required_mcp: tuple[str, ...]
    release_gates: tuple[str, ...] = ()


PLUGIN_STATUSES: tuple[PluginStatus, ...] = (
    PluginStatus(
        "fabric-core",
        "implemented",
        "minecraft_mod_ai.scalable_generator",
        ("Fabric 1.20.1 Java project", "sharded registrars", "sharded GameTests"),
        ("mmm-local", "minecraft-dev"),
        ("Gradle", "GameTest", "JAR validation"),
    ),
    PluginStatus(
        "fabric-datagen",
        "implemented",
        "minecraft_mod_ai.extended_content_generator",
        ("recipes", "loot", "tags", "lang", "models", "textures"),
        ("mmm-local",),
        ("resource validation", "GameTest"),
    ),
    PluginStatus(
        "entity-basic",
        "implemented",
        "minecraft_mod_ai.generator",
        ("bounded boss", "spawn egg", "loot", "GameTest"),
        ("mmm-local", "minecraft-dev"),
        ("Gradle", "GameTest", "runtime combat review"),
    ),
    PluginStatus(
        "entity-geckolib",
        "implemented",
        "minecraft_mod_ai.geckolib_generator",
        ("entity class", "GeoModel", "GeoRenderer", "geo JSON", "animation JSON", "client/server registration"),
        ("blockbench", "minecraft-dev", "jdtls", "mmm-local"),
        ("Blockbench UV", "Gradle", "GameTest", "runtime animation review"),
    ),
    PluginStatus(
        "worldgen-arena",
        "implemented",
        "minecraft_mod_ai.generator",
        ("arena function", "WorldDesignIR", "navigation proof", "preview"),
        ("mmm-local",),
        ("GameTest", "runtime placement"),
    ),
    PluginStatus(
        "worldgen-map",
        "implemented",
        "minecraft_mod_ai.scalable_world_compiler",
        ("partitioned binary NBT", "Jigsaw pools", "processors", "structure sets", "biome tags", "assembly functions", "world ZIP"),
        ("minecraft-dev", "minecraft-runtime-1201", "mmm-local"),
        ("Gradle", "fresh-world placement", "route reachability", "visual review"),
    ),
    PluginStatus(
        "quest-system",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("kill/break/manual objectives", "server persistence", "rewards", "commands"),
        ("jdtls", "minecraft-dev", "mmm-local"),
        ("Gradle", "GameTest", "restart persistence", "multiplayer authority"),
    ),
    PluginStatus(
        "class-skill-system",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("class selection", "StatusEffect skills", "cooldowns", "persistence"),
        ("jdtls", "minecraft-dev", "mmm-local"),
        ("Gradle", "GameTest", "cooldown runtime tests"),
    ),
    PluginStatus(
        "economy-shop",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("server-owned catalog", "balances", "purchase delivery", "atomic persistence"),
        ("jdtls", "minecraft-dev", "mmm-local"),
        ("Gradle", "restart", "concurrency", "multiplayer authority"),
    ),
    PluginStatus(
        "gui-networking",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("read-only menu", "validated action channel", "cooldown", "one-shot reward persistence"),
        ("jdtls", "minecraft-dev", "minecraft-runtime-1201"),
        ("Gradle", "server authority", "replay tests", "GUI screenshots"),
    ),
    PluginStatus(
        "party-guild",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("create", "invite", "accept", "kick", "leave", "disband", "persistence"),
        ("jdtls", "minecraft-runtime-1201"),
        ("Gradle", "two-client test", "permissions", "restart"),
    ),
    PluginStatus(
        "asset-generation",
        "implemented",
        "image adapter plus policy-bounded Minecraft post-process",
        ("concept PNG", "target-resolution PNG"),
        ("mmm-local",),
        ("VisualCritic", "runtime screenshot"),
    ),
    PluginStatus(
        "blockbench-modeling",
        "implemented",
        "restricted Blockbench MCP client",
        ("bbmodel", "GeckoLib export", "render preview", "UV report"),
        ("blockbench",),
        ("external Blockbench service", "tool allowlist", "UV validation"),
    ),
    PluginStatus(
        "audio-voice",
        "implemented",
        "Whisper transcription plus streaming deterministic OGG synthesis and sharded SoundEvent registration",
        ("transcript", "OGG", "sounds.json", "SoundEvent Java"),
        ("mmm-local",),
        ("Gradle", "client playback", "volume and loop review"),
    ),
    PluginStatus(
        "java-language-analysis",
        "implemented",
        "Eclipse JDT LS bounded LSP client",
        ("diagnostics", "workspace symbols"),
        ("jdtls",),
        ("external JDT LS installation",),
    ),
    PluginStatus(
        "code-rag",
        "implemented",
        "whole-project metadata index with relevance retrieval and optional embeddings/reranker",
        ("project index", "ranked source context", "hashes"),
        ("mmm-local",),
    ),
    PluginStatus(
        "runtime-playtest",
        "implemented",
        "first-party disposable runtime manager",
        ("server/client logs", "screenshots", "process receipts", "cleanup"),
        ("minecraft-runtime-1201",),
        ("external Fabric launcher", "explicit EULA acceptance", "server/client readiness"),
    ),
    PluginStatus(
        "mineflayer-playtest",
        "implemented",
        "first-party Mineflayer 1.20.1 JSONL bridge",
        ("movement", "crafting", "container interaction", "chat", "inventory", "condition assertions"),
        ("mineflayer-1201",),
        ("localhost runtime", "interaction plus wait_for assertion"),
    ),
    PluginStatus(
        "fine-tuning",
        "implemented",
        "verified trace store and LLaMA-Factory/TRL configs",
        ("licensed trace store", "SFT JSONL", "reward"),
        ("mmm-local",),
        ("all build/runtime evidence must pass before trace promotion",),
    ),
    PluginStatus(
        "release-security",
        "implemented",
        "policy-native source validator, Gradle runner, JAR inspection and safe ZIP packaging",
        ("reports", "verified JAR", "release ZIP", "SBOM", "provenance"),
        ("mmm-local",),
        ("all approved gates",),
    ),
    PluginStatus(
        "extended-content",
        "implemented",
        "typed sharded Fabric generator for tools, weapons, armor, food, crops, machines, effects, enchantments and commands",
        ("Java registrations", "models", "recipes", "textures", "registry GameTests"),
        ("mmm-local", "minecraft-dev", "jdtls"),
        ("Gradle", "GameTest", "runtime interaction"),
    ),
    PluginStatus(
        "source-patching",
        "implemented",
        "transactional SHA-256 guarded text patcher with rollback and policy-bounded repair loop",
        ("patch receipts", "repair evidence", "project index"),
        ("mmm-local", "jdtls"),
        ("Gradle", "GameTest"),
    ),
    PluginStatus(
        "complete-orchestrator",
        "implemented",
        "single approved graph orchestrating generation, repair, runtime, visual review and distribution",
        ("complete proposal", "source", "JAR", "runtime receipts", "release bundle"),
        ("mmm-local", "minecraft-runtime-1201", "mineflayer-1201"),
        ("every requested external gate",),
    ),
    PluginStatus(
        "distribution-publishing",
        "implemented",
        "Modrinth upload and reviewed CurseForge endpoint adapter",
        ("distribution metadata", "publish receipt"),
        ("mmm-local",),
        ("validated JAR", "provider project ID", "provider token"),
    ),
)


def plugin_manifest() -> dict[str, Any]:
    return {
        "schema_version": "mmm/plugin-manifest-v3",
        "plugins": [
            {
                **asdict(item),
                "outputs": list(item.outputs),
                "required_mcp": list(item.required_mcp),
                "release_gates": list(item.release_gates),
            }
            for item in PLUGIN_STATUSES
        ],
    }


def buildable_plugin_ids() -> frozenset[str]:
    return frozenset(
        item.plugin_id
        for item in PLUGIN_STATUSES
        if item.status == "implemented"
    )
