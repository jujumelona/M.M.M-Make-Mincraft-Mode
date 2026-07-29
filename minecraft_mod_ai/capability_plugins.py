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
    PluginStatus("fabric-core", "implemented", "minecraft_mod_ai.generator", ("Fabric 1.20.1 Java project",), ("mmm-local", "minecraft-dev")),
    PluginStatus("fabric-datagen", "implemented", "minecraft_mod_ai.generator", ("recipes", "loot", "tags", "lang", "models"), ("mmm-local",)),
    PluginStatus("entity-basic", "implemented", "minecraft_mod_ai.generator", ("bounded biped boss", "spawn egg", "GameTest"), ("mmm-local", "minecraft-dev")),
    PluginStatus("entity-geckolib", "runtime-gated", "minecraft_mod_ai.geckolib_generator", ("geo JSON", "animation JSON", "binding contract", "Gradle snippet"), ("blockbench", "minecraft-dev", "jdtls", "mmm-local"), ("Blockbench UV", "Gradle", "GameTest", "runtime animation review")),
    PluginStatus("worldgen-arena", "implemented", "minecraft_mod_ai.generator", ("arena mcfunction", "WorldDesignIR"), ("mmm-local",)),
    PluginStatus("worldgen-map", "runtime-gated", "minecraft_mod_ai.world_compiler", ("binary NBT", "Jigsaw pools", "processors", "structures", "structure sets", "biome tags", "world resource ZIP"), ("minecraft-dev", "minecraft-runtime-1201", "mmm-local"), ("Gradle", "fresh-world placement", "route reachability", "visual review")),
    PluginStatus("quest-system", "runtime-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "mmm-local"), ("Fabric server binding", "persistence", "GameTest", "runtime")),
    PluginStatus("class-skill-system", "runtime-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "mmm-local"), ("Fabric server binding", "cooldown tests", "GameTest")),
    PluginStatus("economy-shop", "runtime-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "mmm-local"), ("transaction authority", "restart", "concurrency")),
    PluginStatus("gui-networking", "runtime-gated", "minecraft_mod_ai.system_pack_generator", ("packet contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "minecraft-runtime-1201"), ("server authority", "replay tests", "GUI screenshots")),
    PluginStatus("party-guild", "runtime-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-runtime-1201"), ("two-client test", "permissions", "restart")),
    PluginStatus("asset-generation", "implemented", "FLUX adapter plus Minecraft texture post-process", ("concept PNG", "16x16 PNG"), ("mmm-local",), ("VisualCritic", "runtime screenshot")),
    PluginStatus("blockbench-modeling", "configuration-required", "restricted Blockbench MCP client", ("bbmodel", "GeckoLib export", "render preview"), ("blockbench",), ("tool allowlist", "UV validation")),
    PluginStatus("audio-voice", "implemented", "Whisper transcription plus deterministic OGG synthesis and SoundEvent registration", ("transcript", "OGG", "sounds.json", "SoundEvent Java"), ("mmm-local",), ("client playback",)),
    PluginStatus("java-language-analysis", "configuration-required", "Eclipse JDT LS bounded LSP client", ("diagnostics", "workspace symbols"), ("jdtls",)),
    PluginStatus("code-rag", "implemented", "lexical index with optional Qwen3 embedding/reranker", ("versioned code index", "ranked evidence"), ("mmm-local",)),
    PluginStatus("runtime-playtest", "configuration-required", "first-party disposable runtime manager", ("logs", "screenshots", "process receipts"), ("minecraft-runtime-1201",), ("server/client ready", "cleanup")),
    PluginStatus("mineflayer-playtest", "configuration-required", "first-party Mineflayer 1.20.1 JSONL bridge", ("path trace", "inventory", "interaction result"), ("mineflayer-1201",), ("localhost only", "task completion")),
    PluginStatus("fine-tuning", "implemented-data-pipeline", "verified trace store and LLaMA-Factory/TRL configs", ("licensed trace store", "SFT JSONL", "reward"), ("mmm-local",), ("all build/runtime evidence must pass")),
    PluginStatus("release-security", "implemented", "validator, Gradle runner, JAR inspection and safe ZIP packaging", ("reports", "JAR", "release ZIP"), ("mmm-local",)),
    PluginStatus("extended-content", "runtime-gated", "typed Fabric generator for tools, weapons, armor, food, crops, machines, effects, enchantments and commands", ("Java registrations", "models", "recipes", "textures"), ("mmm-local", "minecraft-dev", "jdtls"), ("Gradle", "GameTest", "runtime")),
    PluginStatus("source-patching", "implemented", "transactional SHA-256 guarded text patcher with rollback and finite repair loop", ("patch receipts", "repair evidence"), ("mmm-local", "jdtls"), ("Gradle", "GameTest")),
    PluginStatus("complete-orchestrator", "implemented", "single approved graph orchestrating generation, repair, runtime, visual review and distribution", ("complete proposal", "source", "JAR", "runtime receipts", "release bundle"), ("mmm-local", "minecraft-runtime-1201", "mineflayer-1201")),
    PluginStatus("distribution-publishing", "configuration-required", "Modrinth upload plus reviewed CurseForge endpoint adapter", ("distribution metadata", "publish receipt"), ("mmm-local",), ("validated JAR", "provider token")),
)


def plugin_manifest() -> dict[str, Any]:
    return {
        "schema_version": "mmm/plugin-manifest-v2",
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
        item.plugin_id for item in PLUGIN_STATUSES if item.status == "implemented"
    )
