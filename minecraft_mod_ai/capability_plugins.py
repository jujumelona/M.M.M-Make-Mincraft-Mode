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
    PluginStatus("entity-geckolib", "build-gated", "minecraft_mod_ai.geckolib_generator", ("geo JSON", "animation JSON", "binding contract", "Gradle snippet"), ("blockbench", "minecraft-dev", "jdtls", "mmm-local"), ("Blockbench UV", "Gradle", "GameTest", "runtime animation review")),
    PluginStatus("worldgen-arena", "implemented", "minecraft_mod_ai.generator", ("arena mcfunction", "WorldDesignIR"), ("mmm-local",)),
    PluginStatus("worldgen-map", "runtime-gated", "minecraft_mod_ai.world_compiler", ("binary NBT", "Jigsaw pools", "processors", "structures", "structure sets", "biome tags", "world resource ZIP"), ("minecraft-dev", "minecraft-runtime-1201", "mmm-local"), ("Gradle", "fresh-world placement", "route reachability", "visual review")),
    PluginStatus("quest-system", "binding-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "mmm-local"), ("Fabric server binding", "persistence", "GameTest", "runtime")),
    PluginStatus("class-skill-system", "binding-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "mmm-local"), ("Fabric server binding", "cooldown tests", "GameTest")),
    PluginStatus("economy-shop", "binding-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "mmm-local"), ("transaction authority", "restart", "concurrency")),
    PluginStatus("gui-networking", "binding-gated", "minecraft_mod_ai.system_pack_generator", ("packet contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-dev", "minecraft-runtime-1201"), ("server authority", "replay tests", "GUI screenshots")),
    PluginStatus("party-guild", "binding-gated", "minecraft_mod_ai.system_pack_generator", ("Java contract", "validator", "versioned data JSON"), ("jdtls", "minecraft-runtime-1201"), ("two-client test", "permissions", "restart")),
    PluginStatus("asset-generation", "implemented", "FLUX adapter plus Minecraft texture post-process", ("concept PNG", "16x16 PNG"), ("mmm-local",), ("VisualCritic", "runtime screenshot")),
    PluginStatus("blockbench-modeling", "configuration-required", "restricted Blockbench MCP client", ("bbmodel", "GeckoLib export", "render preview"), ("blockbench",), ("tool allowlist", "UV validation")),
    PluginStatus("audio-voice", "partial", "Whisper transcription only; synthesis remains unconfigured", ("transcript",), ("mmm-local",)),
    PluginStatus("java-language-analysis", "configuration-required", "Eclipse JDT LS bounded LSP client", ("diagnostics", "workspace symbols"), ("jdtls",)),
    PluginStatus("code-rag", "implemented", "lexical index with optional Qwen3 embedding/reranker", ("versioned code index", "ranked evidence"), ("mmm-local",)),
    PluginStatus("runtime-playtest", "configuration-required", "first-party disposable runtime manager", ("logs", "screenshots", "process receipts"), ("minecraft-runtime-1201",), ("server/client ready", "cleanup")),
    PluginStatus("mineflayer-playtest", "configuration-required", "first-party Mineflayer 1.20.1 JSONL bridge", ("path trace", "inventory", "interaction result"), ("mineflayer-1201",), ("localhost only", "task completion")),
    PluginStatus("fine-tuning", "implemented-data-pipeline", "verified trace store and LLaMA-Factory/TRL configs", ("licensed trace store", "SFT JSONL", "reward"), ("mmm-local",), ("all build/runtime evidence must pass")),
    PluginStatus("release-security", "implemented", "validator, Gradle runner, JAR inspection and safe ZIP packaging", ("reports", "JAR", "release ZIP"), ("mmm-local",)),
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
