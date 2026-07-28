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


PLUGIN_STATUSES: tuple[PluginStatus, ...] = (
    PluginStatus("fabric-core", "implemented", "minecraft_mod_ai.generator", ("Fabric 1.20.1 Java project",), ("mmm-local", "minecraft-dev")),
    PluginStatus("fabric-datagen", "implemented", "minecraft_mod_ai.generator", ("recipes", "loot", "tags", "lang", "models"), ("mmm-local",)),
    PluginStatus("entity-basic", "implemented", "minecraft_mod_ai.generator", ("bounded biped boss", "spawn egg", "GameTest skeleton"), ("mmm-local", "minecraft-dev")),
    PluginStatus("worldgen-arena", "implemented", "minecraft_mod_ai.generator", ("arena mcfunction", "WorldDesignIR"), ("mmm-local",)),
    PluginStatus("entity-geckolib", "blocked", "not implemented", (), ("minecraft-dev",)),
    PluginStatus("worldgen-map", "blocked", "IR planning only; no Jigsaw/NBT compiler yet", ("world IR",), ("mmm-local", "minecraft-dev")),
    PluginStatus("quest-system", "blocked", "not implemented", (), ("minecraft-dev",)),
    PluginStatus("gui-networking", "blocked", "not implemented", (), ("minecraft-dev",)),
    PluginStatus("asset-generation", "implemented", "FLUX adapter plus deterministic Minecraft texture post-process", ("concept PNG", "16x16 PNG"), ("mmm-local",)),
    PluginStatus("audio-voice", "partial", "Whisper transcription only; no sound/music synthesis", ("transcript",), ("mmm-local",)),
    PluginStatus("runtime-playtest", "blocked", "no reviewed Minecraft 1.20.1 runtime MCP configured", (), ()),
    PluginStatus("release-security", "implemented", "validator, Gradle runner, JAR inspection and safe ZIP packaging", ("reports", "JAR", "release ZIP"), ("mmm-local",)),
)


def plugin_manifest() -> dict[str, Any]:
    return {
        "schema_version": "mmm/plugin-manifest-v1",
        "plugins": [
            {**asdict(item), "outputs": list(item.outputs), "required_mcp": list(item.required_mcp)}
            for item in PLUGIN_STATUSES
        ],
    }


def buildable_plugin_ids() -> frozenset[str]:
    return frozenset(
        item.plugin_id for item in PLUGIN_STATUSES if item.status == "implemented"
    )
