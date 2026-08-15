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
        "central-evidence-broker",
        "implemented",
        "minecraft_mod_ai.central_research plus ecosystem_discovery",
        (
            "request-derived research graph",
            "adaptive exact-version RAG",
            "paginated dependency and licensed-media candidates",
            "Modrinth version and GitHub commit/license inspections",
            "coverage and provenance receipts",
        ),
        ("mmm-research",),
        (
            "exact compatibility inspection",
            "origin license verification",
            "immutable artifact hash before reuse",
        ),
    ),
    PluginStatus(
        "mod-development-methods",
        "implemented",
        "minecraft_mod_ai.mod_development_methods",
        (
            "request-specific implementation method plan",
            "client/server ownership decisions",
            "required evidence and release gates",
            "explicit standalone-map exclusion",
        ),
        ("mmm-frontdoor", "mmm-research", "mmm-generation"),
        ("method plan resolved before generation",),
    ),
    PluginStatus(
        "fabric-core",
        "implemented",
        "minecraft_mod_ai.scalable_generator",
        (
            "approved-target Minecraft Java project",
            "version-locked Gradle contract",
            "common/client initializer split",
            "sharded registrars",
            "sharded GameTests",
        ),
        ("mmm-generation", "mmm-quality", "minecraft-dev"),
        (
            "Gradle",
            "dedicated-server classloading",
            "GameTest",
            "JAR validation",
        ),
    ),
    PluginStatus(
        "fabric-datagen",
        "implemented",
        "minecraft_mod_ai.extended_content_generator",
        (
            "recipes",
            "loot",
            "tags",
            "lang",
            "models",
            "blockstates",
            "textures",
        ),
        ("mmm-generation",),
        ("resource validation", "GameTest"),
    ),
    PluginStatus(
        "fabric-events-mixins",
        "implemented",
        "event-first generation with bounded Mixins and access wideners",
        (
            "Fabric event handlers",
            "Mixin classes only when public APIs are insufficient",
            "mixin configuration",
            "access widener when required",
        ),
        ("mmm-generation", "mmm-quality", "minecraft-dev", "jdtls"),
        (
            "mapping-pinned target validation",
            "dedicated-server check",
            "behavior GameTest",
        ),
    ),
    PluginStatus(
        "fabric-networking-persistence",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        (
            "typed packets",
            "server-authoritative handlers",
            "rate limits",
            "persistent state",
            "schema migrations",
            "restart tests",
        ),
        ("mmm-generation", "mmm-quality", "mmm-runtime"),
        (
            "packet validation",
            "permission tests",
            "restart persistence",
            "multiplayer authority",
        ),
    ),
    PluginStatus(
        "fabric-config",
        "implemented",
        "validated configuration generation selected from compatible libraries or code-owned fallback",
        (
            "configuration schema",
            "defaults",
            "validation",
            "migration",
            "server/client ownership rules",
        ),
        ("mmm-research", "mmm-generation", "mmm-quality"),
        ("invalid-config fallback", "server ownership test"),
    ),
    PluginStatus(
        "fabric-worldgen",
        "implemented",
        "mod-owned datapack and registration generation; never a standalone map or world save",
        (
            "configured and placed features",
            "structures when requested",
            "biomes when requested",
            "dimensions when requested",
            "bootstrap/registration code",
        ),
        ("mmm-research", "mmm-generation", "mmm-quality", "minecraft-dev"),
        (
            "fresh-world generation",
            "datapack schema validation",
            "upgrade compatibility",
            "no world ZIP or schematic artifact",
        ),
    ),
    PluginStatus(
        "entity-basic",
        "implemented",
        "minecraft_mod_ai.generator",
        ("requested entity", "optional spawn egg", "loot", "GameTest"),
        ("mmm-generation", "mmm-quality", "minecraft-dev"),
        ("Gradle", "GameTest", "runtime combat review"),
    ),
    PluginStatus(
        "entity-geckolib",
        "implemented",
        "minecraft_mod_ai.geckolib_generator",
        (
            "entity class",
            "GeoModel",
            "GeoRenderer",
            "geo JSON",
            "animation JSON",
            "client/server registration",
        ),
        ("blockbench", "minecraft-dev", "jdtls", "mmm-generation", "mmm-quality"),
        ("Blockbench UV", "Gradle", "GameTest", "runtime animation review"),
    ),
    PluginStatus(
        "quest-system",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("kill/break/manual objectives", "server persistence", "rewards", "commands"),
        ("jdtls", "minecraft-dev", "mmm-generation", "mmm-quality"),
        ("Gradle", "GameTest", "restart persistence", "multiplayer authority"),
    ),
    PluginStatus(
        "class-skill-system",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("class selection", "StatusEffect skills", "cooldowns", "persistence"),
        ("jdtls", "minecraft-dev", "mmm-generation", "mmm-quality"),
        ("Gradle", "GameTest", "cooldown runtime tests"),
    ),
    PluginStatus(
        "economy-shop",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("server-owned catalog", "balances", "purchase delivery", "atomic persistence"),
        ("jdtls", "minecraft-dev", "mmm-generation", "mmm-quality"),
        ("Gradle", "restart", "concurrency", "multiplayer authority"),
    ),
    PluginStatus(
        "gui-networking",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("screen handler", "client screen", "validated action channel", "cooldown", "one-shot reward persistence"),
        ("jdtls", "minecraft-dev", "mmm-runtime"),
        ("Gradle", "server authority", "replay tests", "GUI screenshots"),
    ),
    PluginStatus(
        "party-guild",
        "implemented",
        "minecraft_mod_ai.system_pack_generator",
        ("create", "invite", "accept", "kick", "leave", "disband", "persistence"),
        ("jdtls", "mmm-runtime"),
        ("Gradle", "two-client test", "permissions", "restart"),
    ),
    PluginStatus(
        "asset-generation",
        "implemented",
        "image adapter plus policy-bounded Minecraft post-process",
        ("concept PNG", "target-resolution PNG"),
        ("mmm-generation", "mmm-quality"),
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
        "speech-input-transcription",
        "implemented",
        "speech-model transcription adapter for planning input and smoke tests; not an in-game microphone or TTS feature",
        ("transcript", "language metadata"),
        ("mmm-research", "mmm-training"),
        ("audio consent", "model revision and license", "language and latency test"),
    ),
    PluginStatus(
        "procedural-audio",
        "implemented",
        "deterministic OGG synthesis and sharded SoundEvent registration; not speech synthesis or voice cloning",
        ("OGG", "sounds.json", "SoundEvent Java"),
        ("mmm-generation",),
        ("Gradle", "client playback", "volume and loop review"),
    ),
    PluginStatus(
        "local-ai-voice-sidecar",
        "implemented",
        "reviewed Java 17 localhost-only asynchronous HTTP boundary; it does not bundle an AI, ASR or TTS model and cannot mutate world state",
        (
            "typed JSON request and response utility",
            "exact-reconstruction policy manifest",
            "bounded timeout, bytes and in-flight concurrency",
        ),
        ("mmm-research", "mmm-generation", "mmm-quality"),
        (
            "approved capability allowlist",
            "exact source and manifest validation",
            "separately installed compatible localhost sidecar",
            "runtime latency and failure fallback test",
            "voice consent when adaptation or conversion is requested",
        ),
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
        ("mmm-research",),
    ),
    PluginStatus(
        "runtime-playtest",
        "implemented",
        "first-party disposable runtime manager",
        ("server/client logs", "screenshots", "process receipts", "cleanup"),
        ("mmm-runtime",),
        ("external Fabric launcher", "explicit EULA acceptance", "server/client readiness"),
    ),
    PluginStatus(
        "mineflayer-playtest",
        "implemented",
        "first-party target-aware Mineflayer JSONL bridge",
        ("movement", "crafting", "container interaction", "chat", "inventory", "condition assertions"),
        ("mmm-runtime",),
        ("localhost runtime", "interaction plus wait_for assertion"),
    ),
    PluginStatus(
        "fine-tuning",
        "implemented",
        "verified trace store and LLaMA-Factory/TRL configs",
        ("licensed trace store", "SFT JSONL", "reward"),
        ("mmm-training",),
        ("all build/runtime evidence must pass before trace promotion",),
    ),
    PluginStatus(
        "release-security",
        "implemented",
        "policy-native source validator, Gradle runner, JAR inspection and safe ZIP packaging",
        ("reports", "verified JAR", "release ZIP", "SBOM", "provenance"),
        ("mmm-quality", "mmm-release"),
        ("all approved gates",),
    ),
    PluginStatus(
        "extended-content",
        "implemented",
        "typed sharded Fabric generator for tools, weapons, armor, food, crops, machines, effects, enchantments and commands",
        ("Java registrations", "models", "recipes", "textures", "registry GameTests"),
        ("mmm-generation", "mmm-quality", "minecraft-dev", "jdtls"),
        ("Gradle", "GameTest", "runtime interaction"),
    ),
    PluginStatus(
        "source-patching",
        "implemented",
        "transactional SHA-256 guarded text patcher with rollback and policy-bounded repair loop",
        ("patch receipts", "repair evidence", "project index"),
        ("mmm-generation", "mmm-quality", "jdtls"),
        ("Gradle", "GameTest"),
    ),
    PluginStatus(
        "complete-orchestrator",
        "implemented",
        "single approved graph orchestrating mod generation, repair, runtime, visual review and distribution",
        ("complete proposal", "source", "JAR", "runtime receipts", "release bundle"),
        ("mmm-generation", "mmm-quality", "mmm-runtime", "mmm-release"),
        ("every requested external gate",),
    ),
    PluginStatus(
        "quality-convergence",
        "implemented",
        "request-derived production contract plus independent evidence evaluator",
        (
            "requirement-to-implementation coverage",
            "conditional quality dimensions",
            "fresh evidence report",
            "plateau detection",
        ),
        ("mmm-frontdoor", "mmm-quality", "mmm-runtime", "mmm-release"),
        (
            "proposal and artifact binding",
            "independent verifier",
            "all relevant dimensions pass",
        ),
    ),
    PluginStatus(
        "distribution-publishing",
        "implemented",
        "Modrinth upload and reviewed CurseForge endpoint adapter",
        ("distribution metadata", "publish receipt"),
        ("mmm-release",),
        ("validated JAR", "provider project ID", "provider token"),
    ),
)


def plugin_manifest() -> dict[str, Any]:
    return {
        "schema_version": "mmm/plugin-manifest-v4",
        "product_scope": "Minecraft Fabric mod projects",
        "standalone_map_generation": False,
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
