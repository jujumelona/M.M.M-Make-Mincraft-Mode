from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any


_MCP_CAPABILITY_STAGES = frozenset(
    {"frontdoor", "planning", "research", "generation", "quality", "runtime"}
)
_MCP_RESEARCH_STAGES = frozenset({"planning", "research", "generation", "quality"})

# Skills whose task can legitimately ask the read-only Minecraft MCP federation for
# target-bound evidence. Automatic planner/coder/repair federation still runs even
# when the Skill does not call these tools explicitly.
_RESEARCH_SKILLS = frozenset(
    {
        "research-minecraft-evidence",
        "gather-adaptive-minecraft-evidence",
        "plan-game-design",
        "inspect-existing-project",
        "generate-fabric-core",
        "generate-datagen",
        "generate-worldgen",
        "generate-geckolib-entity",
        "generate-quest-progression",
        "generate-gui-networking",
        "compile-and-repair",
        "patch-existing-project",
        "execute-complete-production",
        "converge-game-quality",
    }
)
_CAPABILITY_SKILLS = _RESEARCH_SKILLS | frozenset(
    {
        "runtime-playtest",
        "visual-review",
        "freeze-approved-spec",
        "resume-production-run",
    }
)

_REPLACEMENTS = (
    ("Minecraft Java 1.20.1 Fabric", "the approved Minecraft Java Fabric target"),
    ("Minecraft 1.20.1 Fabric", "the approved Minecraft Fabric target"),
    ("Fabric 1.20.1", "the approved Fabric target"),
    ("Java 17", "the PlatformLock Java runtime"),
    ("Yarn 1.20.1+build.1", "the PlatformLock mappings"),
    ("Yarn 1.20.1", "the PlatformLock mappings"),
    ("minecraft-java-1.20.1", "approved-target Minecraft source"),
    ("yarn-javadoc-1.20.1", "approved-target mappings/Javadocs"),
    ("fabric-api-1.20.1", "approved-target Fabric API documentation"),
)


def install(skill_catalog_module: Any) -> None:
    """Compile legacy Skill text into target-dynamic runtime policy.

    The checked-in Skill markdown remains a historical/documentation artifact until
    it is regenerated, but neither source checkout nor packaged wheel may use those
    fixed version phrases as runtime authorization or evidence requirements.
    """

    skill_catalog_module.REVIEWED_TOOL_STAGES["minecraft_mcp_capabilities"] = (
        _MCP_CAPABILITY_STAGES
    )
    skill_catalog_module.REVIEWED_TOOL_STAGES["research_minecraft_mcp"] = (
        _MCP_RESEARCH_STAGES
    )

    original = skill_catalog_module._parse_skill
    if getattr(original, "_mmm_dynamic_platform_skill", False):
        return

    @wraps(original)
    def parse_skill(text: str, expected_name: str):
        frontmatter, policy = original(text, expected_name)
        normalized_frontmatter = _normalize(deepcopy(frontmatter))
        normalized_policy = _normalize(deepcopy(policy))

        allowed = normalized_policy.get("allowed_tools")
        if isinstance(allowed, list):
            values = [str(value) for value in allowed]
            if expected_name in _CAPABILITY_SKILLS and "minecraft_mcp_capabilities" not in values:
                values.append("minecraft_mcp_capabilities")
            if expected_name in _RESEARCH_SKILLS and "research_minecraft_mcp" not in values:
                values.append("research_minecraft_mcp")
            normalized_policy["allowed_tools"] = values

        # The external federation is evidence-only. It never grants mutation or
        # runtime approval and never substitutes its own version for PlatformLock.
        forbidden = normalized_policy.get("forbidden_actions")
        if isinstance(forbidden, list):
            rule = (
                "treating external Minecraft MCP evidence as authorization or as a "
                "replacement for the approved PlatformLock/JDT/Gradle/GameTest gates"
            )
            if rule not in forbidden:
                forbidden.append(rule)

        return normalized_frontmatter, normalized_policy

    parse_skill._mmm_dynamic_platform_skill = True
    skill_catalog_module._parse_skill = parse_skill


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in _REPLACEMENTS:
            result = result.replace(old, new)
        # Any residual standalone historical target is not allowed to become a
        # runtime activation condition. Keep wording readable rather than exposing a
        # fake template token to the model.
        result = result.replace("1.20.1+build.1", "the approved mappings revision")
        result = result.replace("1.20.1", "the approved Minecraft version")
        return result
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value
