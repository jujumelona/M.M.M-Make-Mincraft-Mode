from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any


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

    External Minecraft MCP federation is intentionally internal to reviewed MMM
    planning/generation/quality/runtime tools. Skills therefore do not gain invented
    direct MCP tool names or new authorization surfaces; their existing tool calls
    receive federated evidence automatically at the appropriate stage.
    """

    original = skill_catalog_module._parse_skill
    if getattr(original, "_mmm_dynamic_platform_skill", False):
        return

    @wraps(original)
    def parse_skill(text: str, expected_name: str):
        frontmatter, policy = original(text, expected_name)
        normalized_frontmatter = _normalize(deepcopy(frontmatter))
        normalized_policy = _normalize(deepcopy(policy))

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
