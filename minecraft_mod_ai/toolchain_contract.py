from __future__ import annotations

from typing import Any

from .platform_catalog import FABRIC_1201, adapter_for_lock_values


# Backward-compatible exports. Runtime execution no longer uses these globals as the
# source of truth; it resolves the selected project's adapter instead.
LOOM_VERSION = FABRIC_1201.fabric_loom
GRADLE_VERSION = FABRIC_1201.gradle
GRADLE_SHA256 = FABRIC_1201.gradle_sha256
FABRIC_LOADER_VERSION = FABRIC_1201.fabric_loader


def fabric_dependency_predicates(platform: Any) -> dict[str, str]:
    """Return exact dependency predicates from one reviewed platform lock."""

    platform.validate()
    adapter_for_lock_values(platform)
    values = {
        "fabricloader": platform.fabric_loader,
        "minecraft": platform.minecraft_version,
        "java": platform.java_version,
        "fabric-api": platform.fabric_api,
    }
    for dependency_id, value in values.items():
        if (
            type(value) is not str
            or not value.strip()
            or value != value.strip()
            or any(ord(character) < 0x20 for character in value)
        ):
            raise ValueError(
                f"Invalid locked version for Fabric dependency {dependency_id!r}."
            )
    return values


def install(spec_module: Any, runner_module: Any) -> None:
    """Install compatibility aliases without replacing per-plan PlatformLock logic.

    Older MMM builds monkeypatched ``PlatformLock`` here and therefore forced every
    plan to Minecraft 1.20.1. Platform validation is now owned by the adapter catalog.
    Runner globals remain only as legacy import aliases; ``GradleRunner`` resolves the
    actual project's adapter before wrapper/build execution.
    """

    runner_module.GRADLE_VERSION = GRADLE_VERSION
    runner_module.GRADLE_URL = (
        f"https://services.gradle.org/distributions/gradle-{GRADLE_VERSION}-bin.zip"
    )
    runner_module.GRADLE_SHA256 = GRADLE_SHA256
