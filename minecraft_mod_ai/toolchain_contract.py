from __future__ import annotations

from typing import Any


def fabric_dependency_predicates(platform: Any) -> dict[str, str]:
    """Return exact dependency predicates from one host-selected provider receipt."""

    from . import platform_catalog

    platform.validate()
    # Resolve through the live module attribute so late immutable-receipt installation
    # cannot leave this validator pinned to a stale pre-approval alias.
    platform_catalog.adapter_for_lock_values(platform)
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
    """Retain the explicit bootstrap hook without installing target aliases.

    Toolchain values are resolved only from the selected project's executable
    platform-provider receipt. The bootstrap therefore has nothing to copy into the
    runner as process-global defaults.
    """

    del spec_module, runner_module
