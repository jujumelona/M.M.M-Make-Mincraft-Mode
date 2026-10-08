from __future__ import annotations

from .system_source_template import render_system_source


def _persistent_store_java(
    package_name: str, mod_id: str, *, minecraft_version: str = ""
) -> str:
    source = render_system_source(
        "persistent_store", package_name=package_name, mod_id=mod_id
    )
    if str(minecraft_version).strip().startswith("26."):
        # Changed 2026-10-08: Mojang 26.1 replaced Yarn WorldSavePath.
        return source.replace(
            "import net.minecraft.util.WorldSavePath;",
            "import net.minecraft.world.level.storage.LevelResource;",
        ).replace(
            "server.getSavePath(WorldSavePath.ROOT)",
            "server.getWorldPath(LevelResource.ROOT)",
        )
    return source


def _config_loader_java(package_name: str, mod_id: str) -> str:
    return render_system_source(
        "config_loader", package_name=package_name, mod_id=mod_id
    )


__all__ = ["_config_loader_java", "_persistent_store_java"]
