"""Shared credential, target-filter and provenance support for CurseForge."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# CurseForge's ModLoaderType wire enum, not Minecraft implementation recipes.
# https://docs.curseforge.com/rest-api/#tocS_ModLoaderType
_LOADER_TYPES = {
    "forge": 1,
    "cauldron": 2,
    "liteloader": 3,
    "fabric": 4,
    "quilt": 5,
    "neoforge": 6,
}
_KEY_NAMES = ("MMM_CURSEFORGE_API_KEY", "CURSEFORGE_API_KEY")


def curseforge_api_key() -> str:
    """Resolve the optional host credential without caching or reporting its value."""

    for name in _KEY_NAMES:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    try:
        import google.colab.userdata as colab_userdata  # type: ignore
    except ImportError:
        return ""
    for name in _KEY_NAMES:
        try:
            key = str(colab_userdata.get(name) or "").strip()
        except Exception:  # noqa: BLE001 - optional Colab secrets can deny access.
            key = ""
        if key:
            return key
    return ""


def curseforge_target_filters(
    minecraft_version: str | None = None,
    loader: str | None = None,
) -> dict[str, str]:
    """Send the exact bound target; CurseForge requires version with loader."""

    version = str(minecraft_version or "").strip()
    loader_name = str(loader or "").strip().casefold()
    filters = {"gameVersion": version} if version else {}
    if loader_name:
        if not version:
            raise ValueError("CurseForge loader filtering requires a Minecraft version.")
        if loader_name not in _LOADER_TYPES:
            raise ValueError("The requested loader has no CurseForge ModLoaderType.")
        filters["modLoaderType"] = str(_LOADER_TYPES[loader_name])
    return filters


def curseforge_mod_metadata(
    mod: Mapping[str, Any],
    *,
    minecraft_version: str | None = None,
    loader: str | None = None,
) -> dict[str, Any]:
    """Preserve catalog/source identity and reported versions as discovery evidence."""

    links = mod.get("links") if isinstance(mod.get("links"), Mapping) else {}
    mod_id = mod.get("id")
    raw_indexes = mod.get("latestFilesIndexes")
    version_indexes = [
        {
            "game_version": str(index.get("gameVersion") or ""),
            "file_id": index.get("fileId"),
            "filename": str(index.get("filename") or ""),
            "loader_type": index.get("modLoader"),
        }
        for index in (raw_indexes if isinstance(raw_indexes, list) else [])
        if isinstance(index, Mapping)
    ]
    versions = {index["game_version"] for index in version_indexes if index["game_version"]}
    raw_files = mod.get("latestFiles")
    for file in raw_files if isinstance(raw_files, list) else []:
        if not isinstance(file, Mapping):
            continue
        raw_versions = file.get("gameVersions")
        if isinstance(raw_versions, list):
            versions.update(str(value) for value in raw_versions if str(value).strip())
    return {
        "provider": "curseforge",
        "source_id": f"curseforge:{mod_id}" if mod_id is not None else "",
        "catalog_id": mod_id,
        "mod_id": mod_id,
        "catalog_url": str(links.get("websiteUrl") or ""),
        "source_url": str(links.get("sourceUrl") or ""),
        "slug": str(mod.get("slug") or ""),
        "versions": sorted(versions),
        "version_indexes": version_indexes,
        "requested_minecraft_version": str(minecraft_version or "").strip(),
        "requested_loader": str(loader or "").strip().casefold(),
        "source_reuse_authority": "verification_required",
    }


__all__ = ["curseforge_api_key", "curseforge_mod_metadata", "curseforge_target_filters"]
