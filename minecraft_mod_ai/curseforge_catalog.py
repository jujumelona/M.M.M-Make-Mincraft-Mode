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



def curseforge_mods_class_id(categories: Any) -> int | None:
    """Resolve the Minecraft Mods section from live category metadata.

    Never assume a category ID is fixed across games or API revisions.
    """
    if not isinstance(categories, list):
        return None
    for row in categories:
        if not isinstance(row, Mapping) or row.get("gameId") != 432:
            continue
        if row.get("isClass") is not True:
            continue
        label = str(row.get("name") or "").strip().casefold()
        slug = str(row.get("slug") or "").strip().casefold()
        if label != "mods" and slug not in {"mods", "mc-mods"}:
            continue
        class_id = row.get("id")
        if type(class_id) is int and class_id > 0:
            return class_id
    return None


def curseforge_next_index(
    pagination: Any, *, index: int, page_size: int, row_count: int
) -> int | None:
    """Return a valid next offset, or None when the window is exhausted.

    API search windows cannot cross index + pageSize = 10,000.
    """
    if row_count <= 0:
        return None
    server_index, result_count, total_count = index, row_count, None
    if isinstance(pagination, Mapping):
        try:
            server_index = int(pagination.get("index", index))
            result_count = int(pagination.get("resultCount", row_count))
            raw_total = pagination.get("totalCount")
            total_count = int(raw_total) if raw_total is not None else None
        except (ValueError, TypeError, OverflowError):
            raise ValueError("CurseForge returned invalid pagination.") from None
        if server_index != index or result_count < 0 or result_count > page_size:
            raise ValueError("CurseForge returned inconsistent pagination.")
    next_index = server_index + result_count
    if next_index <= index or next_index + page_size > 10000:
        return None
    if total_count is not None and next_index >= total_count:
        return None
    if row_count < page_size and total_count is None:
        return None
    return next_index


def curseforge_browse_windows(
    categories: Any,
    mods_class_id: int | None,
    *,
    broad_pages: int = 2,
    category_pages: int = 1,
) -> tuple[tuple[int, int | None, int], ...]:
    """One authoritative list of catalogue discovery windows for both callers.

    Every returned window specifies (sortField, categoryId, maximum pages).
    These samples cannot establish exhaustive catalogue coverage.
    """
    if broad_pages < 1 or category_pages < 1:
        raise ValueError("Catalogue page budgets must be positive.")
    windows: list[tuple[int, int | None, int]] = [
        (sort_field, None, broad_pages) for sort_field in (2, 3, 11)
    ]
    if mods_class_id is None or not isinstance(categories, list):
        return tuple(windows)
    children: set[int] = set()
    for item in categories:
        if not isinstance(item, Mapping):
            continue
        if item.get("gameId") != 432 or item.get("classId") != mods_class_id:
            continue
        category_id = item.get("id")
        if item.get("isClass") is not True and type(category_id) is int and category_id > 0:
            children.add(category_id)
    windows.extend((3, category_id, category_pages) for category_id in sorted(children))
    return tuple(windows)


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


__all__ = ["curseforge_api_key", "curseforge_mod_metadata", "curseforge_target_filters", "curseforge_mods_class_id", "curseforge_next_index", "curseforge_browse_windows"]
