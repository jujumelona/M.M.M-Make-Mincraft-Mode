"""Verify a CurseForge-hosted Fabric JAR before it can be a dependency.

Discovery metadata alone NEVER grants dependency or source-transplant
authority. The host must obtain an exact CurseForge file response (authenticated
API), download permitted bytes, check their digest and inspect fabric.mod.json.
This does not authorize redistribution of the JAR in the final mod ZIP.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .curseforge_catalog import curseforge_target_filters


class CurseForgeDependencyError(ValueError):
    pass


def _matched_game_version(manifest_value: Any, version: str) -> bool:
    # Only precise rules are proven here; wildcard/ranges need a real semver
    # resolver and cannot be treated as exact compatibility evidence.
    if manifest_value == "*" or manifest_value == version:
        return True
    if isinstance(manifest_value, list):
        return version in manifest_value or "*" in manifest_value
    return False


def verify_curseforge_dependency(
    jar_path: str | Path,
    *,
    project_id: int,
    file_id: int,
    file_metadata: Mapping[str, Any],
    minecraft_version: str,
    loader: str,
    license_policy_confirmed: bool,
) -> dict[str, Any]:
    """Verify authentic CurseForge metadata already fetched by the host.

    This verifies file integrity and its declared target but does not install
    the dependency, exercise gameplay, or certify redistribution rights.
    """
    curseforge_target_filters(minecraft_version, loader)
    if loader.casefold() != "fabric":
        raise CurseForgeDependencyError("DEPENDENCY_LOADER_UNSUPPORTED")
    if (type(project_id) is not int or project_id <= 0 or
            type(file_id) is not int or file_id <= 0):
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_ID_INVALID")
    if not license_policy_confirmed:
        raise CurseForgeDependencyError("DEPENDENCY_LICENSE_POLICY_UNCONFIRMED")
    if not isinstance(file_metadata, Mapping):
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_METADATA_MISSING")
    if (file_metadata.get("id") != file_id or
            file_metadata.get("modId") != project_id or
            file_metadata.get("isAvailable") is not True):
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_FILE_MISMATCH")
    filename = str(file_metadata.get("fileName") or "")
    if not filename.casefold().endswith(".jar"):
        raise CurseForgeDependencyError("DEPENDENCY_NOT_JAR")
    game_versions = [str(x) for x in file_metadata.get("gameVersions", ())
                     if isinstance(x, str)]
    if minecraft_version not in game_versions or not any(
            value.casefold() == "fabric" for value in game_versions):
        raise CurseForgeDependencyError("DEPENDENCY_VERSION_LOADER_MISMATCH")
    sha1s = [
        str(record.get("value") or "").casefold()
        for record in file_metadata.get("hashes", ())
        if isinstance(record, Mapping) and record.get("algo") == 1
    ]
    if len(sha1s) != 1 or len(sha1s[0]) != 40:
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_HASH_MISSING")
    path = Path(jar_path).expanduser().resolve()
    if not path.is_file() or path.suffix.casefold() != ".jar":
        raise CurseForgeDependencyError("DEPENDENCY_LOCAL_JAR_MISSING")
    data = path.read_bytes()
    if len(data) > 128 * 1024 * 1024:
        raise CurseForgeDependencyError("DEPENDENCY_JAR_TOO_LARGE")
    if hashlib.sha1(data, usedforsecurity=False).hexdigest() != sha1s[0]:
        raise CurseForgeDependencyError("DEPENDENCY_HASH_MISMATCH")
    try:
        with ZipFile(path) as archive:
            if "fabric.mod.json" not in archive.namelist():
                raise CurseForgeDependencyError("DEPENDENCY_NOT_FABRIC_MOD")
            info = archive.getinfo("fabric.mod.json")
            if info.file_size > 512 * 1024:
                raise CurseForgeDependencyError("DEPENDENCY_MANIFEST_TOO_LARGE")
            manifest = json.loads(archive.read(info))
    except (BadZipFile, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise CurseForgeDependencyError("DEPENDENCY_INVALID_JAR") from exc
    if not isinstance(manifest, Mapping) or not manifest.get("id"):
        raise CurseForgeDependencyError("DEPENDENCY_MANIFEST_MISSING_ID")
    dep = manifest.get("depends")
    if isinstance(dep, Mapping) and "minecraft" in dep and not _matched_game_version(
            dep["minecraft"], minecraft_version):
        raise CurseForgeDependencyError("DEPENDENCY_MANIFEST_VERSION_UNPROVED")
    return {
        "schema_version": "mmm/curseforge-dependency-proof-v1",
        "mode": "dependency", "source": "curseforge_authenticated_file",
        "project_id": project_id, "file_id": file_id,
        "jar_path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "provider_sha1": sha1s[0],
        "fabric_mod_id": str(manifest["id"]),
        "minecraft_version": minecraft_version, "loader": "fabric",
        "local_integrity_passed": True,
        "license_policy_confirmed": True,
        "runtime_verified": False, "source_reuse": False,
        "redistribution_authorized": False,
    }


def fetch_exact_curseforge_file_metadata(
    project_id: int, file_id: int, *, client: Any = None,
) -> dict[str, Any]:
    """Read an explicit file identity from authenticated CurseForge REST.

    This is NOT catalogue search. Neither a model nor a page URL may supply an
    unchecked download address. Network and credentials stay host-owned.
    """
    import httpx
    from .curseforge_catalog import curseforge_api_key

    if type(project_id) is not int or type(file_id) is not int or min(project_id, file_id) <= 0:
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_ID_INVALID")
    key = curseforge_api_key()
    if not key:
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_CREDENTIAL_MISSING")
    own_client = client is None
    session = client if client is not None else httpx.Client(timeout=10.0)
    try:
        response = session.get(
            f"https://api.curseforge.com/v1/mods/{project_id}/files/{file_id}",
            headers={"x-api-key": key, "Accept": "application/json"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CurseForgeDependencyError(
            "DEPENDENCY_PROVIDER_FETCH_FAILED: " + type(exc).__name__
        ) from None
    finally:
        if own_client:
            session.close()
    if not isinstance(data, Mapping) or not isinstance(data.get("data"), Mapping):
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_RESPONSE_INVALID")
    result = dict(data["data"])
    if result.get("id") != file_id or result.get("modId") != project_id:
        raise CurseForgeDependencyError("DEPENDENCY_PROVIDER_FILE_MISMATCH")
    return result
