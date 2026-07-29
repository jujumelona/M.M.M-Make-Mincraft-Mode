from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import httpx


class PublishingError(RuntimeError):
    pass


def build_distribution_metadata(
    *,
    jar_path: str | Path,
    mod_id: str,
    version: str,
    name: str,
    changelog: str,
    game_versions: tuple[str, ...] = ("1.20.1",),
    loaders: tuple[str, ...] = ("fabric",),
    release_type: str = "release",
) -> dict[str, Any]:
    jar = Path(jar_path).expanduser().resolve()
    if not jar.is_file() or jar.is_symlink() or jar.suffix.lower() != ".jar":
        raise PublishingError("A validated regular JAR is required for publishing.")
    if release_type not in {"release", "beta", "alpha"}:
        raise PublishingError("release_type must be release, beta or alpha.")
    digest = "sha256:" + hashlib.sha256(jar.read_bytes()).hexdigest()
    return {
        "schema_version": "mmm/distribution-metadata-v1",
        "mod_id": mod_id,
        "version": version,
        "name": name,
        "changelog": changelog,
        "game_versions": list(game_versions),
        "loaders": list(loaders),
        "release_type": release_type,
        "jar_path": str(jar),
        "jar_name": jar.name,
        "jar_sha256": digest,
        "jar_size_bytes": jar.stat().st_size,
    }


def publish_modrinth(
    metadata: dict[str, Any],
    *,
    project_id: str,
    token_env: str = "MODRINTH_TOKEN",
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise PublishingError(f"{token_env} is required for Modrinth publishing.")
    jar = _validated_jar(metadata)
    file_part = "primary"
    data = {
        "name": f"{metadata['name']} {metadata['version']}",
        "version_number": metadata["version"],
        "changelog": metadata["changelog"],
        "dependencies": [],
        "game_versions": metadata["game_versions"],
        "version_type": metadata["release_type"],
        "loaders": metadata["loaders"],
        "featured": False,
        "project_id": project_id,
        "file_parts": [file_part],
        "primary_file": file_part,
        "status": "listed",
        "requested_status": "listed",
    }
    headers = {
        "Authorization": token,
        "User-Agent": "jujumelona/M.M.M-Make-Mincraft-Mode",
    }
    with jar.open("rb") as stream, httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            "https://api.modrinth.com/v2/version",
            headers=headers,
            files={
                "data": (None, json.dumps(data), "application/json"),
                file_part: (jar.name, stream, "application/java-archive"),
            },
        )
    if response.status_code not in {200, 201}:
        raise PublishingError(
            f"Modrinth upload failed with HTTP {response.status_code}: {response.text[:1000]}"
        )
    result = response.json()
    return {
        "schema_version": "mmm/publish-receipt-v1",
        "provider": "modrinth",
        "status": "PUBLISHED",
        "version_id": result.get("id") if isinstance(result, dict) else None,
        "jar_sha256": metadata["jar_sha256"],
        "response": result,
    }


def publish_curseforge(
    metadata: dict[str, Any],
    *,
    project_id: str,
    upload_url_env: str = "MMM_CURSEFORGE_UPLOAD_URL",
    token_env: str = "CURSEFORGE_TOKEN",
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    token = os.environ.get(token_env, "").strip()
    upload_url = os.environ.get(upload_url_env, "").strip()
    if not token:
        raise PublishingError(f"{token_env} is required for CurseForge publishing.")
    if not upload_url:
        raise PublishingError(
            f"{upload_url_env} must contain the reviewed CurseForge project upload endpoint."
        )
    if not upload_url.startswith("https://") or project_id not in upload_url:
        raise PublishingError("CurseForge upload URL must be HTTPS and include the project ID.")
    jar = _validated_jar(metadata)
    game_version_ids = os.environ.get("MMM_CURSEFORGE_GAME_VERSION_IDS", "").split(",")
    game_version_ids = [int(value) for value in game_version_ids if value.strip().isdigit()]
    if not game_version_ids:
        raise PublishingError("MMM_CURSEFORGE_GAME_VERSION_IDS must list reviewed numeric IDs.")
    curse_metadata = {
        "changelog": metadata["changelog"],
        "changelogType": "markdown",
        "displayName": f"{metadata['name']} {metadata['version']}",
        "gameVersions": game_version_ids,
        "releaseType": metadata["release_type"],
    }
    with jar.open("rb") as stream, httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            upload_url,
            headers={"X-Api-Token": token, "User-Agent": "M.M.M-Make-Mincraft-Mode"},
            files={
                "metadata": (None, json.dumps(curse_metadata), "application/json"),
                "file": (jar.name, stream, "application/java-archive"),
            },
        )
    if response.status_code not in {200, 201}:
        raise PublishingError(
            f"CurseForge upload failed with HTTP {response.status_code}: {response.text[:1000]}"
        )
    result = response.json()
    return {
        "schema_version": "mmm/publish-receipt-v1",
        "provider": "curseforge",
        "status": "PUBLISHED",
        "jar_sha256": metadata["jar_sha256"],
        "response": result,
    }


def package_distribution_bundle(
    metadata: dict[str, Any],
    *,
    output_zip: str | Path,
    source_zip: str | Path | None = None,
) -> dict[str, Any]:
    jar = _validated_jar(metadata)
    target = Path(output_zip).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(jar, f"binary/{jar.name}")
        archive.writestr(
            "distribution-metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if source_zip is not None:
            source = Path(source_zip).expanduser().resolve()
            if not source.is_file() or source.is_symlink():
                raise PublishingError("source_zip must be a regular file.")
            archive.write(source, f"source/{source.name}")
    return {
        "schema_version": "mmm/distribution-bundle-v1",
        "status": "PACKAGED",
        "path": str(target),
        "sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def _validated_jar(metadata: dict[str, Any]) -> Path:
    if metadata.get("schema_version") != "mmm/distribution-metadata-v1":
        raise PublishingError("Unsupported distribution metadata schema.")
    jar = Path(str(metadata["jar_path"])).expanduser().resolve()
    if not jar.is_file() or jar.is_symlink():
        raise PublishingError("Distribution JAR is missing.")
    digest = "sha256:" + hashlib.sha256(jar.read_bytes()).hexdigest()
    if digest != metadata.get("jar_sha256"):
        raise PublishingError("Distribution JAR changed after metadata was created.")
    return jar
