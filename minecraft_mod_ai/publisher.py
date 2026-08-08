from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from .toolchain_contract import fabric_dependency_predicates


class PublishingError(RuntimeError):
    pass


_FABRIC_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_MODRINTH_PROJECT_ID = re.compile(r"^[A-Za-z0-9]{3,64}$")
_MAX_FABRIC_METADATA_BYTES = 1_048_576
_DEPENDENCY_RELATIONSHIPS = {
    "depends": "required",
    "recommends": "optional",
    "suggests": "optional",
    "conflicts": "incompatible",
    "breaks": "incompatible",
}
_PLATFORM_DEPENDENCIES = frozenset({"minecraft", "java", "fabricloader"})

# Stable project identifiers verified from the providers' published project
# pages. Unknown custom mods are never guessed from their Fabric mod ID.
_KNOWN_MODRINTH_PROJECTS = {
    "fabric-api": "P7dR8mSH",
    "geckolib": "8BmcQJ2H",
}


def read_fabric_metadata_file(path: str | Path) -> dict[str, Any]:
    """Read bounded source-tree Fabric metadata without accepting symlinks."""

    requested_path = Path(path).expanduser()
    if requested_path.is_symlink():
        raise PublishingError("fabric.mod.json must be a regular file.")
    metadata_path = requested_path.resolve()
    if not metadata_path.is_file():
        raise PublishingError("fabric.mod.json must be a regular file.")
    if metadata_path.stat().st_size > _MAX_FABRIC_METADATA_BYTES:
        raise PublishingError("fabric.mod.json exceeds the metadata size limit.")
    try:
        raw = metadata_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishingError(f"Invalid fabric.mod.json: {exc}") from exc
    return _validated_fabric_metadata_object(value)


def dependency_inventory_from_metadata(
    fabric_metadata: Mapping[str, Any],
    *,
    platform_lock: Any,
    modrinth_project_ids: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize every declared Fabric relationship and bind the tested lock."""

    metadata = _validated_fabric_metadata_object(fabric_metadata)
    expected = fabric_dependency_predicates(platform_lock)
    dependencies = _normalize_fabric_dependencies(metadata)
    required = {
        item["mod_id"]: item["version_predicates"]
        for item in dependencies
        if item["fabric_section"] == "depends"
    }
    for dependency_id, predicate in expected.items():
        if required.get(dependency_id) != [predicate]:
            raise PublishingError(
                f"fabric.mod.json must bind {dependency_id!r} to the tested "
                f"predicate {predicate!r}."
            )

    project_ids = _validated_modrinth_project_ids(
        modrinth_project_ids,
        declared_ids={item["mod_id"] for item in dependencies},
    )
    resolved = {**_KNOWN_MODRINTH_PROJECTS, **project_ids}
    for item in dependencies:
        item["platform_dependency"] = item["mod_id"] in _PLATFORM_DEPENDENCIES
        item["modrinth_project_id"] = resolved.get(item["mod_id"])
    return dependencies


def fabric_dependency_components(
    dependencies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert normalized Fabric dependencies into CycloneDX components."""

    components: list[dict[str, Any]] = []
    for dependency in dependencies:
        dependency_id = str(dependency["mod_id"])
        predicates = [str(value) for value in dependency["version_predicates"]]
        relationship = str(dependency["relationship"])
        component: dict[str, Any] = {
            "type": "framework" if dependency_id in _PLATFORM_DEPENDENCIES else "library",
            "name": _dependency_display_name(dependency_id),
            "bom-ref": f"fabric:{dependency_id}:{relationship}",
            "scope": (
                "required"
                if relationship == "required"
                else "optional"
                if relationship == "optional"
                else "excluded"
            ),
            "properties": [
                {"name": "fabric:mod-id", "value": dependency_id},
                {
                    "name": "fabric:relationship",
                    "value": str(dependency["fabric_section"]),
                },
                {
                    "name": "fabric:version-predicates",
                    "value": json.dumps(predicates, separators=(",", ":")),
                },
            ],
        }
        if len(predicates) == 1 and _is_exact_predicate(predicates[0]):
            component["version"] = predicates[0]
            purl = _dependency_purl(dependency_id, predicates[0])
            if purl is not None:
                component["purl"] = purl
        components.append(component)
    return components


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
    platform_lock: Any | None = None,
    modrinth_project_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    requested_jar = Path(jar_path).expanduser()
    if requested_jar.is_symlink():
        raise PublishingError("A validated regular JAR is required for publishing.")
    jar = requested_jar.resolve()
    if not jar.is_file() or jar.suffix.lower() != ".jar":
        raise PublishingError("A validated regular JAR is required for publishing.")
    if release_type not in {"release", "beta", "alpha"}:
        raise PublishingError("release_type must be release, beta or alpha.")
    if platform_lock is None:
        from .spec import PlatformLock

        platform_lock = PlatformLock()
    platform_lock.validate()
    expected_games = (platform_lock.minecraft_version,)
    expected_loaders = (platform_lock.loader,)
    if game_versions != expected_games:
        raise PublishingError(
            "Distribution game_versions must exactly match the tested PlatformLock."
        )
    if loaders != expected_loaders:
        raise PublishingError(
            "Distribution loaders must exactly match the tested PlatformLock."
        )

    fabric_metadata, fabric_metadata_sha256 = _read_fabric_metadata_from_jar(jar)
    if fabric_metadata.get("id") != mod_id:
        raise PublishingError("Distribution mod_id does not match the JAR metadata.")
    if fabric_metadata.get("version") != version:
        raise PublishingError("Distribution version does not match the JAR metadata.")
    dependencies = dependency_inventory_from_metadata(
        fabric_metadata,
        platform_lock=platform_lock,
        modrinth_project_ids=modrinth_project_ids,
    )
    project_ids = {
        key: value
        for key, value in _validated_modrinth_project_ids(
            modrinth_project_ids,
            declared_ids={item["mod_id"] for item in dependencies},
        ).items()
    }
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
        "environment": fabric_metadata["environment"],
        "platform_lock": asdict(platform_lock),
        "fabric_metadata_sha256": fabric_metadata_sha256,
        "fabric_dependencies": dependencies,
        "dependency_project_ids": project_ids,
        "modrinth_dependencies": _modrinth_dependencies(
            dependencies,
            require_resolved=False,
        ),
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
    jar = _validated_jar(metadata)
    modrinth_dependencies = _modrinth_dependencies(
        metadata["fabric_dependencies"],
        require_resolved=True,
    )
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise PublishingError(f"{token_env} is required for Modrinth publishing.")
    file_part = "primary"
    data = {
        "name": f"{metadata['name']} {metadata['version']}",
        "version_number": metadata["version"],
        "changelog": metadata["changelog"],
        "dependencies": modrinth_dependencies,
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
    metadata_text = (
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    metadata_sha256 = "sha256:" + hashlib.sha256(
        metadata_text.encode("utf-8")
    ).hexdigest()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(jar, f"binary/{jar.name}")
        archive.writestr("distribution-metadata.json", metadata_text)
        archive.writestr(
            "supply_chain/sbom.cdx.json",
            json.dumps(
                _distribution_sbom(metadata),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        archive.writestr(
            "supply_chain/provenance.json",
            json.dumps(
                {
                    "schema_version": "mmm/distribution-provenance-v1",
                    "distribution_metadata_sha256": metadata_sha256,
                    "binary_sha256": metadata["jar_sha256"],
                    "fabric_metadata_sha256": metadata["fabric_metadata_sha256"],
                    "platform_lock": metadata["platform_lock"],
                    "environment": metadata["environment"],
                    "declared_fabric_dependencies": metadata[
                        "fabric_dependencies"
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
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


def _distribution_sbom(metadata: Mapping[str, Any]) -> dict[str, Any]:
    application_ref = f"fabric:{metadata['mod_id']}@{metadata['version']}"
    dependencies = metadata["fabric_dependencies"]
    components = fabric_dependency_components(dependencies)
    runtime_refs = [
        f"fabric:{item['mod_id']}:{item['relationship']}"
        for item in dependencies
        if item["fabric_section"] == "depends"
    ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": metadata["mod_id"],
                "version": metadata["version"],
                "bom-ref": application_ref,
            }
        },
        "components": components,
        "dependencies": [
            {
                "ref": application_ref,
                "dependsOn": runtime_refs,
            }
        ],
    }


def _validated_jar(metadata: dict[str, Any]) -> Path:
    if metadata.get("schema_version") != "mmm/distribution-metadata-v1":
        raise PublishingError("Unsupported distribution metadata schema.")
    try:
        requested_jar = Path(str(metadata["jar_path"])).expanduser()
    except KeyError as exc:
        raise PublishingError("Distribution metadata is missing jar_path.") from exc
    if requested_jar.is_symlink():
        raise PublishingError("Distribution JAR is missing.")
    jar = requested_jar.resolve()
    if not jar.is_file():
        raise PublishingError("Distribution JAR is missing.")
    digest = "sha256:" + hashlib.sha256(jar.read_bytes()).hexdigest()
    if digest != metadata.get("jar_sha256"):
        raise PublishingError("Distribution JAR changed after metadata was created.")
    if metadata.get("jar_name") != jar.name or metadata.get("jar_size_bytes") != jar.stat().st_size:
        raise PublishingError("Distribution JAR identity metadata is inconsistent.")

    from .spec import PlatformLock

    platform_lock = PlatformLock()
    platform_lock.validate()
    if metadata.get("platform_lock") != asdict(platform_lock):
        raise PublishingError("Distribution PlatformLock is missing or changed.")
    if metadata.get("game_versions") != [platform_lock.minecraft_version]:
        raise PublishingError("Distribution game versions are not the tested target.")
    if metadata.get("loaders") != [platform_lock.loader]:
        raise PublishingError("Distribution loader is not the tested target.")

    fabric_metadata, metadata_sha256 = _read_fabric_metadata_from_jar(jar)
    if metadata.get("fabric_metadata_sha256") != metadata_sha256:
        raise PublishingError("Fabric metadata digest is inconsistent.")
    if metadata.get("mod_id") != fabric_metadata.get("id"):
        raise PublishingError("Distribution mod ID no longer matches the JAR.")
    if metadata.get("version") != fabric_metadata.get("version"):
        raise PublishingError("Distribution version no longer matches the JAR.")
    if metadata.get("environment") != fabric_metadata.get("environment"):
        raise PublishingError("Distribution environment no longer matches the JAR.")

    project_ids = metadata.get("dependency_project_ids")
    if not isinstance(project_ids, dict):
        raise PublishingError("dependency_project_ids must be an object.")
    dependencies = dependency_inventory_from_metadata(
        fabric_metadata,
        platform_lock=platform_lock,
        modrinth_project_ids=project_ids,
    )
    if metadata.get("fabric_dependencies") != dependencies:
        raise PublishingError("Declared Fabric dependencies changed after inspection.")
    expected_modrinth = _modrinth_dependencies(
        dependencies,
        require_resolved=False,
    )
    if metadata.get("modrinth_dependencies") != expected_modrinth:
        raise PublishingError("Modrinth dependency metadata is inconsistent.")
    return jar


def _read_fabric_metadata_from_jar(
    jar: Path,
) -> tuple[dict[str, Any], str]:
    try:
        with zipfile.ZipFile(jar) as archive:
            entries = [
                info for info in archive.infolist() if info.filename == "fabric.mod.json"
            ]
            if len(entries) != 1:
                raise PublishingError(
                    "Distribution JAR must contain exactly one root fabric.mod.json."
                )
            entry = entries[0]
            unix_kind = (entry.external_attr >> 16) & 0o170000
            if (
                entry.is_dir()
                or entry.flag_bits & 0x1
                or unix_kind == 0o120000
                or entry.file_size > _MAX_FABRIC_METADATA_BYTES
            ):
                raise PublishingError("Unsafe fabric.mod.json entry in distribution JAR.")
            raw = archive.read(entry)
    except PublishingError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise PublishingError(f"Invalid distribution JAR: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishingError(f"Invalid JAR fabric.mod.json: {exc}") from exc
    return (
        _validated_fabric_metadata_object(value),
        "sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _validated_fabric_metadata_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublishingError("fabric.mod.json must contain one JSON object.")
    if value.get("schemaVersion") != 1:
        raise PublishingError("Only fabric.mod.json schemaVersion 1 is supported.")
    mod_id = value.get("id")
    if not isinstance(mod_id, str) or not _FABRIC_ID.fullmatch(mod_id):
        raise PublishingError("fabric.mod.json has an invalid mod ID.")
    version = value.get("version")
    if (
        not isinstance(version, str)
        or not version.strip()
        or version != version.strip()
        or any(ord(character) < 0x20 for character in version)
    ):
        raise PublishingError("fabric.mod.json has an invalid version.")
    environment = value.get("environment", "*")
    if environment not in {"*", "client", "server"}:
        raise PublishingError("fabric.mod.json has an unknown environment.")
    normalized = dict(value)
    normalized["environment"] = environment
    return normalized


def _normalize_fabric_dependencies(
    fabric_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for section, relationship in _DEPENDENCY_RELATIONSHIPS.items():
        raw_section = fabric_metadata.get(section, {})
        if not isinstance(raw_section, dict):
            raise PublishingError(f"fabric.mod.json {section} must be an object.")
        for dependency_id in sorted(raw_section):
            if not isinstance(dependency_id, str) or not _FABRIC_ID.fullmatch(
                dependency_id
            ):
                raise PublishingError(
                    f"fabric.mod.json {section} contains an invalid dependency ID."
                )
            if dependency_id == fabric_metadata["id"]:
                raise PublishingError("A Fabric mod may not depend on or conflict with itself.")
            prior = seen.get(dependency_id)
            if prior is not None:
                raise PublishingError(
                    f"Dependency {dependency_id!r} is declared in both {prior} and {section}."
                )
            seen[dependency_id] = section
            predicates = _normalize_version_predicates(
                raw_section[dependency_id],
                dependency_id=dependency_id,
            )
            result.append(
                {
                    "mod_id": dependency_id,
                    "fabric_section": section,
                    "relationship": relationship,
                    "version_predicates": predicates,
                    "source": "fabric.mod.json",
                }
            )
    result.sort(key=lambda item: (item["mod_id"], item["fabric_section"]))
    return result


def _normalize_version_predicates(value: Any, *, dependency_id: str) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list) and value:
        candidates = value
    else:
        raise PublishingError(
            f"Dependency {dependency_id!r} needs a string or non-empty string list."
        )
    if len(candidates) > 32:
        raise PublishingError(f"Dependency {dependency_id!r} has too many predicates.")
    normalized: list[str] = []
    for predicate in candidates:
        if (
            not isinstance(predicate, str)
            or not predicate.strip()
            or predicate != predicate.strip()
            or len(predicate) > 256
            or "${" in predicate
            or any(ord(character) < 0x20 for character in predicate)
        ):
            raise PublishingError(
                f"Dependency {dependency_id!r} has an invalid version predicate."
            )
        if predicate in normalized:
            raise PublishingError(
                f"Dependency {dependency_id!r} repeats a version predicate."
            )
        normalized.append(predicate)
    return normalized


def _validated_modrinth_project_ids(
    value: Mapping[str, str] | None,
    *,
    declared_ids: set[str],
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PublishingError("modrinth_project_ids must be an object.")
    result: dict[str, str] = {}
    for dependency_id, project_id in value.items():
        if dependency_id not in declared_ids:
            raise PublishingError(
                f"Modrinth mapping references undeclared dependency {dependency_id!r}."
            )
        if dependency_id in _PLATFORM_DEPENDENCIES:
            raise PublishingError(
                f"Platform dependency {dependency_id!r} is not a Modrinth project dependency."
            )
        if not isinstance(project_id, str) or not _MODRINTH_PROJECT_ID.fullmatch(
            project_id
        ):
            raise PublishingError(
                f"Dependency {dependency_id!r} has an invalid Modrinth project ID."
            )
        known_project_id = _KNOWN_MODRINTH_PROJECTS.get(dependency_id)
        if known_project_id is not None and project_id != known_project_id:
            raise PublishingError(
                f"Dependency {dependency_id!r} conflicts with its verified Modrinth "
                "project ID."
            )
        result[dependency_id] = project_id
    return dict(sorted(result.items()))


def _modrinth_dependencies(
    dependencies: list[dict[str, Any]],
    *,
    require_resolved: bool,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    unresolved: list[str] = []
    for dependency in dependencies:
        if dependency.get("platform_dependency") is True:
            continue
        dependency_id = str(dependency.get("mod_id", ""))
        project_id = dependency.get("modrinth_project_id")
        if not isinstance(project_id, str) or not _MODRINTH_PROJECT_ID.fullmatch(
            project_id
        ):
            unresolved.append(dependency_id)
            continue
        relationship = dependency.get("relationship")
        if relationship not in {"required", "optional", "incompatible"}:
            raise PublishingError(
                f"Dependency {dependency_id!r} has an unknown publication relationship."
            )
        result.append(
            {
                "project_id": project_id,
                "dependency_type": str(relationship),
            }
        )
    if unresolved and require_resolved:
        raise PublishingError(
            "Modrinth project IDs are required for declared external dependencies: "
            + ", ".join(sorted(unresolved))
        )
    result.sort(key=lambda item: (item["project_id"], item["dependency_type"]))
    return result


def _dependency_display_name(dependency_id: str) -> str:
    return {
        "fabricloader": "Fabric Loader",
        "fabric-api": "Fabric API",
        "minecraft": "Minecraft",
        "java": "Java",
        "geckolib": "GeckoLib",
    }.get(dependency_id, dependency_id)


def _is_exact_predicate(predicate: str) -> bool:
    return bool(
        re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]*", predicate)
        and predicate != "*"
        and not re.search(r"(?:^|\.)[xX*](?:\.|$)", predicate)
    )


def _dependency_purl(dependency_id: str, version: str) -> str | None:
    if dependency_id == "fabricloader":
        return f"pkg:maven/net.fabricmc/fabric-loader@{version}"
    if dependency_id == "fabric-api":
        return f"pkg:maven/net.fabricmc.fabric-api/fabric-api@{version}"
    if dependency_id == "geckolib":
        return (
            "pkg:maven/software.bernie.geckolib/"
            f"geckolib-fabric-1.20.1@{version}"
        )
    if dependency_id in {"minecraft", "java"}:
        return f"pkg:generic/{dependency_id}@{version}"
    return None
