from __future__ import annotations

"""Authoritative external dependency resolution for reuse proof sandboxes.

This module is the single owner of dependency names, target coordinates, Maven
repositories, and Gradle configurations used by executable reuse verification.
No downstream adapter is allowed to reinterpret an already-resolved coordinate.
Unknown dependencies and unsupported target tuples fail closed.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DependencyResolutionReceipt:
    donor_declared_coordinate: str
    requested_constraint: str
    target_loader: str
    target_minecraft: str
    resolved_coordinate: str
    repository: str
    selected_version: str
    gradle_configuration: str = "modImplementation"
    artifact_hash: str = ""
    resolution_fingerprint: str = ""
    resolution_reason: str = "exact_match"
    is_resolved: bool = True

    @property
    def repository_url(self) -> str:
        """Compatibility alias for callers that use the more explicit name."""

        return self.repository

    @property
    def dependency_name(self) -> str:
        """Stable dependency identity derived from the authoritative receipt."""

        if self.resolved_coordinate:
            return self.resolved_coordinate.rsplit(":", 1)[0]
        return self.donor_declared_coordinate

    def to_dict(self) -> dict[str, Any]:
        return {
            "donor_declared_coordinate": self.donor_declared_coordinate,
            "requested_constraint": self.requested_constraint,
            "target_loader": self.target_loader,
            "target_minecraft": self.target_minecraft,
            "resolved_coordinate": self.resolved_coordinate,
            "repository": self.repository,
            "repository_url": self.repository,
            "selected_version": self.selected_version,
            "gradle_configuration": self.gradle_configuration,
            # artifact_hash is intentionally empty until actual Maven artifact bytes
            # are downloaded and verified.  A coordinate hash is not an artifact hash.
            "artifact_hash": self.artifact_hash,
            "resolution_fingerprint": self.resolution_fingerprint,
            "resolution_reason": self.resolution_reason,
            "is_resolved": self.is_resolved,
        }


_CANONICAL_DEPENDENCY_REGISTRY: dict[str, dict[str, Any]] = {
    "geckolib": {
        "aliases": {
            "geckolib",
            "geckolib_fabric",
            "geckolib_neoforge",
            "geckolib_forge",
            "software.bernie.geckolib:geckolib-fabric",
            "software.bernie.geckolib:geckolib-neoforge",
            "software.bernie.geckolib:geckolib-forge",
        },
        "repository": "https://dl.cloudsmith.io/public/geckolib3/geckolib/maven/",
        "group": "software.bernie.geckolib",
        "name_by_loader": {
            "fabric": "geckolib-fabric",
            "neoforge": "geckolib-neoforge",
            "forge": "geckolib-forge",
        },
        "version_matrix": {
            "1.21.1": "4.6.0",
            "1.21": "4.5.0",
            "1.20.1": "4.4.4",
            "1.19.2": "4.0.0",
        },
    },
    "cloth_config": {
        "aliases": {
            "cloth_config",
            "cloth-config",
            "cloth_config_fabric",
            "cloth_config_neoforge",
            "cloth_config_forge",
            "me.shedaniel.cloth:cloth-config-fabric",
            "me.shedaniel.cloth:cloth-config-neoforge",
            "me.shedaniel.cloth:cloth-config-forge",
        },
        "repository": "https://maven.shedaniel.me/",
        "group": "me.shedaniel.cloth",
        "name_by_loader": {
            "fabric": "cloth-config-fabric",
            "neoforge": "cloth-config-neoforge",
            "forge": "cloth-config-forge",
        },
        "version_matrix": {
            "1.21.1": "15.0.127",
            "1.21": "15.0.120",
            "1.20.1": "11.1.106",
        },
    },
    "cardinal_components": {
        "aliases": {
            "cardinal_components",
            "cardinal-components",
            "cardinal_components_api",
            "cardinal-components-api",
            "cardinal_components_base",
            "cardinal-components-base",
            "dev.onyxstudios.cardinal-components-api:cardinal-components-api",
            "dev.onyxstudios.cardinal-components-api:cardinal-components-base",
        },
        "repository": "https://ladysnake.jfrog.io/artifactory/mods",
        "group": "dev.onyxstudios.cardinal-components-api",
        "name_by_loader": {
            "fabric": "cardinal-components-api",
        },
        "version_matrix": {
            "1.21.1": "6.1.1",
            "1.21": "6.0.0",
            "1.20.1": "5.2.2",
        },
    },
    "patchouli": {
        "aliases": {
            "patchouli",
            "vazkii.patchouli:patchouli",
        },
        "repository": "https://maven.blamejared.com/",
        "group": "vazkii.patchouli",
        "name_by_loader": {
            "fabric": "Patchouli",
            "neoforge": "Patchouli",
            "forge": "Patchouli",
        },
        "version_matrix": {
            "1.21.1": {
                "fabric": "1.21.1-84-FABRIC",
                "neoforge": "1.21.1-84-NEOFORGE",
                "forge": "1.21.1-84-FORGE",
            },
            "1.20.1": {
                "fabric": "1.20.1-84-FABRIC",
                "forge": "1.20.1-84-FORGE",
            },
        },
    },
}


def _normalized_dependency_token(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    pieces = value.split(":")
    if len(pieces) >= 2:
        value = ":".join(pieces[:2])
    return value.casefold().replace("-", "_")


def _build_dependency_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for key, entry in _CANONICAL_DEPENDENCY_REGISTRY.items():
        aliases = set(entry.get("aliases", ()))
        aliases.add(key)
        for artifact in entry.get("name_by_loader", {}).values():
            aliases.add(str(artifact))
            aliases.add(f"{entry.get('group', '')}:{artifact}")
        for alias in aliases:
            token = _normalized_dependency_token(alias)
            if not token:
                continue
            previous = index.get(token)
            if previous is not None and previous != key:
                raise RuntimeError(
                    f"Dependency alias collision: {alias!r} maps to {previous!r} and {key!r}"
                )
            index[token] = key
    return index


_CANONICAL_DEPENDENCY_ALIAS_INDEX = _build_dependency_alias_index()


def _canonical_dependency_key(raw: str) -> str:
    token = _normalized_dependency_token(raw)
    return _CANONICAL_DEPENDENCY_ALIAS_INDEX.get(token, "") if token else ""


def _selected_version(
    entry: Mapping[str, Any],
    *,
    target_loader: str,
    target_minecraft: str,
) -> str:
    version_map = entry.get("version_matrix", {})
    raw_version = version_map.get(target_minecraft)
    if raw_version is None:
        mc_prefix = ".".join(target_minecraft.split(".")[:2])
        raw_version = version_map.get(mc_prefix)
    if isinstance(raw_version, Mapping):
        return str(raw_version.get(target_loader.casefold()) or "").strip()
    return str(raw_version or "").strip()


def _resolution_fingerprint(
    *,
    repository: str,
    coordinate: str,
    configuration: str,
    target_loader: str,
    target_minecraft: str,
) -> str:
    payload = (
        f"{repository}\n{coordinate}\n{configuration}\n"
        f"{target_loader.casefold()}\n{target_minecraft}"
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_dependency_for_target(
    dep_name: str,
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
) -> DependencyResolutionReceipt:
    """Resolve one donor dependency against the reviewed target matrix."""

    loader = str(target_loader or "").strip().casefold()
    minecraft = str(target_minecraft or "").strip()
    canonical_key = _canonical_dependency_key(dep_name)
    entry = _CANONICAL_DEPENDENCY_REGISTRY.get(canonical_key)

    if not entry:
        return DependencyResolutionReceipt(
            donor_declared_coordinate=dep_name,
            requested_constraint=dep_name,
            target_loader=loader,
            target_minecraft=minecraft,
            resolved_coordinate="",
            repository="",
            selected_version="",
            gradle_configuration="",
            resolution_reason="NO_VERIFIED_COORDINATE",
            is_resolved=False,
        )

    repo_url = str(entry["repository"])
    group = str(entry["group"])
    artifact_name = str(entry.get("name_by_loader", {}).get(loader) or "")
    if not artifact_name:
        return DependencyResolutionReceipt(
            donor_declared_coordinate=dep_name,
            requested_constraint=f"{loader}@{minecraft}",
            target_loader=loader,
            target_minecraft=minecraft,
            resolved_coordinate="",
            repository=repo_url,
            selected_version="",
            gradle_configuration="",
            resolution_reason=f"UNSUPPORTED_LOADER_{loader.upper()}",
            is_resolved=False,
        )

    version = _selected_version(
        entry,
        target_loader=loader,
        target_minecraft=minecraft,
    )
    if not version:
        return DependencyResolutionReceipt(
            donor_declared_coordinate=dep_name,
            requested_constraint=f"{loader}@{minecraft}",
            target_loader=loader,
            target_minecraft=minecraft,
            resolved_coordinate="",
            repository=repo_url,
            selected_version="",
            gradle_configuration="",
            resolution_reason=f"NO_COMPATIBLE_VERSION_FOR_MC_{minecraft}",
            is_resolved=False,
        )

    coordinate = f"{group}:{artifact_name}:{version}"
    configuration = "modImplementation" if loader == "fabric" else "implementation"
    return DependencyResolutionReceipt(
        donor_declared_coordinate=dep_name,
        requested_constraint=f"{loader}@{minecraft}",
        target_loader=loader,
        target_minecraft=minecraft,
        resolved_coordinate=coordinate,
        repository=repo_url,
        selected_version=version,
        gradle_configuration=configuration,
        artifact_hash="",
        resolution_fingerprint=_resolution_fingerprint(
            repository=repo_url,
            coordinate=coordinate,
            configuration=configuration,
            target_loader=loader,
            target_minecraft=minecraft,
        ),
        resolution_reason="exact_matrix_match",
        is_resolved=True,
    )


def _gradle_repo_line(repository: str, *, kotlin: bool) -> str:
    if kotlin:
        return f'    maven {{ url = uri("{repository}") }}\n'
    return f"    maven {{ url = uri('{repository}') }}\n"


def _gradle_dependency_line(
    configuration: str,
    coordinate: str,
    *,
    kotlin: bool,
) -> str:
    if kotlin:
        return f'    {configuration}("{coordinate}")\n'
    return f"    {configuration} '{coordinate}'\n"


def inject_resolved_dependencies_into_build_gradle(
    build_content: str,
    receipts: Sequence[DependencyResolutionReceipt],
    *,
    is_kotlin_dsl: bool = False,
) -> tuple[str, bool]:
    """Inject exact reviewed receipt coordinates without re-resolving their names."""

    if not receipts:
        return build_content, False
    unresolved = [receipt for receipt in receipts if not receipt.is_resolved]
    if unresolved:
        names = ", ".join(receipt.donor_declared_coordinate for receipt in unresolved)
        raise ValueError(f"cannot inject unresolved dependencies: {names}")

    exact_receipts = tuple(
        receipt
        for receipt in receipts
        if receipt.resolved_coordinate
        and receipt.repository
        and receipt.gradle_configuration
        and receipt.resolution_fingerprint
    )
    if len(exact_receipts) != len(receipts):
        raise ValueError("resolved dependency receipt is missing authoritative Gradle fields")

    modified = build_content
    changed = False
    repositories = tuple(dict.fromkeys(receipt.repository for receipt in exact_receipts))
    dependencies = tuple(
        dict.fromkeys(
            (receipt.gradle_configuration, receipt.resolved_coordinate)
            for receipt in exact_receipts
        )
    )

    missing_repositories = [repo for repo in repositories if repo not in modified]
    if missing_repositories:
        repo_lines = "".join(
            _gradle_repo_line(repo, kotlin=is_kotlin_dsl)
            for repo in missing_repositories
        )
        marker = "repositories {"
        if marker in modified:
            modified = modified.replace(marker, f"{marker}\n{repo_lines}", 1)
        else:
            modified += f"\nrepositories {{\n{repo_lines}}}\n"
        changed = True

    missing_dependencies = [
        (configuration, coordinate)
        for configuration, coordinate in dependencies
        if coordinate not in modified
    ]
    if missing_dependencies:
        dep_lines = "".join(
            _gradle_dependency_line(
                configuration,
                coordinate,
                kotlin=is_kotlin_dsl,
            )
            for configuration, coordinate in missing_dependencies
        )
        marker = "dependencies {"
        if marker in modified:
            modified = modified.replace(marker, f"{marker}\n{dep_lines}", 1)
        else:
            modified += f"\ndependencies {{\n{dep_lines}}}\n"
        changed = True

    for receipt in exact_receipts:
        if receipt.resolved_coordinate not in modified:
            raise ValueError(
                f"dependency injection did not materialize {receipt.resolved_coordinate}"
            )
        if receipt.repository not in modified:
            raise ValueError(
                f"dependency injection did not materialize repository {receipt.repository}"
            )

    return modified, changed


def parse_donor_build_metadata(files: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract reviewed external dependency identifiers from donor metadata."""

    declared: list[str] = []

    def add_if_known(value: str) -> None:
        canonical = _canonical_dependency_key(value)
        if canonical:
            declared.append(canonical)

    for rel_path, content in files.items():
        text = (
            content
            if isinstance(content, str)
            else content.decode("utf-8", errors="ignore")
        )
        path = rel_path.lower()

        if "fabric.mod.json" in path:
            for match in re.findall(r'"([a-zA-Z0-9_.-]+)":\s*"[^"]+"', text):
                if match not in {"fabricloader", "fabric", "minecraft", "java"}:
                    add_if_known(match)

        if "build.gradle" in path:
            coordinate_pattern = (
                r"(?:modImplementation|implementation|include|api)"
                r"\s*\(?\s*[\"']([^\"':]+):([^\"':]+):([^\"']+)[\"']"
            )
            for group, artifact, _version in re.findall(coordinate_pattern, text):
                add_if_known(f"{group}:{artifact}")

        if "libs.versions.toml" in path or path.endswith(".toml"):
            for group, artifact in re.findall(
                r'module\s*=\s*[\"\']([^\"\':]+):([^\"\':]+)[\"\']',
                text,
            ):
                add_if_known(f"{group}:{artifact}")
            for match in re.findall(
                r'modId\s*=\s*[\"\']([a-zA-Z0-9_.-]+)[\"\']',
                text,
            ):
                if match not in {"minecraft", "forge", "neoforge"}:
                    add_if_known(match)

    return tuple(dict.fromkeys(declared))
