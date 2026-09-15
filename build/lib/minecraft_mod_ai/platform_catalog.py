from __future__ import annotations

"""Executable Minecraft platform-provider registry.

A loader name alone is not support. A target is selectable only when its provider
can resolve and validate the complete generation/build/validation toolchain.
Candidate discovery is deliberately bounded: platform selection must never crawl
an entire historical Minecraft catalogue and fail one version at a time.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .platform_live_discovery import (
    PlatformDiscoveryError,
    _emit_discovery_log,
)
from .target_contract import (
    TargetContract,
)

# Platform evidence uses adapter as domain vocabulary, but the executable receipt has
# one canonical runtime representation: TargetContract. Keep the exported name as a
# type synonym rather than reintroducing a second adapter class or conversion layer.
PlatformAdapter = TargetContract


@dataclass(frozen=True)
class PlatformProvider:
    loader: str
    provider_id: str
    discover_versions: Callable[[int], tuple[str, ...]]
    resolve: Callable[[str], TargetContract]
    host_authoritative: bool = False


_PROVIDER_LOCK = RLock()
_PROVIDERS: dict[str, PlatformProvider] = {}


def register_platform_provider(provider: PlatformProvider, *, replace: bool = False) -> None:
    loader = _loader_id(provider.loader)
    if not loader:
        raise ValueError("Platform provider loader must not be empty.")
    if loader != provider.loader:
        provider = PlatformProvider(
            loader=loader,
            provider_id=provider.provider_id,
            discover_versions=provider.discover_versions,
            resolve=provider.resolve,
            host_authoritative=provider.host_authoritative,
        )
    with _PROVIDER_LOCK:
        if loader in _PROVIDERS and not replace:
            raise ValueError(f"Executable provider already registered for loader={loader!r}.")
        _PROVIDERS[loader] = provider


def executable_loaders() -> tuple[str, ...]:
    with _PROVIDER_LOCK:
        return tuple(sorted(_PROVIDERS))


def provider_for_loader(loader: str) -> PlatformProvider:
    normalized = _loader_id(loader)
    with _PROVIDER_LOCK:
        provider = _PROVIDERS.get(normalized)
    if provider is None:
        raise ValueError(
            f"No executable platform provider is installed for loader={normalized!r}."
        )
    return provider


def _candidate_versions(
    provider: PlatformProvider,
    *,
    limit: int,
) -> tuple[str, ...]:
    bound = max(1, int(limit))
    raw = provider.discover_versions(bound)
    normalized = tuple(
        dict.fromkeys(
            version
            for item in raw
            if (version := str(item).strip())
        )
    )
    return normalized[:bound]


def _resolve_candidate(
    provider: PlatformProvider,
    version: str,
) -> tuple[TargetContract | None, str | None]:
    try:
        adapter = provider.resolve(version)
        adapter.validate()
        if adapter.minecraft_version != version or adapter.loader != provider.loader:
            raise ValueError("PINNED_VERSION_SUBSTITUTION: provider changed the requested target")
        return adapter, None
    except (PlatformDiscoveryError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _record_diagnostic(
    message: str,
    diagnostics: list[str] | None,
    *,
    emit: bool = True,
) -> None:
    if diagnostics is not None:
        diagnostics.append(message)
    if emit:
        _emit_discovery_log(message)


def discover_target_keys(
    *,
    loader: str | None = None,
    minecraft_version: str | None = None,
    limit_per_loader: int = 12,
    diagnostics: list[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    loaders = (provider_for_loader(loader).loader,) if loader else executable_loaders()
    requested_version = str(minecraft_version or "").strip()
    bound = max(1, int(limit_per_loader))
    result: list[tuple[str, str]] = []

    for loader_id in loaders:
        provider = provider_for_loader(loader_id)
        if requested_version:
            adapter, error = _resolve_candidate(provider, requested_version)
            if adapter is not None:
                result.append((loader_id, requested_version))
            else:
                _record_diagnostic(
                    f"target unavailable loader={loader_id} version={requested_version}: {error}",
                    diagnostics,
                )
            continue

        try:
            versions = _candidate_versions(provider, limit=bound)
        except Exception as exc:  # noqa: BLE001
            _record_diagnostic(
                f"version discovery failed loader={loader_id}: {type(exc).__name__}: {exc}",
                diagnostics,
            )
            continue
        if not versions:
            _record_diagnostic(
                f"provider returned no candidate Minecraft versions loader={loader_id}",
                diagnostics,
            )
            continue

        executable_count = 0
        for version in versions:
            adapter, error = _resolve_candidate(provider, version)
            if adapter is None:
                _record_diagnostic(
                    f"target skipped loader={loader_id} version={version}: {error}",
                    diagnostics,
                )
                continue
            executable_count += 1
            result.append((loader_id, version))
        if executable_count == 0:
            _record_diagnostic(
                f"provider exposed no executable target in newest {len(versions)} candidates loader={loader_id}",
                diagnostics,
            )
    return tuple(result)


def supported_minecraft_versions(*, loader: str | None = None) -> tuple[str, ...]:
    keys = discover_target_keys(loader=loader, limit_per_loader=32)
    values: list[str] = []
    seen: set[str] = set()
    for _loader, version in keys:
        if version not in seen:
            seen.add(version)
            values.append(version)
    return tuple(values)


def adapters_for_version(minecraft_version: str) -> tuple[TargetContract, ...]:
    version = str(minecraft_version).strip()
    result: list[TargetContract] = []
    for loader in executable_loaders():
        try:
            result.append(adapter_for_target(version, loader))
        except ValueError:
            continue
    return tuple(result)


def adapter_for_target(minecraft_version: str, loader: str) -> TargetContract:
    version = str(minecraft_version).strip()
    if not version:
        raise ValueError("Minecraft version must not be empty when resolving an exact target.")
    normalized_loader = _loader_id(loader)
    provider = provider_for_loader(normalized_loader)
    try:
        adapter = provider.resolve(version)
        adapter.validate()
        if adapter.minecraft_version != version or adapter.loader != normalized_loader:
            from .resolved_version_context import VersionContextError

            raise VersionContextError("PINNED_VERSION_SUBSTITUTION", requested=version, actual=adapter.minecraft_version)
        return adapter
    except (PlatformDiscoveryError, ValueError) as exc:
        _emit_discovery_log(
            f"target resolution failed loader={normalized_loader} version={version}: "
            f"{type(exc).__name__}: {exc}"
        )
        if isinstance(exc, ValueError):
            raise
        raise ValueError(str(exc)) from None
    except Exception as exc:
        _emit_discovery_log(
            f"target resolution crashed loader={normalized_loader} version={version}: "
            f"{type(exc).__name__}: {exc}",
            exc_info=True,
        )
        raise


def newest_adapter(*, loader: str) -> TargetContract:
    keys = discover_target_keys(loader=loader, limit_per_loader=12)
    if not keys:
        raise ValueError(f"No executable platform target for loader={loader!r}.")
    return adapter_for_target(keys[0][1], keys[0][0])


def adapter_for_lock_values(value: Any) -> TargetContract:
    adapter = adapter_for_target(
        str(getattr(value, "minecraft_version", "")),
        str(getattr(value, "loader", "")),
    )
    fields = [
        "edition",
        "loader",
        "minecraft_version",
        "java_version",
        "fabric_loader",
        "fabric_api",
        "fabric_loom",
        "gradle",
    ]
    if adapter.mappings_applicable:
        fields.append("yarn_mappings")
    mismatches = [
        field for field in fields if getattr(value, field, None) != getattr(adapter, field)
    ]
    if mismatches:
        raise ValueError(
            "Platform lock disagrees with the executable provider receipt for fields "
            f"{mismatches}."
        )
    return adapter


def _project_platform_lock(root: Path) -> Path | None:
    """Resolve the authoritative lock for a project or an MMM generation checkpoint.

    Checkpoints intentionally exclude ``.minecraft_ai`` runtime state. They are still
    derived from one generated project, so an exact host-owned checkpoint path may use
    that project's sibling platform lock. No arbitrary ancestor search is permitted.
    """

    direct = root / ".minecraft_ai" / "platform-lock.json"
    if direct.is_file() and not direct.is_symlink():
        return direct

    if root.name != "project":
        return None
    checkpoint_root = root.parent
    checkpoint_directory = checkpoint_root.parent
    metadata_root = checkpoint_directory.parent
    key = checkpoint_root.name
    if (
        checkpoint_directory.name != ".mmm-custom-checkpoints"
        or metadata_root.name != ".minecraft_ai"
        or len(key) != 64
        or any(character not in "0123456789abcdef" for character in key)
    ):
        return None
    inherited = metadata_root / "platform-lock.json"
    if inherited.is_file() and not inherited.is_symlink():
        return inherited
    return None


def _fabric_descriptor_identifies_project(root: Path) -> bool:
    """Use only a valid local Fabric descriptor as loader evidence.

    This is deliberately narrower than scanning Gradle scripts or parent directories.
    A regular ``fabric.mod.json`` with the required descriptor identity fields is
    authoritative evidence that the staged project targets Fabric even when host
    runtime metadata was excluded from a generation checkpoint.
    """

    descriptor = root / "src" / "main" / "resources" / "fabric.mod.json"
    if not descriptor.is_file() or descriptor.is_symlink():
        return False
    try:
        raw = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict):
        return False
    schema_version = raw.get("schemaVersion")
    mod_id = raw.get("id")
    version = raw.get("version")
    return (
        type(schema_version) is int
        and schema_version >= 1
        and isinstance(mod_id, str)
        and bool(mod_id.strip())
        and isinstance(version, str)
        and bool(version.strip())
    )


def adapter_from_project(project_root: str | Path) -> TargetContract:
    root = Path(project_root).expanduser().resolve()
    lock_file = _project_platform_lock(root)
    if lock_file is not None:
        raw = json.loads(lock_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Generated platform lock must be an object.")
        loader = str(raw.get("loader") or "").strip()
        version = str(raw.get("minecraft_version") or "").strip()
        adapter = adapter_for_target(version, loader)
        fields = [
            "minecraft_version",
            "loader",
            "java_version",
            "fabric_loader",
            "fabric_api",
            "fabric_loom",
            "gradle",
        ]
        if adapter.mappings_applicable:
            fields.append("yarn_mappings")
        for field in fields:
            if str(raw.get(field) or "") != str(getattr(adapter, field) or ""):
                raise ValueError(
                    f"Generated platform lock disagrees with executable provider: {field}"
                )
        return adapter

    properties = _read_gradle_properties(root / "gradle.properties")
    minecraft_version = properties.get("minecraft_version", "").strip()
    loader = properties.get("loader", "").strip().casefold()
    if not loader:
        if (
            properties.get("loader_version") and properties.get("fabric_version")
        ) or _fabric_descriptor_identifies_project(root):
            loader = "fabric"
        else:
            raise ValueError("Existing project loader could not be identified unambiguously.")
    adapter = adapter_for_target(minecraft_version, loader)
    if loader == "fabric":
        expected = {
            "loader_version": adapter.fabric_loader,
            "fabric_version": adapter.fabric_api,
            "loom_version": adapter.fabric_loom,
        }
        if adapter.mappings_kind == "yarn":
            expected["yarn_mappings"] = adapter.mappings_version
        for key, expected_value in expected.items():
            actual = properties.get(key)
            if actual and actual != expected_value:
                raise ValueError(
                    f"Project Gradle property {key} disagrees with executable provider discovery."
                )
    return adapter


def platform_catalog_receipt() -> dict[str, Any]:
    providers = []
    for loader in executable_loaders():
        provider = provider_for_loader(loader)
        keys = discover_target_keys(loader=loader, limit_per_loader=32)
        providers.append(
            {
                "loader": loader,
                "provider_id": provider.provider_id,
                "minecraft_versions": [version for _loader, version in keys],
            }
        )
    return {
        "schema_version": "mmm/executable-platform-registry-v3",
        "providers": providers,
    }


def _fabric_versions(limit: int) -> tuple[str, ...]:
    from .host_version_catalog import host_versions

    return host_versions(max(1, int(limit)))


def _fabric_adapter(minecraft_version: str) -> TargetContract:
    from .host_version_catalog import host_target

    return host_target(minecraft_version)


def _read_gradle_properties(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"gradle.properties is missing: {path}")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _loader_id(value: str) -> str:
    return str(value or "").strip().casefold()


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


register_platform_provider(
    PlatformProvider(
        loader="fabric",
        provider_id="host-coherent-version-catalog-v1",
        discover_versions=_fabric_versions,
        resolve=_fabric_adapter,
        host_authoritative=True,
    )
)
