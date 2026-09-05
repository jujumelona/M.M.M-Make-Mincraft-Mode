from __future__ import annotations

"""Executable Minecraft platform-provider registry.

A loader name alone is not support. A target is selectable only when its provider
can resolve and validate the complete generation/build/validation toolchain.
Candidate discovery is deliberately bounded: platform selection must never crawl
an entire historical Minecraft catalogue and fail one version at a time.
"""

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .platform_live_discovery import (
    PlatformDiscoveryError,
    _emit_discovery_log,
    discover_fabric_target,
    latest_stable_versions,
)

_NATIVE_NAME_MIN_VERSION = (26, 1)


def _version_key(value: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\D|$)", str(value or "").strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _uses_native_names(value: str) -> bool:
    version = _version_key(value)
    return version is not None and version >= _NATIVE_NAME_MIN_VERSION


@dataclass(frozen=True)
class PlatformAdapter:
    adapter_id: str
    edition: str
    loader: str
    minecraft_version: str
    java_version: str
    # Backward-compatible coordinate consumed by existing generators. It is empty for
    # Minecraft 26.1+ native/unobfuscated targets where mappings are inapplicable.
    yarn_mappings: str
    mappings_kind: str
    mappings_version: str
    fabric_loader: str
    fabric_api: str
    fabric_loom: str
    gradle: str
    gradle_sha256: str
    data_pack_version: str
    resource_pack_version: str
    resource_pack_format: int
    release_metadata_url: str
    source_api_family: str
    deterministic_module_kinds: frozenset[str]

    @property
    def mappings_applicable(self) -> bool:
        return not _uses_native_names(self.minecraft_version)

    def validate(self) -> None:
        required = {
            "adapter_id": self.adapter_id,
            "edition": self.edition,
            "loader": self.loader,
            "minecraft_version": self.minecraft_version,
            "java_version": self.java_version,
            "fabric_loader": self.fabric_loader,
            "fabric_api": self.fabric_api,
            "fabric_loom": self.fabric_loom,
            "gradle": self.gradle,
            "gradle_sha256": self.gradle_sha256,
            "data_pack_version": self.data_pack_version,
            "resource_pack_version": self.resource_pack_version,
            "release_metadata_url": self.release_metadata_url,
            "source_api_family": self.source_api_family,
        }
        if self.mappings_applicable:
            required.update(
                {
                    "yarn_mappings": self.yarn_mappings,
                    "mappings_kind": self.mappings_kind,
                    "mappings_version": self.mappings_version,
                }
            )
        missing = sorted(key for key, value in required.items() if not str(value).strip())
        if missing:
            raise ValueError(
                "Executable platform provider returned partial target metadata: "
                f"{missing}."
            )
        if self.mappings_applicable:
            if self.mappings_kind not in {"mojang", "yarn"}:
                raise ValueError(f"Unsupported mappings kind: {self.mappings_kind!r}.")
            if self.mappings_kind == "mojang" and self.mappings_version != "mojang":
                raise ValueError("Mojang mappings must use the canonical mappings_version='mojang'.")
            if self.yarn_mappings != self.mappings_version:
                raise ValueError(
                    "Legacy yarn_mappings compatibility coordinate disagrees with mappings_version."
                )
        elif any((self.yarn_mappings, self.mappings_kind, self.mappings_version)):
            raise ValueError(
                "Minecraft 26.1+ native/unobfuscated targets must not expose legacy mapping coordinates."
            )
        if _uses_native_names(self.minecraft_version):
            if not str(self.java_version).isdigit() or int(self.java_version) < 25:
                raise ValueError(
                    f"Minecraft {self.minecraft_version} native-name target requires Java 25+."
                )
        if not re.fullmatch(r"[0-9a-f]{64}", self.gradle_sha256):
            raise ValueError("Executable platform provider returned an invalid Gradle SHA-256.")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", self.data_pack_version):
            raise ValueError("Executable platform provider returned an invalid data pack version.")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", self.resource_pack_version):
            raise ValueError("Executable platform provider returned an invalid resource pack version.")
        expected_major = int(self.resource_pack_version.split(".", 1)[0])
        if type(self.resource_pack_format) is not int or self.resource_pack_format <= 0:
            raise ValueError("Resource pack format must be a positive provider-derived integer.")
        if self.resource_pack_format != expected_major:
            raise ValueError(
                "Resource pack format major disagrees with the exact provider resource-pack version."
            )
        if not self.release_metadata_url.startswith(
            (
                "https://www.minecraft.net/",
                "https://feedback.minecraft.net/",
                "https://piston-meta.mojang.com/",
                "https://launcher.mojang.com/",
            )
        ):
            raise ValueError(
                "Pack metadata must be grounded in an official Minecraft/Mojang metadata URL."
            )

    def public_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["deterministic_module_kinds"] = sorted(self.deterministic_module_kinds)
        if self.mappings_applicable:
            value["mappings"] = {
                "kind": self.mappings_kind,
                "version": self.mappings_version,
            }
        else:
            value.pop("yarn_mappings", None)
            value.pop("mappings_kind", None)
            value.pop("mappings_version", None)
        value["naming_regime"] = {
            "kind": "mapped_obfuscated" if self.mappings_applicable else "native_unobfuscated",
            "mappings_applicable": self.mappings_applicable,
            "minecraft_version": self.minecraft_version,
        }
        value["pack_versions"] = {
            "data": self.data_pack_version,
            "resource": self.resource_pack_version,
            "resource_major": self.resource_pack_format,
        }
        return value


@dataclass(frozen=True)
class PlatformProvider:
    loader: str
    provider_id: str
    discover_versions: Callable[[int], tuple[str, ...]]
    resolve: Callable[[str], PlatformAdapter]


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
    """Read one bounded newest-first candidate page from a provider.

    The old implementation repeatedly doubled the page size until every historical
    Minecraft release had been enumerated. That converted ordinary target selection
    into an unbounded compatibility crawl. Providers are now asked once and the caller's
    candidate bound is authoritative even if a provider accidentally returns more rows.
    """

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
) -> tuple[PlatformAdapter | None, str | None]:
    """Resolve a candidate without emitting a traceback for normal incompatibility."""

    try:
        adapter = provider.resolve(version)
        adapter.validate()
        return adapter, None
    except (PlatformDiscoveryError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - a bad candidate must not poison discovery
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
    """Return only complete executable platform targets.

    Exact version requests bypass catalogue enumeration and resolve that target directly.
    Automatic selection examines at most ``limit_per_loader`` newest candidates per
    provider and rejects incomplete candidates before they become optimizer inputs.
    """

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
        except Exception as exc:  # noqa: BLE001 - one provider must not hide other providers
            _record_diagnostic(
                f"version discovery failed loader={loader_id}: "
                f"{type(exc).__name__}: {exc}",
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
                f"provider exposed no executable target in newest {len(versions)} candidates "
                f"loader={loader_id}",
                diagnostics,
            )

    return tuple(result)


def supported_minecraft_versions(*, loader: str | None = None) -> tuple[str, ...]:
    """Return bounded, provider-validated executable Minecraft versions."""

    keys = discover_target_keys(loader=loader, limit_per_loader=32)
    values: list[str] = []
    seen: set[str] = set()
    for _loader, version in keys:
        if version not in seen:
            seen.add(version)
            values.append(version)
    return tuple(values)


def adapters_for_version(minecraft_version: str) -> tuple[PlatformAdapter, ...]:
    version = str(minecraft_version).strip()
    result: list[PlatformAdapter] = []
    for loader in executable_loaders():
        try:
            result.append(adapter_for_target(version, loader))
        except ValueError:
            continue
    return tuple(result)


def adapter_for_target(minecraft_version: str, loader: str) -> PlatformAdapter:
    version = str(minecraft_version).strip()
    if not version:
        raise ValueError("Minecraft version must not be empty when resolving an exact target.")
    normalized_loader = _loader_id(loader)
    provider = provider_for_loader(normalized_loader)
    try:
        adapter = provider.resolve(version)
        adapter.validate()
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


def newest_adapter(*, loader: str) -> PlatformAdapter:
    keys = discover_target_keys(loader=loader, limit_per_loader=12)
    if not keys:
        raise ValueError(f"No executable platform target for loader={loader!r}.")
    return adapter_for_target(keys[0][1], keys[0][0])


def adapter_for_lock_values(value: Any) -> PlatformAdapter:
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


def adapter_from_project(project_root: str | Path) -> PlatformAdapter:
    root = Path(project_root).expanduser().resolve()
    lock_file = root / ".minecraft_ai" / "platform-lock.json"
    if lock_file.is_file() and not lock_file.is_symlink():
        import json

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
        if properties.get("loader_version") and properties.get("fabric_version"):
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
    try:
        return latest_stable_versions(limit=max(1, int(limit)))
    except PlatformDiscoveryError as exc:
        _emit_discovery_log(
            f"Fabric stable-version discovery failed: {type(exc).__name__}: {exc}",
            exc_info=True,
        )
        raise


def _fabric_adapter(minecraft_version: str) -> PlatformAdapter:
    version = str(minecraft_version).strip()
    if not version:
        raise ValueError("Minecraft version must not be empty for Fabric discovery.")
    try:
        target = discover_fabric_target(version)
    except PlatformDiscoveryError as exc:
        raise ValueError(str(exc)) from None
    digest = target.discovery_sha256.split(":", 1)[-1][:12]
    try:
        resource_major = int(target.resource_pack_version.split(".", 1)[0])
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "Official target discovery returned an invalid resource-pack version."
        ) from exc
    native_names = _uses_native_names(target.minecraft_version)
    mappings_kind = "" if native_names else target.mappings_kind
    mappings_version = "" if native_names else target.mappings_version
    adapter = PlatformAdapter(
        adapter_id=f"fabric_live_{_safe_id(version)}_{digest}",
        edition="java",
        loader="fabric",
        minecraft_version=target.minecraft_version,
        java_version=target.java_version,
        yarn_mappings=mappings_version,
        mappings_kind=mappings_kind,
        mappings_version=mappings_version,
        fabric_loader=target.loader_version,
        fabric_api=target.fabric_api_version,
        fabric_loom=target.loom_version,
        gradle=target.gradle_version,
        gradle_sha256=target.gradle_sha256,
        data_pack_version=target.data_pack_version,
        resource_pack_version=target.resource_pack_version,
        resource_pack_format=resource_major,
        release_metadata_url=target.release_metadata_url,
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )
    adapter.validate()
    return adapter


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
        provider_id="official-fabric-meta-maven-minecraft-release-v3",
        discover_versions=_fabric_versions,
        resolve=_fabric_adapter,
    )
)
