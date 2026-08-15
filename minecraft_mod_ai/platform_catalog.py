from __future__ import annotations

"""Executable Minecraft platform-provider registry.

A loader name is not support.  A target becomes selectable only when a registered
provider can resolve the complete toolchain needed by generation/build/validation.
The registry is loader-neutral; the built-in provider currently implemented by MMM
is Fabric.  Additional loaders must register a real resolver/provider before the
optimizer can ever select them.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from .platform_live_discovery import (
    PlatformDiscoveryError,
    discover_fabric_target,
    latest_stable_versions,
)


@dataclass(frozen=True)
class PlatformAdapter:
    adapter_id: str
    edition: str
    loader: str
    minecraft_version: str
    java_version: str
    yarn_mappings: str
    fabric_loader: str
    fabric_api: str
    fabric_loom: str
    gradle: str
    gradle_sha256: str
    resource_pack_format: int
    source_api_family: str
    deterministic_module_kinds: frozenset[str]

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["deterministic_module_kinds"] = sorted(self.deterministic_module_kinds)
        return value


@dataclass(frozen=True)
class PlatformProvider:
    loader: str
    provider_id: str
    discover_versions: Callable[[int], tuple[str, ...]]
    resolve: Callable[[str], PlatformAdapter]


_GRADLE_8_12_SHA256 = "7a00d51fb93147819aab76024feece20b6b84e420694101f276be952e08bef03"

# These are offline compatibility receipts for already exercised targets, not an
# allowlist and not an automatic preference order.
FABRIC_1201 = PlatformAdapter(
    adapter_id="fabric_1_20_1",
    edition="java",
    loader="fabric",
    minecraft_version="1.20.1",
    java_version="17",
    yarn_mappings="1.20.1+build.1",
    fabric_loader="0.17.2",
    fabric_api="0.92.11+1.20.1",
    fabric_loom="1.10.5",
    gradle="8.12",
    gradle_sha256=_GRADLE_8_12_SHA256,
    resource_pack_format=15,
    source_api_family="fabric_1201",
    deterministic_module_kinds=frozenset(
        {
            "item", "block", "tool", "weapon", "armor", "food", "crop",
            "machine", "recipe", "effect", "enchantment", "command",
            "advancement", "loot", "entity", "boss", "npc", "quest",
            "class", "skill", "economy", "shop", "gui", "networking",
            "party", "guild", "structure", "biome", "dimension",
            "world_event", "audio", "integration", "custom_java", "fluid",
        }
    ),
)

FABRIC_1211 = PlatformAdapter(
    adapter_id="fabric_1_21_1",
    edition="java",
    loader="fabric",
    minecraft_version="1.21.1",
    java_version="21",
    yarn_mappings="1.21.1+build.3",
    fabric_loader="0.19.3",
    fabric_api="0.116.15+1.21.1",
    fabric_loom="1.10.5",
    gradle="8.12",
    gradle_sha256=_GRADLE_8_12_SHA256,
    resource_pack_format=34,
    source_api_family="fabric_1211",
    deterministic_module_kinds=frozenset(
        {"item", "block", "recipe", "advancement", "loot", "command"}
    ),
)

PLATFORM_ADAPTERS: tuple[PlatformAdapter, ...] = (FABRIC_1201, FABRIC_1211)
_BY_SEED_TARGET = {(item.loader, item.minecraft_version): item for item in PLATFORM_ADAPTERS}
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


def supported_minecraft_versions(*, loader: str | None = None) -> tuple[str, ...]:
    """Return provider-discovered versions; never imply support for an absent provider."""
    if loader is not None:
        return provider_for_loader(loader).discover_versions(32)
    values: list[str] = []
    seen: set[str] = set()
    for loader_id in executable_loaders():
        for version in provider_for_loader(loader_id).discover_versions(32):
            if version not in seen:
                seen.add(version)
                values.append(version)
    return tuple(values)


def discover_target_keys(
    *,
    loader: str | None = None,
    minecraft_version: str | None = None,
    limit_per_loader: int = 12,
) -> tuple[tuple[str, str], ...]:
    """Enumerate only targets backed by an executable provider."""
    loaders = (provider_for_loader(loader).loader,) if loader else executable_loaders()
    result: list[tuple[str, str]] = []
    for loader_id in loaders:
        provider = provider_for_loader(loader_id)
        versions = provider.discover_versions(max(1, int(limit_per_loader)))
        if minecraft_version:
            versions = tuple(value for value in versions if value == minecraft_version)
            if not versions:
                # Exact explicit/existing targets may be older than the discovery
                # window. Resolve them directly and include only if executable.
                try:
                    provider.resolve(str(minecraft_version))
                except ValueError:
                    continue
                versions = (str(minecraft_version),)
        for version in versions[: max(1, int(limit_per_loader))]:
            try:
                provider.resolve(version)
            except ValueError:
                continue
            result.append((loader_id, version))
    return tuple(result)


def adapters_for_version(minecraft_version: str) -> tuple[PlatformAdapter, ...]:
    version = str(minecraft_version).strip()
    result: list[PlatformAdapter] = []
    for loader in executable_loaders():
        try:
            result.append(provider_for_loader(loader).resolve(version))
        except ValueError:
            continue
    return tuple(result)


def adapter_for_target(minecraft_version: str, loader: str) -> PlatformAdapter:
    version = str(minecraft_version).strip()
    if not version:
        raise ValueError("Minecraft version must not be empty when resolving an exact target.")
    return provider_for_loader(loader).resolve(version)


def newest_adapter(*, loader: str) -> PlatformAdapter:
    """Compatibility helper only. Automatic planning must use platform_optimizer."""
    versions = supported_minecraft_versions(loader=loader)
    if not versions:
        raise ValueError(f"No discoverable platform target for loader={loader!r}.")
    return adapter_for_target(versions[0], loader)


def adapter_for_lock_values(value: Any) -> PlatformAdapter:
    adapter = adapter_for_target(
        str(getattr(value, "minecraft_version", "")),
        str(getattr(value, "loader", "")),
    )
    fields = (
        "edition",
        "loader",
        "minecraft_version",
        "java_version",
        "yarn_mappings",
        "fabric_loader",
        "fabric_api",
        "fabric_loom",
        "gradle",
    )
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
        for field in (
            "minecraft_version", "loader", "java_version", "yarn_mappings",
            "fabric_loader", "fabric_api", "fabric_loom", "gradle",
        ):
            if raw.get(field) != getattr(adapter, field):
                raise ValueError(
                    f"Generated platform lock disagrees with executable provider: {field}"
                )
        return adapter

    properties = _read_gradle_properties(root / "gradle.properties")
    minecraft_version = properties.get("minecraft_version", "").strip()
    loader = properties.get("loader", "").strip().casefold()
    if not loader:
        # Detect only unambiguous, executable Gradle markers. Never assume Fabric.
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
        if adapter.yarn_mappings != "mojang":
            expected["yarn_mappings"] = adapter.yarn_mappings
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
        providers.append(
            {
                "loader": loader,
                "provider_id": provider.provider_id,
                "minecraft_versions": list(provider.discover_versions(32)),
            }
        )
    return {
        "schema_version": "mmm/executable-platform-registry-v1",
        "providers": providers,
        "offline_compatibility_seeds": [item.public_dict() for item in PLATFORM_ADAPTERS],
    }


def _fabric_versions(limit: int) -> tuple[str, ...]:
    try:
        return latest_stable_versions(limit=max(1, int(limit)))
    except PlatformDiscoveryError:
        return tuple(item.minecraft_version for item in PLATFORM_ADAPTERS)[: max(1, int(limit))]


def _fabric_adapter(minecraft_version: str) -> PlatformAdapter:
    version = str(minecraft_version).strip()
    seed = _BY_SEED_TARGET.get(("fabric", version))
    if seed is not None:
        return seed
    try:
        target = discover_fabric_target(version)
    except PlatformDiscoveryError as exc:
        raise ValueError(str(exc)) from exc
    digest = target.discovery_sha256.split(":", 1)[-1][:12]
    return PlatformAdapter(
        adapter_id=f"fabric_live_{_safe_id(version)}_{digest}",
        edition="java",
        loader="fabric",
        minecraft_version=target.minecraft_version,
        java_version=target.java_version,
        yarn_mappings=target.mappings_version,
        fabric_loader=target.loader_version,
        fabric_api=target.fabric_api_version,
        fabric_loom=target.loom_version,
        gradle=target.gradle_version,
        gradle_sha256=target.gradle_sha256,
        resource_pack_format=0,
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


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
        provider_id="official-fabric-meta-maven-template-v1",
        discover_versions=_fabric_versions,
        resolve=_fabric_adapter,
    )
)
