from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .platform_live_discovery import (
    PlatformDiscoveryError,
    discover_fabric_target,
    latest_stable_versions,
)


@dataclass(frozen=True)
class PlatformAdapter:
    """Immutable target resolved from official platform metadata.

    Static entries below are offline compatibility seeds for old saved proposals, not
    a supported-version allowlist. Unknown/future Fabric versions are resolved live.
    """

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


_GRADLE_8_12_SHA256 = "7a00d51fb93147819aab76024feece20b6b84e420694101f276be952e08bef03"

# Offline seeds preserve existing MMM proposals and the already exercised deterministic
# templates. They are optimisations/fallbacks only; adding a Minecraft release here is
# NOT required for the live platform path.
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

# Backward-compatible public symbol. This is deliberately named/treated as seeds in
# all resolution functions below and is no longer the set of supported versions.
PLATFORM_ADAPTERS: tuple[PlatformAdapter, ...] = (FABRIC_1201, FABRIC_1211)
_BY_SEED_TARGET = {(item.loader, item.minecraft_version): item for item in PLATFORM_ADAPTERS}


def _live_adapter(minecraft_version: str) -> PlatformAdapter:
    target = discover_fabric_target(minecraft_version)
    digest = target.discovery_sha256.split(":", 1)[-1][:12]
    return PlatformAdapter(
        adapter_id=f"fabric_live_{_safe_id(minecraft_version)}_{digest}",
        edition="java",
        loader="fabric",
        minecraft_version=target.minecraft_version,
        java_version=target.java_version,
        # Legacy field name is retained in Proposal v1. For unobfuscated/Mojang-mapped
        # releases the value is the explicit marker 'mojang'; generators inspect
        # source_api_family and must not construct a Yarn Maven coordinate from it.
        yarn_mappings=target.mappings_version,
        fabric_loader=target.loader_version,
        fabric_api=target.fabric_api_version,
        fabric_loom=target.loom_version,
        gradle=target.gradle_version,
        gradle_sha256=target.gradle_sha256,
        # Live targets are bootstrapped by Fabric's official template provider; MMM
        # does not guess a pack-format integer for an unseen game version.
        resource_pack_format=0,
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def supported_minecraft_versions(*, loader: str = "fabric") -> tuple[str, ...]:
    loader = loader.strip().lower()
    if loader != "fabric":
        return ()
    try:
        return latest_stable_versions(limit=32)
    except PlatformDiscoveryError:
        # Offline compatibility only. Online runs use Fabric Meta as source of truth.
        return tuple(item.minecraft_version for item in PLATFORM_ADAPTERS)


def newest_adapter(*, loader: str = "fabric") -> PlatformAdapter:
    versions = supported_minecraft_versions(loader=loader)
    if not versions:
        raise ValueError(f"No discoverable platform target for loader={loader!r}.")
    # Fabric Meta is ordered newest-first. Do not invent semantic ordering for
    # snapshots/calendar versions.
    return adapter_for_target(versions[0], loader)


def adapter_for_target(minecraft_version: str, loader: str = "fabric") -> PlatformAdapter:
    version = str(minecraft_version).strip()
    normalized_loader = str(loader).strip().lower()
    if normalized_loader != "fabric":
        raise ValueError(
            f"No executable platform provider is installed for loader={normalized_loader!r}."
        )
    seed = _BY_SEED_TARGET.get((normalized_loader, version))
    if seed is not None:
        return seed
    try:
        return _live_adapter(version)
    except PlatformDiscoveryError as exc:
        raise ValueError(str(exc)) from exc


def adapter_for_lock_values(value: Any) -> PlatformAdapter:
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
    for seed in PLATFORM_ADAPTERS:
        if all(getattr(value, field, None) == getattr(seed, field) for field in fields):
            return seed

    # Future/live proposals are validated against fresh official discovery rather
    # than against a list embedded in MMM source code.
    try:
        adapter = adapter_for_target(
            str(getattr(value, "minecraft_version", "")),
            str(getattr(value, "loader", "fabric")),
        )
    except ValueError as exc:
        raise ValueError(
            "Platform lock could not be verified against official discovery: " + str(exc)
        ) from exc
    mismatches = [
        field
        for field in fields
        if getattr(value, field, None) != getattr(adapter, field)
    ]
    if mismatches:
        supplied = {field: getattr(value, field, None) for field in fields}
        raise ValueError(
            "Platform lock disagrees with the current official target receipt "
            f"for fields {mismatches}: {supplied!r}"
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
        adapter = adapter_for_target(
            str(raw.get("minecraft_version", "")),
            str(raw.get("loader", "fabric")),
        )
        for field in (
            "minecraft_version", "loader", "java_version", "yarn_mappings",
            "fabric_loader", "fabric_api", "fabric_loom", "gradle",
        ):
            if raw.get(field) != getattr(adapter, field):
                raise ValueError(
                    f"Generated platform lock disagrees with official target: {field}"
                )
        return adapter

    properties = _read_gradle_properties(root / "gradle.properties")
    minecraft_version = properties.get("minecraft_version", "")
    loader = properties.get("loader", "fabric") or "fabric"
    adapter = adapter_for_target(minecraft_version, loader)
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
                f"Project Gradle property {key} disagrees with official target discovery."
            )
    return adapter


def platform_catalog_receipt() -> dict[str, Any]:
    versions = supported_minecraft_versions(loader="fabric")
    return {
        "schema_version": "mmm/live-platform-discovery-v1",
        "provider": "official_fabric_meta_maven_and_template",
        "minecraft_versions": list(versions),
        "offline_compatibility_seeds": [item.public_dict() for item in PLATFORM_ADAPTERS],
    }


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


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")
