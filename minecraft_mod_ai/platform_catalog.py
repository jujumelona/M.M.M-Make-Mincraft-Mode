from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


# Platform adapters are code-owned compatibility envelopes. A planner may select one,
# but it may not invent version tuples. Every downstream generator/build/validator
# consumes the same exact tuple.
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


_GRADLE_8_12_SHA256 = "7a00d51fb93147819aab76024feece20b6b84e420694101f276be952e08bef03"

# The 1.20.1 adapter preserves the already exercised MMM source API family.
# Loom 1.10.x is a stable release line paired with Gradle 8.12. The loader/API
# coordinates remain exact so generated metadata never advertises a broader target
# than the project actually builds against.
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

# 1.21.1 is a separate source API family (Java 21, Identifier.of/Item.Settings,
# resource-pack format 34). MMM currently enables deterministic bootstrap generation
# only for the source shapes explicitly adapted below. More complex requested systems
# stay on the mature 1.20.1 adapter unless the user explicitly requests 1.21.1, in
# which case unsupported deterministic coverage fails closed instead of silently
# producing 1.20.1 source under a 1.21.1 label.
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
_BY_ID = {item.adapter_id: item for item in PLATFORM_ADAPTERS}
_BY_TARGET = {(item.loader, item.minecraft_version): item for item in PLATFORM_ADAPTERS}


def supported_minecraft_versions(*, loader: str = "fabric") -> tuple[str, ...]:
    values = [
        item.minecraft_version
        for item in PLATFORM_ADAPTERS
        if item.loader == loader
    ]
    return tuple(sorted(values, key=_version_key))


def newest_adapter(*, loader: str = "fabric") -> PlatformAdapter:
    candidates = [item for item in PLATFORM_ADAPTERS if item.loader == loader]
    if not candidates:
        raise ValueError(f"No reviewed platform adapter for loader={loader!r}.")
    return max(candidates, key=lambda item: _version_key(item.minecraft_version))


def adapter_for_target(minecraft_version: str, loader: str = "fabric") -> PlatformAdapter:
    key = (loader.strip().lower(), minecraft_version.strip())
    adapter = _BY_TARGET.get(key)
    if adapter is None:
        supported = ", ".join(supported_minecraft_versions(loader=key[0])) or "none"
        raise ValueError(
            f"No reviewed platform adapter for {key[0]} Minecraft {key[1]}; "
            f"supported versions: {supported}."
        )
    return adapter


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
    matches = [
        adapter
        for adapter in PLATFORM_ADAPTERS
        if all(getattr(value, field) == getattr(adapter, field) for field in fields)
    ]
    if len(matches) != 1:
        supplied = {field: getattr(value, field, None) for field in fields}
        raise ValueError(
            "Platform lock is not one exact reviewed adapter tuple: " + repr(supplied)
        )
    return matches[0]


def adapter_from_project(project_root: str | Path) -> PlatformAdapter:
    root = Path(project_root).expanduser().resolve()
    lock_file = root / ".minecraft_ai" / "platform-lock.json"
    if lock_file.is_file() and not lock_file.is_symlink():
        import json

        raw = json.loads(lock_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Generated platform lock must be an object.")
        adapter_id = raw.get("adapter_id")
        adapter = _BY_ID.get(str(adapter_id))
        if adapter is None:
            raise ValueError(f"Unknown generated platform adapter: {adapter_id!r}")
        for field in (
            "minecraft_version", "loader", "java_version", "yarn_mappings",
            "fabric_loader", "fabric_api", "fabric_loom", "gradle",
        ):
            if raw.get(field) != getattr(adapter, field):
                raise ValueError(
                    f"Generated platform lock disagrees with adapter {adapter.adapter_id}: {field}"
                )
        return adapter

    properties = _read_gradle_properties(root / "gradle.properties")
    minecraft_version = properties.get("minecraft_version", "")
    loader = properties.get("loader", "fabric") or "fabric"
    adapter = adapter_for_target(minecraft_version, loader)
    expected = {
        "yarn_mappings": adapter.yarn_mappings,
        "loader_version": adapter.fabric_loader,
        "fabric_version": adapter.fabric_api,
        "loom_version": adapter.fabric_loom,
    }
    for key, value in expected.items():
        if properties.get(key) != value:
            raise ValueError(
                f"Project Gradle property {key} does not match reviewed adapter {adapter.adapter_id}."
            )
    return adapter


def platform_catalog_receipt() -> dict[str, Any]:
    return {
        "schema_version": "mmm/platform-adapter-catalog-v1",
        "adapters": [item.public_dict() for item in PLATFORM_ADAPTERS],
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


def _version_key(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)
