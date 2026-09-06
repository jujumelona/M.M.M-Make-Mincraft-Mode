from __future__ import annotations

"""Single source of truth for Minecraft target coordinate semantics.

Provider discovery owns *which* exact Fabric/Loader/API coordinates exist.  This module
owns the cross-stage contract for the target identity itself: Minecraft version, loader,
naming regime, mappings applicability/value, and minimum Java semantics.  Planning,
generation, and coder handoff must call this module instead of rebuilding those rules.
"""

import re
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any

_NATIVE_NAME_MIN_VERSION = (26, 1)
_JAVA_21_MIN_VERSION = (1, 20, 5)


class TargetContractError(ValueError):
    """Raised when target coordinates contradict the canonical target contract."""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def minecraft_version_tuple(value: Any) -> tuple[int, ...]:
    text = _text(value)
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:[-+][A-Za-z0-9._-]+)?", text)
    if not match:
        raise TargetContractError(
            f"TARGET_MINECRAFT_VERSION: unsupported or unparseable Minecraft version {text!r}."
        )
    parts = [int(match.group(1)), int(match.group(2))]
    if match.group(3) is not None:
        parts.append(int(match.group(3)))
    return tuple(parts)


def uses_native_names(value: Any) -> bool:
    return minecraft_version_tuple(value)[:2] >= _NATIVE_NAME_MIN_VERSION


def mappings_applicable(value: Any) -> bool:
    return not uses_native_names(value)


def minimum_java_major(value: Any) -> int | None:
    version = minecraft_version_tuple(value)
    if version[:2] >= _NATIVE_NAME_MIN_VERSION:
        return 25
    if version and version[0] == 1:
        padded = version + (0,) * (3 - len(version))
        if padded[:3] >= _JAVA_21_MIN_VERSION:
            return 21
    return None


def mapping_value(target: Mapping[str, Any]) -> str:
    """Read the canonical mapping coordinate from a target-shaped mapping.

    ``mappings`` may be either the canonical scalar used at generation boundaries or the
    provider receipt object ``{"kind": ..., "version": ...}``.  Legacy aliases are read
    only to ingest existing provider receipts; callers must not invent a mapping from an
    unrelated field such as ``source_api_family``.
    """

    raw = target.get("mappings")
    if isinstance(raw, Mapping):
        return _text(raw.get("version"))
    if raw is not None:
        return _text(raw)
    return _text(target.get("mappings_version") or target.get("yarn_mappings"))


@dataclass(frozen=True)
class TargetCoordinates:
    minecraft_version: str
    loader: str
    mappings: str
    mappings_applicable: bool
    naming_regime: str
    minimum_java_major: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_target_coordinates(
    minecraft_version: Any,
    loader: Any,
    mappings: Any = "",
    *,
    declared_mappings_applicable: Any | None = None,
) -> TargetCoordinates:
    """Validate the target identity used by every planning/generation boundary.

    Mappings are required iff the Minecraft target uses mapped/obfuscated names.  Native
    targets require an empty mapping coordinate.  A caller-provided applicability flag is
    accepted only when it agrees with the version-derived canonical fact.
    """

    version = _text(minecraft_version)
    loader_id = _text(loader).casefold()
    mapping = _text(mappings)
    if not version:
        raise TargetContractError("TARGET_MINECRAFT_VERSION: Minecraft version is required.")
    if not loader_id:
        raise TargetContractError("TARGET_LOADER: loader is required.")

    applicable = mappings_applicable(version)
    if declared_mappings_applicable is not None:
        if type(declared_mappings_applicable) is not bool:
            raise TargetContractError(
                "TARGET_MAPPINGS_APPLICABLE: declared applicability must be boolean."
            )
        if declared_mappings_applicable is not applicable:
            raise TargetContractError(
                "TARGET_MAPPINGS_APPLICABLE: declared applicability contradicts the "
                "Minecraft-version naming regime."
            )

    if applicable and not mapping:
        raise TargetContractError(
            "TARGET_MAPPINGS_REQUIRED: mappings are required for this mapped/obfuscated target."
        )
    if not applicable and mapping:
        raise TargetContractError(
            "TARGET_MAPPINGS_INAPPLICABLE: native/unobfuscated targets must not carry a "
            "legacy mapping coordinate."
        )

    return TargetCoordinates(
        minecraft_version=version,
        loader=loader_id,
        mappings=mapping,
        mappings_applicable=applicable,
        naming_regime="mapped_obfuscated" if applicable else "native_unobfuscated",
        minimum_java_major=minimum_java_major(version),
    )


def target_coordinates_from_mapping(target: Mapping[str, Any]) -> TargetCoordinates:
    regime = target.get("naming_regime")
    declared: Any | None = None
    if isinstance(regime, Mapping) and "mappings_applicable" in regime:
        declared = regime.get("mappings_applicable")
    elif "mappings_applicable" in target:
        declared = target.get("mappings_applicable")
    return validate_target_coordinates(
        target.get("minecraft_version"),
        target.get("loader"),
        mapping_value(target),
        declared_mappings_applicable=declared,
    )


@dataclass(frozen=True)
class TargetContract:
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
        return mappings_applicable(self.minecraft_version)

    def validate(self) -> None:
        validate_target_coordinates(self.minecraft_version, self.loader, self.mappings_version)
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
        minimum_java = minimum_java_major(self.minecraft_version)
        if minimum_java is not None:
            if not str(self.java_version).isdigit() or int(self.java_version) < minimum_java:
                raise ValueError(
                    f"Minecraft {self.minecraft_version} requires Java {minimum_java}+; "
                    f"got {self.java_version}."
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


def target_contract_from_mapping(value: Mapping[str, Any]) -> TargetContract:
    """Deserialize a provider receipt through the same validation as live discovery."""
    from dataclasses import fields

    coordinates = target_coordinates_from_mapping(value)
    raw = {field.name: value[field.name] for field in fields(TargetContract) if field.name in value}
    mapping = value.get("mappings")
    kind = mapping.get("kind", "") if isinstance(mapping, Mapping) else value.get("mappings_kind", "")
    raw.setdefault("mappings_kind", kind)
    raw.setdefault("mappings_version", coordinates.mappings)
    raw.setdefault("yarn_mappings", coordinates.mappings)
    raw["deterministic_module_kinds"] = frozenset(raw.get("deterministic_module_kinds", ()))
    try:
        contract = TargetContract(**raw)
    except TypeError as exc:
        raise TargetContractError("TARGET_RECEIPT: incomplete provider receipt") from exc
    contract.validate()
    # Validate derived receipt fields too: a contradictory public view is never authority.
    canonical = contract.public_dict()
    for key in ("naming_regime", "pack_versions", "mappings"):
        if key in value and value[key] != canonical.get(key):
            raise TargetContractError(f"TARGET_RECEIPT: contradictory {key}")
    return contract


__all__ = [
    "TargetContract",
    "target_contract_from_mapping",
    "TargetContractError",
    "TargetCoordinates",
    "mapping_value",
    "mappings_applicable",
    "minecraft_version_tuple",
    "minimum_java_major",
    "target_coordinates_from_mapping",
    "uses_native_names",
    "validate_target_coordinates",
]
