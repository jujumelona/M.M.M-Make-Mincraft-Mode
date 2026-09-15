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
    values = []
    if raw is not None:
        values.append(_text(raw.get("version")) if isinstance(raw, Mapping) else _text(raw))
    values.extend(_text(target[key]) for key in ("mappings_version", "yarn_mappings") if key in target)
    if len(set(values)) > 1:
        raise TargetContractError("TARGET_MAPPINGS_ALIAS: mapping aliases disagree")
    return values[0] if values else ""



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


_BASE_REQUIRED_TARGET_FIELDS = (
    "minecraft_version",
    "loader",
    "java_version",
    "fabric_loader",
    "fabric_api",
    "fabric_loom",
    "gradle",
    "gradle_sha256",
    "data_pack_version",
    "resource_pack_version",
    "resource_pack_format",
    "release_metadata_url",
)
_LEGACY_MAPPING_FIELDS = ("mappings_kind", "mappings_version")


def _is_unresolved(value: Any) -> bool:
    return not _text(value) or _text(value).casefold() == "unresolved"


def required_target_fields(coordinates: Mapping[str, Any]) -> tuple[str, ...]:
    version = coordinates.get("minecraft_version")
    if _is_unresolved(version):
        return _BASE_REQUIRED_TARGET_FIELDS
    try:
        mapping_required = mappings_applicable(version)
    except TargetContractError as exc:
        raise TargetContractError(str(exc)) from exc
    return _BASE_REQUIRED_TARGET_FIELDS + (_LEGACY_MAPPING_FIELDS if mapping_required else ())


def validate_complete_target(coordinates: Mapping[str, Any]) -> dict[str, Any]:
    required_fields = required_target_fields(coordinates)
    missing = [field for field in required_fields if _is_unresolved(coordinates.get(field))]
    if missing:
        raise TargetContractError(
            "TARGET_GROUNDING_INCOMPLETE: executable provider target is missing "
            + ", ".join(missing)
        )

    try:
        canonical = target_coordinates_from_mapping(coordinates)
    except TargetContractError as exc:
        raise TargetContractError(str(exc)) from exc

    minimum = canonical.minimum_java_major
    java = _text(coordinates.get("java_version"))
    if minimum is not None and (not java.isdigit() or int(java) < minimum):
        raise TargetContractError(f"Minecraft {canonical.minecraft_version} requires Java {minimum}+; got {java}.")

    mappings_receipt: dict[str, str] | None = None
    if canonical.mappings_applicable:
        mappings_kind = _text(coordinates.get("mappings_kind")).casefold()
        mappings_version = _text(coordinates.get("mappings_version"))
        if mappings_kind not in {"mojang", "yarn"}:
            raise TargetContractError(
                f"TARGET_MAPPINGS_KIND: unsupported mappings kind {mappings_kind!r}."
            )
        if canonical.mappings != mappings_version:
            raise TargetContractError(
                "TARGET_MAPPINGS_ALIAS: canonical mapping coordinate disagrees with mappings_version."
            )
        if mappings_kind == "mojang" and mappings_version != "mojang":
            raise TargetContractError("Mojang mappings must use canonical mappings_version='mojang'.")
        mappings_receipt = {"kind": mappings_kind, "version": canonical.mappings}

    gradle_sha = _text(coordinates.get("gradle_sha256")).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", gradle_sha):
        raise TargetContractError(
            "TARGET_GRADLE_RECEIPT: target Gradle SHA-256 is missing or invalid."
        )

    data_pack = _text(coordinates.get("data_pack_version"))
    resource_pack = _text(coordinates.get("resource_pack_version"))
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", data_pack):
        raise TargetContractError(
            f"TARGET_DATA_PACK_VERSION: invalid data pack version {data_pack!r}."
        )
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", resource_pack):
        raise TargetContractError(
            f"TARGET_RESOURCE_PACK_VERSION: invalid resource pack version {resource_pack!r}."
        )
    resource_format = coordinates.get("resource_pack_format")
    if type(resource_format) is not int or resource_format <= 0:
        raise TargetContractError(
            "TARGET_RESOURCE_PACK_FORMAT: provider-derived format must be a positive integer."
        )
    if resource_format != int(resource_pack.split(".", 1)[0]):
        raise TargetContractError(
            "TARGET_RESOURCE_PACK_FORMAT: format major disagrees with exact resource pack version."
        )
    release_url = _text(coordinates.get("release_metadata_url"))
    if not release_url.startswith(
        (
            "https://www.minecraft.net/",
            "https://feedback.minecraft.net/",
            "https://piston-meta.mojang.com/",
            "https://launcher.mojang.com/",
        )
    ):
        raise TargetContractError(
            "TARGET_PACK_PROVENANCE: pack metadata is not grounded in an official Minecraft/Mojang metadata URL."
        )

    result = dict(coordinates)
    if mappings_receipt is None:
        for field in (*_LEGACY_MAPPING_FIELDS, "yarn_mappings", "mappings"):
            result.pop(field, None)
    else:
        result["mappings"] = mappings_receipt
    result["naming_regime"] = {
        "kind": canonical.naming_regime,
        "mappings_applicable": canonical.mappings_applicable,
        "minecraft_version": canonical.minecraft_version,
    }
    result["pack_versions"] = {
        "data": data_pack,
        "resource": resource_pack,
        "resource_major": resource_format,
    }
    result["target_schema_version"] = "3"
    return result


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
    host_facts_json: str = ""

    @property
    def version_context(self):
        from .resolved_version_context import ResolvedVersionContext

        return ResolvedVersionContext.from_target(self)

    @property
    def mappings_applicable(self) -> bool:
        return mappings_applicable(self.minecraft_version)

    def validate(self) -> None:
        validate_complete_target(asdict(self))
        for field in ("adapter_id", "edition", "source_api_family"):
            if _is_unresolved(getattr(self, field)):
                raise TargetContractError(f"TARGET_PROVIDER: missing {field}")

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
    "validate_complete_target",
    "required_target_fields",
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
