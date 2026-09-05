from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from .json_stream import canonical_json_sha256

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
PACKAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLATFORM_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]*$")
_PLATFORM_RECEIPT_HASH_FIELDS = (
    "adapter_id",
    "edition",
    "loader",
    "minecraft_version",
    "java_version",
    "yarn_mappings",
    "mappings_kind",
    "mappings_version",
    "fabric_loader",
    "fabric_api",
    "fabric_loom",
    "gradle",
    "gradle_sha256",
    "gradle_distribution_url",
    "data_pack_version",
    "resource_pack_version",
    "resource_pack_format",
    "release_metadata_url",
    "source_api_family",
    "deterministic_module_kinds",
)
_PLATFORM_EXTENDED_FIELDS = (
    "adapter_id",
    "mappings_kind",
    "mappings_version",
    "gradle_sha256",
    "gradle_distribution_url",
    "data_pack_version",
    "resource_pack_version",
    "resource_pack_format",
    "release_metadata_url",
    "source_api_family",
    "receipt_sha256",
)
_PLATFORM_MAPPING_FIELDS = ("mappings_kind", "mappings_version")
_NATIVE_NAME_MIN_VERSION = (26, 1)
JAVA_RESERVED_WORDS = frozenset(
    {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
        "class", "const", "continue", "default", "do", "double", "else", "enum",
        "exports", "extends", "final", "finally", "float", "for", "goto", "if",
        "implements", "import", "instanceof", "int", "interface", "long", "module",
        "native", "new", "non-sealed", "open", "opens", "package", "permits",
        "private", "protected", "provides", "public", "record", "requires", "return",
        "sealed", "short", "static", "strictfp", "super", "switch", "synchronized",
        "this", "throw", "throws", "to", "transient", "transitive", "true", "try",
        "uses", "var", "void", "volatile", "while", "with", "yield", "false", "null",
    }
)
RESERVED_MOD_IDS = frozenset({"minecraft", "fabric", "fabricloader", "java"})
RESERVED_GENERATED_FIELDS = frozenset({"MOD_ID", "LOGGER"})
RESERVED_PACKAGE_PREFIXES = (
    "java.",
    "javax.",
    "jdk.",
    "sun.",
    "net.minecraft.",
    "com.mojang.",
    "net.fabricmc.",
)
BOSS_MODEL_KINDS = frozenset(
    {
        "biped_blockbench",
        "quadruped_blockbench",
        "flying_blockbench",
        "serpentine_blockbench",
        "construct_blockbench",
        "custom_geckolib",
    }
)


class SpecValidationError(ValueError):
    """Raised when a proposal or mod specification violates the contract."""


class ContentKind(str, Enum):
    ITEM = "item"
    BLOCK = "block"


class ProposalStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_user_approval"
    APPROVED = "approved"


def _validate_platform_token(field_name: str, value: str) -> None:
    text = str(value)
    if not text or not _PLATFORM_TOKEN_PATTERN.fullmatch(text):
        raise SpecValidationError(
            f"Platform lock {field_name} is not a canonical provider coordinate."
        )


def _platform_receipt_value(value: Any, field_name: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def platform_receipt_payload(value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _PLATFORM_RECEIPT_HASH_FIELDS:
        item = _platform_receipt_value(value, field_name, None)
        if field_name == "deterministic_module_kinds":
            item = sorted(str(entry) for entry in (item or ()))
        payload[field_name] = item
    return payload


def platform_receipt_sha256(value: Any) -> str:
    return canonical_json_sha256(platform_receipt_payload(value))


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = str(value or "").split(".")
    if not parts or not all(part.isdigit() for part in parts):
        return ()
    return tuple(int(part) for part in parts)


def _uses_native_names(value: str) -> bool:
    version = _version_tuple(value)
    return len(version) >= 2 and version[:2] >= _NATIVE_NAME_MIN_VERSION


@dataclass(frozen=True)
class PlatformLock:
    """Immutable coordinates copied from one already-validated provider receipt.

    Provider/network validation belongs to platform resolution. This value object is
    intentionally validated offline so proposal hashing, persistence, loading, approval,
    and deterministic validation cannot rediscover the target or depend on network state.
    """

    edition: str = ""
    loader: str = ""
    minecraft_version: str = ""
    java_version: str = ""
    yarn_mappings: str = ""
    fabric_loader: str = ""
    fabric_api: str = ""
    fabric_loom: str = ""
    gradle: str = ""
    adapter_id: str = ""
    mappings_kind: str = ""
    mappings_version: str = ""
    gradle_sha256: str = ""
    gradle_distribution_url: str = ""
    data_pack_version: str = ""
    resource_pack_version: str = ""
    resource_pack_format: int = 0
    release_metadata_url: str = ""
    source_api_family: str = ""
    deterministic_module_kinds: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def is_unresolved(self) -> bool:
        return not any(
            (
                self.edition,
                self.loader,
                self.minecraft_version,
                self.java_version,
                self.yarn_mappings,
                self.fabric_loader,
                self.fabric_api,
                self.fabric_loom,
                self.gradle,
            )
        )

    def has_full_execution_receipt(self) -> bool:
        required = tuple(
            field_name
            for field_name in _PLATFORM_EXTENDED_FIELDS
            if field_name not in _PLATFORM_MAPPING_FIELDS
        )
        if not all(
            _platform_receipt_value(self, field_name, None) not in (None, "", 0)
            for field_name in required
        ):
            return False
        if _uses_native_names(self.minecraft_version):
            return not any(
                str(_platform_receipt_value(self, field_name, "") or "").strip()
                for field_name in (*_PLATFORM_MAPPING_FIELDS, "yarn_mappings")
            )
        return all(
            _platform_receipt_value(self, field_name, None) not in (None, "", 0)
            for field_name in _PLATFORM_MAPPING_FIELDS
        ) and bool(self.yarn_mappings.strip())

    def validate(self) -> None:
        native_names = _uses_native_names(self.minecraft_version)
        coordinates = {
            "edition": self.edition,
            "loader": self.loader,
            "minecraft_version": self.minecraft_version,
            "java_version": self.java_version,
            "fabric_loader": self.fabric_loader,
            "fabric_api": self.fabric_api,
            "fabric_loom": self.fabric_loom,
            "gradle": self.gradle,
        }
        if not native_names:
            coordinates["yarn_mappings"] = self.yarn_mappings
        if not any(coordinates.values()):
            raise SpecValidationError(
                "Platform target is unresolved. Resolve one executable provider receipt "
                "before approval or generation."
            )
        if not all(str(value).strip() for value in coordinates.values()):
            raise SpecValidationError(
                "Platform target is partial. A platform lock must be either fully "
                "unresolved during planning or a complete provider receipt."
            )
        for field_name, value in coordinates.items():
            _validate_platform_token(field_name, str(value))
        if self.edition != self.edition.casefold():
            raise SpecValidationError("Platform lock edition must be canonical lowercase.")
        if self.loader != self.loader.casefold():
            raise SpecValidationError("Platform lock loader must be canonical lowercase.")
        if not self.java_version.isdigit() or int(self.java_version) <= 0:
            raise SpecValidationError(
                "Platform lock java_version must be a positive Java major version."
            )

        extended_present = any(
            _platform_receipt_value(self, field_name, None) not in (None, "", 0)
            for field_name in _PLATFORM_EXTENDED_FIELDS
        ) or bool(self.deterministic_module_kinds)
        if not extended_present:
            return
        if not self.has_full_execution_receipt():
            raise SpecValidationError(
                "Execution platform receipt is partial or incompatible with the target naming regime."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.gradle_sha256.casefold()):
            raise SpecValidationError("Platform lock gradle_sha256 must be a full SHA-256.")
        if not SHA256_PATTERN.fullmatch(self.receipt_sha256.casefold()):
            raise SpecValidationError(
                "Platform lock receipt_sha256 must be a lowercase SHA-256 receipt."
            )
        if self.receipt_sha256.casefold() != platform_receipt_sha256(self):
            raise SpecValidationError(
                "Platform lock receipt SHA-256 does not match its immutable provider coordinates."
            )
        expected_gradle_url = (
            f"https://services.gradle.org/distributions/gradle-{self.gradle}-bin.zip"
        )
        if self.gradle_distribution_url != expected_gradle_url:
            raise SpecValidationError(
                "Platform lock Gradle distribution URL disagrees with the pinned version."
            )
        parsed_gradle = urlparse(self.gradle_distribution_url)
        if parsed_gradle.scheme != "https" or parsed_gradle.hostname != "services.gradle.org":
            raise SpecValidationError(
                "Platform lock Gradle distribution URL must use the official HTTPS host."
            )
        if native_names:
            if any((self.yarn_mappings, self.mappings_kind, self.mappings_version)):
                raise SpecValidationError(
                    "Minecraft 26.1+ native/unobfuscated targets must not carry legacy mapping coordinates."
                )
        else:
            if self.mappings_kind not in {"mojang", "yarn"}:
                raise SpecValidationError("Platform lock mappings_kind must be mojang or yarn.")
            if self.mappings_version != self.yarn_mappings:
                raise SpecValidationError(
                    "Platform lock mappings_version must equal the executable mappings coordinate."
                )
        if any(not str(item).strip() for item in self.deterministic_module_kinds):
            raise SpecValidationError(
                "Platform lock deterministic_module_kinds contains an empty capability."
            )
        if len(set(self.deterministic_module_kinds)) != len(self.deterministic_module_kinds):
            raise SpecValidationError(
                "Platform lock deterministic_module_kinds must not contain duplicates."
            )
        if type(self.resource_pack_format) is not int or self.resource_pack_format <= 0:
            raise SpecValidationError(
                "Platform lock resource_pack_format must be a positive integer."
            )
        try:
            resource_major = int(self.resource_pack_version.split(".", 1)[0])
        except ValueError as exc:
            raise SpecValidationError(
                "Platform lock resource_pack_version must start with a numeric major."
            ) from exc
        if resource_major != self.resource_pack_format:
            raise SpecValidationError(
                "Platform lock resource pack format disagrees with resource_pack_version."
            )
        release = urlparse(self.release_metadata_url)
        if release.scheme != "https" or release.hostname not in {
            "www.minecraft.net",
            "feedback.minecraft.net",
            "piston-meta.mojang.com",
            "launcher.mojang.com",
        }:
            raise SpecValidationError(
                "Platform lock release metadata must be an official Minecraft/Mojang HTTPS URL."
            )
        mc_tuple = _version_tuple(self.minecraft_version)
        if mc_tuple and mc_tuple[0] == 1 and mc_tuple >= (1, 20, 5) and int(self.java_version) < 21:
            raise SpecValidationError(
                f"Minecraft {self.minecraft_version} requires a Java 21+ execution target; "
                f"got Java {self.java_version}."
            )
        if native_names and int(self.java_version) < 25:
            raise SpecValidationError(
                f"Minecraft {self.minecraft_version} native-name target requires Java 25+; "
                f"got Java {self.java_version}."
            )
        if self.mappings_kind == "yarn":
            mapping_match = re.match(r"^(\d+(?:\.\d+){1,2})\+", self.yarn_mappings)
            if mapping_match and mapping_match.group(1) != self.minecraft_version:
                raise SpecValidationError(
                    "Platform lock mappings coordinate disagrees with minecraft_version."
                )
        api_match = re.search(r"\+(\d+(?:\.\d+){1,2})$", self.fabric_api)
        if api_match and api_match.group(1) != self.minecraft_version:
            raise SpecValidationError(
                "Platform lock Fabric API coordinate disagrees with minecraft_version."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContentSpec:
    content_id: str
    kind: ContentKind
    display_name_en: str
    display_name_ko: str
    color: str = "#74c7ec"
    recipe: bool = True

    def validate(self) -> None:
        if type(self.recipe) is not bool:
            raise SpecValidationError(
                f"recipe must be a JSON boolean for {self.content_id}."
            )
        if not ID_PATTERN.fullmatch(self.content_id):
            raise SpecValidationError(
                f"Invalid content_id {self.content_id!r}; use lowercase snake_case."
            )
        if len(self.display_name_en.strip()) < 2:
            raise SpecValidationError(
                f"English display name is missing for {self.content_id}."
            )
        if len(self.display_name_ko.strip()) < 1:
            raise SpecValidationError(
                f"Korean display name is missing for {self.content_id}."
            )
        if not HEX_COLOR_PATTERN.fullmatch(self.color):
            raise SpecValidationError(
                f"Invalid RGB color {self.color!r} for {self.content_id}."
            )


@dataclass(frozen=True)
class BossSpec:
    entity_id: str
    display_name_en: str
    display_name_ko: str
    max_health: float = 160.0
    attack_damage: float = 10.0
    movement_speed: float = 0.28
    scale: float = 1.35
    primary_color: str = "#89dceb"
    secondary_color: str = "#cba6f7"
    model_kind: str = "biped_blockbench"

    def validate(self) -> None:
        numeric_fields = {
            "max_health": self.max_health,
            "attack_damage": self.attack_damage,
            "movement_speed": self.movement_speed,
            "scale": self.scale,
        }
        for field_name, value in numeric_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise SpecValidationError(
                    f"Boss {field_name} must be a positive finite JSON number."
                )
        if not ID_PATTERN.fullmatch(self.entity_id):
            raise SpecValidationError(
                f"Invalid boss entity_id: {self.entity_id!r}."
            )
        if self.model_kind not in BOSS_MODEL_KINDS:
            raise SpecValidationError(
                f"Unsupported boss model_kind {self.model_kind!r}; "
                f"use one of {sorted(BOSS_MODEL_KINDS)}."
            )
        for color in (self.primary_color, self.secondary_color):
            if not HEX_COLOR_PATTERN.fullmatch(color):
                raise SpecValidationError(f"Invalid boss color: {color!r}.")


@dataclass(frozen=True)
class ModSpec:
    mod_id: str
    mod_name: str
    package_name: str
    version: str
    summary: str
    contents: tuple[ContentSpec, ...]
    boss: BossSpec | None = None
    platform: PlatformLock = field(default_factory=PlatformLock)

    def validate(self) -> None:
        self.platform.validate()
        if not ID_PATTERN.fullmatch(self.mod_id):
            raise SpecValidationError(
                f"Invalid mod_id {self.mod_id!r}; use lowercase snake_case."
            )
        if self.mod_id in RESERVED_MOD_IDS:
            raise SpecValidationError(
                f"Reserved mod_id is not allowed: {self.mod_id}"
            )
        if not PACKAGE_PATTERN.fullmatch(self.package_name):
            raise SpecValidationError(
                f"Invalid package_name {self.package_name!r}; "
                "use a lowercase dotted Java package."
            )
        if self.package_name.startswith(RESERVED_PACKAGE_PREFIXES):
            raise SpecValidationError(
                f"Reserved platform package prefix is not allowed: {self.package_name}"
            )
        reserved_components = set(self.package_name.split(".")) & JAVA_RESERVED_WORDS
        if reserved_components:
            raise SpecValidationError(
                f"Java reserved package component is not allowed: "
                f"{sorted(reserved_components)}"
            )
        if not re.fullmatch(
            r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]+)?", self.version
        ):
            raise SpecValidationError(
                f"Invalid semantic version {self.version!r}."
            )
        seen: set[str] = set()
        for content in self.contents:
            content.validate()
            if content.content_id in seen:
                raise SpecValidationError(
                    f"Duplicate content_id: {content.content_id}"
                )
            if content.content_id.upper() in RESERVED_GENERATED_FIELDS:
                raise SpecValidationError(
                    "Content ID collides with a generated Java field: "
                    f"{content.content_id}"
                )
            seen.add(content.content_id)
        if self.boss is not None:
            self.boss.validate()
            if self.boss.entity_id in seen:
                raise SpecValidationError(
                    f"Boss ID collides with content ID: {self.boss.entity_id}"
                )
            if self.boss.entity_id.upper() in RESERVED_GENERATED_FIELDS:
                raise SpecValidationError(
                    "Boss ID collides with a generated Java field: "
                    f"{self.boss.entity_id}"
                )
            spawn_egg_id = f"{self.boss.entity_id}_spawn_egg"
            if spawn_egg_id in seen:
                raise SpecValidationError(
                    f"Boss spawn egg ID collides with content ID: {spawn_egg_id}"
                )
            seen.update((self.boss.entity_id, spawn_egg_id))


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    title: str
    url: str
    authority: str
    version_scope: str
    verified_on: str
    trust_tier: str = "official_primary"
    retrieval_policy: str = "data_only"
    record_sha256: str = ""


@dataclass(frozen=True)
class DeferredRequest:
    capability: str
    reason: str
    suggested_phase: str


@dataclass(frozen=True)
class Proposal:
    schema_version: str
    proposal_version: int
    status: ProposalStatus
    requested_prompt: str
    spec: ModSpec
    assumptions: tuple[str, ...]
    exclusions: tuple[str, ...]
    deferred_requests: tuple[DeferredRequest, ...]
    acceptance_tests: tuple[str, ...]
    evidence_sources: tuple[EvidenceSource, ...]
    evidence_snapshot_hash: str = ""
    capability_manifest_hash: str = ""
    imported_source_snapshot_hash: str = ""
    risk_approvals: tuple[str, ...] = ()
    approval_hash: str = ""

    _TOP_LEVEL_KEYS = frozenset(
        {
            "schema_version",
            "proposal_version",
            "status",
            "requested_prompt",
            "spec",
            "assumptions",
            "exclusions",
            "deferred_requests",
            "acceptance_tests",
            "evidence_sources",
            "evidence_snapshot_hash",
            "capability_manifest_hash",
            "imported_source_snapshot_hash",
            "risk_approvals",
            "approval_hash",
        }
    )
    _BACKWARD_COMPATIBLE_KEYS = frozenset(
        {
            "evidence_snapshot_hash",
            "capability_manifest_hash",
            "imported_source_snapshot_hash",
        }
    )

    def validate(self) -> None:
        if self.schema_version != "minecraft-mod-ai/proposal-v1":
            raise SpecValidationError(
                f"Unsupported schema_version: {self.schema_version}"
            )
        if type(self.proposal_version) is not int or self.proposal_version < 1:
            raise SpecValidationError(
                "proposal_version must be a positive JSON integer."
            )
        if not self.requested_prompt.strip():
            raise SpecValidationError("requested_prompt must not be empty.")
        self.spec.validate()
        if not self.acceptance_tests:
            raise SpecValidationError(
                "At least one acceptance test is required."
            )
        if not self.evidence_sources:
            raise SpecValidationError(
                "At least one authoritative evidence source is required."
            )
        from .knowledge import evidence_snapshot_hash, validate_trusted_evidence

        validate_trusted_evidence(
            self.evidence_sources,
            minecraft_version=self.spec.platform.minecraft_version,
        )
        if not SHA256_PATTERN.fullmatch(self.evidence_snapshot_hash):
            raise SpecValidationError(
                "evidence_snapshot_hash must be a lowercase sha256 digest."
            )
        if self.evidence_snapshot_hash != evidence_snapshot_hash(
            self.evidence_sources
        ):
            raise SpecValidationError(
                "evidence_snapshot_hash does not match the trusted evidence records."
            )
        if not SHA256_PATTERN.fullmatch(self.capability_manifest_hash):
            raise SpecValidationError(
                "capability_manifest_hash must be a lowercase sha256 digest."
            )
        if self.imported_source_snapshot_hash and not SHA256_PATTERN.fullmatch(
            self.imported_source_snapshot_hash
        ):
            raise SpecValidationError(
                "imported_source_snapshot_hash must be empty or a lowercase sha256 digest."
            )
        if self.approval_hash:
            if not SHA256_PATTERN.fullmatch(self.approval_hash):
                raise SpecValidationError(
                    "approval_hash must be a lowercase sha256 digest."
                )
            if self.approval_hash != self.calculate_hash():
                raise SpecValidationError(
                    "approval_hash does not match the immutable proposal payload."
                )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_version": self.proposal_version,
            "status": ProposalStatus.AWAITING_APPROVAL.value,
            "requested_prompt": self.requested_prompt,
            "spec": self.spec,
            "assumptions": self.assumptions,
            "exclusions": self.exclusions,
            "deferred_requests": self.deferred_requests,
            "acceptance_tests": self.acceptance_tests,
            "evidence_sources": self.evidence_sources,
            "evidence_snapshot_hash": self.evidence_snapshot_hash,
            "capability_manifest_hash": self.capability_manifest_hash,
            "imported_source_snapshot_hash": self.imported_source_snapshot_hash,
            "risk_approvals": self.risk_approvals,
            "approval_hash": "",
        }

    def calculate_hash(self) -> str:
        return canonical_json_sha256(self._hash_payload())

    def with_hash(self) -> Proposal:
        from .capabilities import capability_manifest_hash
        from .knowledge import evidence_snapshot_hash

        proposal = Proposal(
            schema_version=self.schema_version,
            proposal_version=self.proposal_version,
            status=ProposalStatus.AWAITING_APPROVAL,
            requested_prompt=self.requested_prompt,
            spec=self.spec,
            assumptions=self.assumptions,
            exclusions=self.exclusions,
            deferred_requests=self.deferred_requests,
            acceptance_tests=self.acceptance_tests,
            evidence_sources=self.evidence_sources,
            evidence_snapshot_hash=(
                self.evidence_snapshot_hash
                or evidence_snapshot_hash(self.evidence_sources)
            ),
            capability_manifest_hash=(
                self.capability_manifest_hash or capability_manifest_hash()
            ),
            imported_source_snapshot_hash=self.imported_source_snapshot_hash,
            risk_approvals=self.risk_approvals,
            approval_hash="",
        )
        return Proposal(
            **{
                **proposal.__dict__,
                "approval_hash": proposal.calculate_hash(),
            }
        )

    def approve(self, supplied_hash: str) -> Proposal:
        self.validate()
        expected = self.calculate_hash()
        if supplied_hash != expected:
            raise SpecValidationError(
                "Approval hash mismatch. The displayed proposal changed or "
                "the wrong hash was used."
            )
        return Proposal(**{**self.__dict__, "status": ProposalStatus.APPROVED})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        for content in data["spec"]["contents"]:
            content["kind"] = (
                content["kind"].value
                if isinstance(content["kind"], Enum)
                else content["kind"]
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        unknown = set(data) - cls._TOP_LEVEL_KEYS
        missing = cls._TOP_LEVEL_KEYS - cls._BACKWARD_COMPATIBLE_KEYS - set(data)
        if unknown:
            raise SpecValidationError(
                f"Unknown proposal fields: {sorted(unknown)}"
            )
        if missing:
            raise SpecValidationError(
                f"Missing proposal fields: {sorted(missing)}"
            )
        spec_data = dict(data["spec"])
        platform_data = spec_data.pop("platform")
        content_data = spec_data.pop("contents")
        boss_data = spec_data.pop("boss", None)
        arena_data = spec_data.pop("arena", None)
        if arena_data is not None:
            raise SpecValidationError()
        platform = PlatformLock(**platform_data)
        contents = tuple(
            ContentSpec(
                content_id=item["content_id"],
                kind=ContentKind(item["kind"]),
                display_name_en=item["display_name_en"],
                display_name_ko=item["display_name_ko"],
                color=item.get("color", "#74c7ec"),
                recipe=_json_bool(
                    item.get("recipe", True), "contents[].recipe"
                ),
            )
            for item in content_data
        )
        spec = ModSpec(
            contents=contents,
            boss=BossSpec(**boss_data) if boss_data else None,
            platform=platform,
            **spec_data,
        )
        evidence_sources = tuple(
            EvidenceSource(**item) for item in data["evidence_sources"]
        )
        from .capabilities import capability_manifest_hash
        from .knowledge import evidence_snapshot_hash

        proposal = cls(
            schema_version=data["schema_version"],
            proposal_version=data["proposal_version"],
            status=ProposalStatus(data["status"]),
            requested_prompt=data["requested_prompt"],
            spec=spec,
            assumptions=tuple(data["assumptions"]),
            exclusions=tuple(data["exclusions"]),
            deferred_requests=tuple(
                DeferredRequest(**item)
                for item in data["deferred_requests"]
            ),
            acceptance_tests=tuple(data["acceptance_tests"]),
            evidence_sources=evidence_sources,
            evidence_snapshot_hash=data.get(
                "evidence_snapshot_hash",
                evidence_snapshot_hash(evidence_sources),
            ),
            capability_manifest_hash=data.get(
                "capability_manifest_hash",
                capability_manifest_hash(),
            ),
            imported_source_snapshot_hash=data.get(
                "imported_source_snapshot_hash", ""
            ),
            risk_approvals=tuple(data["risk_approvals"]),
            approval_hash=data["approval_hash"],
        )
        proposal.validate()
        return proposal


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise SpecValidationError(
            f"{field_name} must be a JSON boolean."
        )
    return value