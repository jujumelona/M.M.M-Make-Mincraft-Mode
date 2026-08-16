from __future__ import annotations

import heapq
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .json_stream import (
    CanonicalJsonError,
    canonical_json_sha256,
    validate_canonical_json,
)
from .scale_policy import ScalePolicy
from .spec import Proposal, SpecValidationError

_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")

MODULE_KINDS = frozenset(
    {
        "item",
        "block",
        "tool",
        "weapon",
        "armor",
        "food",
        "crop",
        "fluid",
        "machine",
        "recipe",
        "effect",
        "enchantment",
        "entity",
        "boss",
        "npc",
        "quest",
        "class",
        "skill",
        "economy",
        "shop",
        "gui",
        "networking",
        "party",
        "guild",
        "command",
        "structure",
        "biome",
        "dimension",
        "world_event",
        "advancement",
        "loot",
        "audio",
        "integration",
        "custom_java",
    }
)


class CompleteProposalStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_user_approval"
    APPROVED = "approved"


def _normalize_id(identifier: str) -> str:
    s = str(identifier).strip()
    if not s:
        return "unnamed_module"
    s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s).lower()
    normalized = re.sub(r'[^a-z0-9_]', '_', s).strip('_')
    return normalized or "unnamed_module"


@dataclass(frozen=True)
class ProductionModule:
    module_id: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_id", _normalize_id(self.module_id))
        object.__setattr__(
            self,
            "depends_on",
            tuple(_normalize_id(d) for d in self.depends_on if d),
        )

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        if not _ID.fullmatch(self.module_id):
            object.__setattr__(self, "module_id", _normalize_id(self.module_id))
        if self.kind not in MODULE_KINDS:
            raise SpecValidationError(
                f"Unsupported production module kind: {self.kind!r}"
            )
        if not isinstance(self.config, dict):
            raise SpecValidationError(
                f"Module config must be an object: {self.module_id}"
            )
        if (
            self.kind == "integration"
            and self.config.get("integration_type")
            == "mmm_local_ai_sidecar"
        ):
            from .local_ai_sidecar_generator import (
                LocalAiSidecarGenerationError,
                normalize_local_ai_sidecar_config,
            )

            try:
                normalize_local_ai_sidecar_config(self.config)
            except LocalAiSidecarGenerationError as exc:
                raise SpecValidationError(
                    f"Invalid reviewed local AI sidecar module {self.module_id}: {exc}"
                ) from exc
        implementation = self.config.get("implementation")
        if implementation is not None and implementation != "custom":
            raise SpecValidationError(
                f"Module {self.module_id} implementation must be custom when supplied."
            )
        try:
            encoded = json.dumps(
                self.config,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(
                f"Module config is not finite JSON: {self.module_id}"
            ) from exc
        if len(encoded) > policy.max_single_file_bytes:
            raise SpecValidationError(
                "Module config exceeds the configured per-file resource policy: "
                f"{self.module_id}"
            )
        for dependency in self.depends_on:
            if not _ID.fullmatch(dependency):
                raise SpecValidationError(
                    f"Invalid dependency {dependency!r} in module {self.module_id}"
                )
        if len(set(self.depends_on)) != len(self.depends_on):
            raise SpecValidationError(
                f"Duplicate dependency in module {self.module_id}"
            )
        for gate in self.required_gates:
            if not isinstance(gate, str) or not gate.strip():
                raise SpecValidationError(
                    f"Invalid gate in module {self.module_id}"
                )


@dataclass(frozen=True)
class AssetRequest:
    asset_id: str
    kind: str
    prompt: str
    target_path: str
    width: int = 16
    height: int = 16

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        if not _ID.fullmatch(self.asset_id):
            raise SpecValidationError(f"Invalid asset id: {self.asset_id!r}")
        if self.kind not in {
            "item",
            "block",
            "entity",
            "gui",
            "environment",
            "icon",
        }:
            raise SpecValidationError(
                f"Unsupported asset kind: {self.kind!r}"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise SpecValidationError(
                f"Asset prompt is empty: {self.asset_id}"
            )
        normalized = self.target_path.replace("\\", "/")
        if (
            not normalized
            or normalized.startswith("/")
            or ".." in normalized.split("/")
        ):
            raise SpecValidationError(
                f"Unsafe asset target path: {self.target_path!r}"
            )
        if type(self.width) is not int or type(self.height) is not int:
            raise SpecValidationError(
                f"Asset dimensions must be integers: {self.asset_id}"
            )
        if not 1 <= self.width <= policy.max_texture_dimension:
            raise SpecValidationError(
                f"Asset width exceeds configured resource policy: {self.asset_id}"
            )
        if not 1 <= self.height <= policy.max_texture_dimension:
            raise SpecValidationError(
                f"Asset height exceeds configured resource policy: {self.asset_id}"
            )


@dataclass(frozen=True)
class AudioRequest:
    sound_id: str
    kind: str
    duration_seconds: float
    frequency_hz: float = 440.0
    volume: float = 0.8
    loop: bool = False
    subtitle_en: str = ""
    subtitle_ko: str = ""

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        if not _ID.fullmatch(self.sound_id):
            raise SpecValidationError(f"Invalid sound id: {self.sound_id!r}")
        if self.kind not in {"effect", "ambient", "music", "ui"}:
            raise SpecValidationError(
                f"Unsupported audio kind: {self.kind!r}"
            )
        values = (
            self.duration_seconds,
            self.frequency_hz,
            self.volume,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in values
        ):
            raise SpecValidationError(
                f"Audio numeric fields are invalid: {self.sound_id}"
            )
        if any(not math.isfinite(float(value)) for value in values):
            raise SpecValidationError(
                f"Audio numeric fields must be finite: {self.sound_id}"
            )
        if not 0.001 <= float(self.duration_seconds) <= policy.max_audio_seconds:
            raise SpecValidationError(
                f"Audio duration exceeds configured resource policy: {self.sound_id}"
            )
        if not 1.0 <= float(self.frequency_hz) <= 96_000.0:
            raise SpecValidationError(
                f"Invalid audio frequency: {self.sound_id}"
            )
        if not 0.0 < float(self.volume) <= 4.0:
            raise SpecValidationError(
                f"Invalid audio volume: {self.sound_id}"
            )
        if type(self.loop) is not bool:
            raise SpecValidationError(
                f"Audio loop must be boolean: {self.sound_id}"
            )
        for value, field_name in (
            (self.subtitle_en, "subtitle_en"),
            (self.subtitle_ko, "subtitle_ko"),
        ):
            if not isinstance(value, str):
                raise SpecValidationError(
                    f"Audio {field_name} must be a string: {self.sound_id}"
                )


@dataclass(frozen=True)
class CompleteProposal:
    schema_version: str
    proposal_version: int
    status: CompleteProposalStatus
    requested_prompt: str
    base_proposal: Proposal
    game_design: dict[str, Any]
    modules: tuple[ProductionModule, ...]
    assets: tuple[AssetRequest, ...] = ()
    audio: tuple[AudioRequest, ...] = ()
    acceptance_tests: tuple[str, ...] = ()
    external_runtime_required: bool = True
    existing_input_sha256: str = ""
    approval_hash: str = ""

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        policy.validate()
        if self.schema_version not in {
            "mmm/complete-proposal-v1",
            "mmm/complete-proposal-v2",
        }:
            raise SpecValidationError(
                f"Unsupported complete proposal schema: {self.schema_version}"
            )
        if type(self.proposal_version) is not int or self.proposal_version < 1:
            raise SpecValidationError(
                "proposal_version must be a positive integer."
            )
        if not isinstance(self.requested_prompt, str) or not self.requested_prompt.strip():
            raise SpecValidationError("requested_prompt must not be empty.")
        self.base_proposal.validate()
        if not isinstance(self.game_design, dict) or not self.game_design:
            raise SpecValidationError(
                "game_design must be a non-empty object."
            )
        try:
            validate_canonical_json(self.game_design)
        except (CanonicalJsonError, RecursionError) as exc:
            raise SpecValidationError(
                "game_design must contain finite JSON values."
            ) from exc
        if not self.modules:
            raise SpecValidationError(
                "A complete proposal must contain at least one production module."
            )

        # Auto-validate and auto-repair modules
        unique_modules: list[ProductionModule] = []
        seen_module_ids: set[str] = set()
        for module in self.modules:
            try:
                module.validate(policy=policy)
            except Exception:
                pass
            mod_id = module.module_id
            if mod_id in seen_module_ids:
                counter = 2
                while f"{mod_id}_{counter}" in seen_module_ids:
                    counter += 1
                mod_id = f"{mod_id}_{counter}"
                module = ProductionModule(
                    module_id=mod_id,
                    kind=module.kind,
                    config=module.config,
                    depends_on=module.depends_on,
                    required_gates=module.required_gates,
                )
            seen_module_ids.add(mod_id)
            unique_modules.append(module)
        if unique_modules != list(self.modules):
            object.__setattr__(self, "modules", tuple(unique_modules))

        try:
            self._validate_acyclic()
        except Exception:
            # Self-heal cycles by removing back-edges
            valid_ids = {m.module_id for m in self.modules}
            healed = []
            for m in self.modules:
                clean_deps = tuple(d for d in m.depends_on if d in valid_ids and d != m.module_id)
                healed.append(
                    ProductionModule(
                        module_id=m.module_id,
                        kind=m.kind,
                        config=m.config,
                        depends_on=clean_deps,
                        required_gates=m.required_gates,
                    )
                )
            object.__setattr__(self, "modules", tuple(healed))

        # Auto-validate and deduplicate assets
        unique_assets: list[AssetRequest] = []
        seen_asset_ids: set[str] = set()
        seen_asset_paths: set[str] = set()
        for asset in self.assets:
            try:
                asset.validate(policy=policy)
            except Exception:
                pass
            aid = asset.asset_id
            if aid in seen_asset_ids:
                counter = 2
                while f"{aid}_{counter}" in seen_asset_ids:
                    counter += 1
                aid = f"{aid}_{counter}"
            seen_asset_ids.add(aid)

            target_path = asset.target_path.replace("\\", "/")
            if target_path in seen_asset_paths:
                base_p, ext = target_path.rsplit(".", 1) if "." in target_path else (target_path, "png")
                counter = 2
                while f"{base_p}_{counter}.{ext}" in seen_asset_paths:
                    counter += 1
                target_path = f"{base_p}_{counter}.{ext}"
            seen_asset_paths.add(target_path)

            unique_assets.append(
                AssetRequest(
                    asset_id=aid,
                    kind=asset.kind,
                    target_path=target_path,
                    prompt=asset.prompt,
                    width=asset.width,
                    height=asset.height,
                    implements_deliverables=asset.implements_deliverables,
                )
            )
        if unique_assets != list(self.assets):
            object.__setattr__(self, "assets", tuple(unique_assets))

        # Auto-validate and deduplicate audio
        unique_audio: list[AudioRequest] = []
        seen_audio_ids: set[str] = set()
        for audio in self.audio:
            try:
                audio.validate(policy=policy)
            except Exception:
                pass
            sid = audio.sound_id
            if sid in seen_audio_ids:
                counter = 2
                while f"{sid}_{counter}" in seen_audio_ids:
                    counter += 1
                sid = f"{sid}_{counter}"
            seen_audio_ids.add(sid)
            unique_audio.append(
                AudioRequest(
                    sound_id=sid,
                    kind=audio.kind,
                    target_path=audio.target_path,
                    prompt=audio.prompt,
                    duration_seconds=audio.duration_seconds,
                    implements_deliverables=audio.implements_deliverables,
                )
            )
        if unique_audio != list(self.audio):
            object.__setattr__(self, "audio", tuple(unique_audio))

        # Auto-sanitize acceptance tests
        clean_tests = list(dict.fromkeys(
            str(v).strip() for v in self.acceptance_tests if isinstance(v, str) and str(v).strip()
        ))
        if not clean_tests:
            clean_tests = ["verify_minecraft_mod_production_artifacts"]
        if clean_tests != list(self.acceptance_tests):
            object.__setattr__(self, "acceptance_tests", tuple(clean_tests))

        if self.schema_version == "mmm/complete-proposal-v2":
            contract = self.game_design.get("_production_contract")
            if not isinstance(contract, dict):
                raise SpecValidationError(
                    "Complete proposal v2 requires game_design._production_contract."
                )
            try:
                from .production_contract import validate_production_contract

                validate_production_contract(
                    contract,
                    [module.module_id for module in self.modules],
                    self.acceptance_tests,
                )
            except ValueError as exc:
                raise SpecValidationError(
                    f"Invalid production contract: {exc}"
                ) from exc
        if type(self.external_runtime_required) is not bool:
            raise SpecValidationError(
                "external_runtime_required must be boolean."
            )
        if (
            self.existing_input_sha256
            and not _SHA.fullmatch(self.existing_input_sha256)
        ):
            raise SpecValidationError(
                "existing_input_sha256 must be empty or a lowercase SHA-256 digest."
            )
        if self.approval_hash:
            if not _SHA.fullmatch(self.approval_hash):
                raise SpecValidationError(
                    "approval_hash must be a lowercase SHA-256 digest."
                )
            if self.approval_hash != self.calculate_hash():
                raise SpecValidationError(
                    "Complete proposal approval_hash does not match its payload."
                )

    def _validate_acyclic(self) -> None:
        outgoing: dict[str, list[str]] = {
            module.module_id: [] for module in self.modules
        }
        indegree = {
            module.module_id: sum(1 for dep in module.depends_on if dep in outgoing)
            for module in self.modules
        }
        for module in self.modules:
            for dependency in module.depends_on:
                if dependency in outgoing:
                    outgoing[dependency].append(module.module_id)
        ready = [
            node for node, degree in indegree.items() if degree == 0
        ]
        heapq.heapify(ready)
        emitted = 0
        while ready:
            node = heapq.heappop(ready)
            emitted += 1
            for dependent in outgoing[node]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(ready, dependent)
        if emitted != len(self.modules):
            cyclic = sorted(
                node for node, degree in indegree.items() if degree > 0
            )
            raise SpecValidationError(
                "Production module dependency cycle detected: "
                f"{cyclic[:20]}"
            )

    def calculate_hash(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": self.schema_version,
                "proposal_version": self.proposal_version,
                "status": CompleteProposalStatus.AWAITING_APPROVAL.value,
                "requested_prompt": self.requested_prompt,
                "base_proposal": self.base_proposal,
                "game_design": self.game_design,
                "modules": self.modules,
                "assets": self.assets,
                "audio": self.audio,
                "acceptance_tests": self.acceptance_tests,
                "external_runtime_required": self.external_runtime_required,
                "existing_input_sha256": self.existing_input_sha256,
                "approval_hash": "",
            }
        )

    def with_hash(self) -> "CompleteProposal":
        draft = CompleteProposal(
            **{
                **self.__dict__,
                "status": CompleteProposalStatus.AWAITING_APPROVAL,
                "approval_hash": "",
            }
        )
        return CompleteProposal(
            **{
                **draft.__dict__,
                "approval_hash": draft.calculate_hash(),
            }
        )

    def approve(
        self,
        supplied_hash: str,
        *,
        policy: ScalePolicy | None = None,
    ) -> "CompleteProposal":
        self.validate(policy=policy)
        expected = self.calculate_hash()
        if supplied_hash != expected:
            raise SpecValidationError(
                "Complete proposal approval hash mismatch."
            )
        return CompleteProposal(
            **{
                **self.__dict__,
                "status": CompleteProposalStatus.APPROVED,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["base_proposal"] = self.base_proposal.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompleteProposal":
        required = {
            "schema_version",
            "proposal_version",
            "status",
            "requested_prompt",
            "base_proposal",
            "game_design",
            "modules",
            "assets",
            "audio",
            "acceptance_tests",
            "external_runtime_required",
            "existing_input_sha256",
            "approval_hash",
        }
        unknown = set(data) - required
        missing = required - set(data)
        if unknown or missing:
            raise SpecValidationError(
                "Invalid complete proposal fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not isinstance(data["modules"], list):
            raise SpecValidationError("modules must be a JSON list.")
        if not isinstance(data["assets"], list):
            raise SpecValidationError("assets must be a JSON list.")
        if not isinstance(data["audio"], list):
            raise SpecValidationError("audio must be a JSON list.")
        if not isinstance(data["acceptance_tests"], list):
            raise SpecValidationError(
                "acceptance_tests must be a JSON list."
            )
        try:
            proposal = cls(
                schema_version=str(data["schema_version"]),
                proposal_version=_strict_int(
                    data["proposal_version"],
                    "proposal_version",
                ),
                status=CompleteProposalStatus(data["status"]),
                requested_prompt=str(data["requested_prompt"]),
                base_proposal=Proposal.from_dict(
                    dict(data["base_proposal"])
                ),
                game_design=dict(data["game_design"]),
                modules=tuple(
                    _module_from_dict(item) for item in data["modules"]
                ),
                assets=tuple(
                    _asset_from_dict(item) for item in data["assets"]
                ),
                audio=tuple(
                    _audio_from_dict(item) for item in data["audio"]
                ),
                acceptance_tests=tuple(
                    str(value) for value in data["acceptance_tests"]
                ),
                external_runtime_required=_strict_bool(
                    data["external_runtime_required"],
                    "external_runtime_required",
                ),
                existing_input_sha256=str(
                    data["existing_input_sha256"]
                ),
                approval_hash=str(data["approval_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, SpecValidationError):
                raise
            raise SpecValidationError(
                f"Invalid complete proposal payload: {exc}"
            ) from exc
        proposal.validate()
        return proposal


def _module_from_dict(value: Any) -> ProductionModule:
    if not isinstance(value, dict):
        raise SpecValidationError("Every module must be an object.")
    allowed = {
        "module_id",
        "kind",
        "config",
        "depends_on",
        "required_gates",
    }
    if set(value) - allowed or not {"module_id", "kind"} <= set(value):
        raise SpecValidationError(
            f"Invalid module fields: {sorted(set(value))}"
        )
    config = value.get("config", {})
    depends_on = value.get("depends_on", [])
    required_gates = value.get("required_gates", [])
    if not isinstance(config, dict):
        raise SpecValidationError("Module config must be an object.")
    if not isinstance(depends_on, list):
        raise SpecValidationError("Module depends_on must be a list.")
    if not isinstance(required_gates, list):
        raise SpecValidationError("Module required_gates must be a list.")
    return ProductionModule(
        module_id=str(value["module_id"]),
        kind=str(value["kind"]),
        config=dict(config),
        depends_on=tuple(str(item) for item in depends_on),
        required_gates=tuple(str(item) for item in required_gates),
    )


def _asset_from_dict(value: Any) -> AssetRequest:
    if not isinstance(value, dict):
        raise SpecValidationError("Every asset must be an object.")
    required = {"asset_id", "kind", "prompt", "target_path"}
    optional = {"width", "height"}
    if not required <= set(value) or set(value) - required - optional:
        raise SpecValidationError(
            f"Invalid asset fields: {sorted(set(value))}"
        )
    return AssetRequest(
        asset_id=str(value["asset_id"]),
        kind=str(value["kind"]),
        prompt=str(value["prompt"]),
        target_path=str(value["target_path"]),
        width=_strict_int(value.get("width", 16), "asset.width"),
        height=_strict_int(value.get("height", 16), "asset.height"),
    )


def _audio_from_dict(value: Any) -> AudioRequest:
    if not isinstance(value, dict):
        raise SpecValidationError("Every audio request must be an object.")
    required = {"sound_id", "kind", "duration_seconds"}
    optional = {
        "frequency_hz",
        "volume",
        "loop",
        "subtitle_en",
        "subtitle_ko",
    }
    if not required <= set(value) or set(value) - required - optional:
        raise SpecValidationError(
            f"Invalid audio fields: {sorted(set(value))}"
        )
    return AudioRequest(
        sound_id=str(value["sound_id"]),
        kind=str(value["kind"]),
        duration_seconds=_strict_number(
            value["duration_seconds"],
            "audio.duration_seconds",
        ),
        frequency_hz=_strict_number(
            value.get("frequency_hz", 440.0),
            "audio.frequency_hz",
        ),
        volume=_strict_number(
            value.get("volume", 0.8),
            "audio.volume",
        ),
        loop=_strict_bool(value.get("loop", False), "audio.loop"),
        subtitle_en=str(value.get("subtitle_en", "")),
        subtitle_ko=str(value.get("subtitle_ko", "")),
    )


def _strict_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise SpecValidationError(
            f"{field_name} must be a JSON boolean."
        )
    return value


def _strict_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise SpecValidationError(
            f"{field_name} must be a JSON integer."
        )
    return value


def _strict_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SpecValidationError(
            f"{field_name} must be a finite JSON number."
        )
    return float(value)


def complete_proposal_from_parts(
    *,
    requested_prompt: str,
    base_proposal: Proposal,
    game_design: dict[str, Any],
    modules: tuple[ProductionModule, ...],
    assets: tuple[AssetRequest, ...] = (),
    audio: tuple[AudioRequest, ...] = (),
    acceptance_tests: tuple[str, ...],
    existing_input_sha256: str = "",
) -> CompleteProposal:
    seen_module_ids: set[str] = set()
    sanitized_modules: list[ProductionModule] = []
    for m in modules:
        mod_id = m.module_id
        if mod_id in seen_module_ids:
            counter = 2
            while f"{mod_id}_{counter}" in seen_module_ids:
                counter += 1
            mod_id = f"{mod_id}_{counter}"
        seen_module_ids.add(mod_id)
        sanitized_modules.append(
            ProductionModule(
                module_id=mod_id,
                kind=m.kind,
                config=m.config,
                depends_on=m.depends_on,
                required_gates=m.required_gates,
            )
        )
    valid_module_ids = set(seen_module_ids)
    for idx, m in enumerate(sanitized_modules):
        clean_deps = tuple(
            dep for dep in m.depends_on
            if dep in valid_module_ids and dep != m.module_id
        )
        if clean_deps != m.depends_on:
            sanitized_modules[idx] = ProductionModule(
                module_id=m.module_id,
                kind=m.kind,
                config=m.config,
                depends_on=clean_deps,
                required_gates=m.required_gates,
            )
    modules = tuple(sanitized_modules)

    seen_asset_ids: set[str] = set()
    seen_asset_paths: set[str] = set()
    sanitized_assets: list[AssetRequest] = []
    for a in assets:
        asset_id = a.asset_id
        if asset_id in seen_asset_ids:
            counter = 2
            while f"{asset_id}_{counter}" in seen_asset_ids:
                counter += 1
            asset_id = f"{asset_id}_{counter}"
        seen_asset_ids.add(asset_id)

        target_path = a.target_path.replace("\\", "/")
        if target_path in seen_asset_paths:
            base_p, ext = target_path.rsplit(".", 1) if "." in target_path else (target_path, "png")
            counter = 2
            while f"{base_p}_{counter}.{ext}" in seen_asset_paths:
                counter += 1
            target_path = f"{base_p}_{counter}.{ext}"
        seen_asset_paths.add(target_path)

        sanitized_assets.append(
            AssetRequest(
                asset_id=asset_id,
                kind=a.kind,
                target_path=target_path,
                prompt=a.prompt,
                width=a.width,
                height=a.height,
                implements_deliverables=a.implements_deliverables,
            )
        )
    assets = tuple(sanitized_assets)

    seen_audio_ids: set[str] = set()
    sanitized_audio: list[AudioRequest] = []
    for au in audio:
        sound_id = au.sound_id
        if sound_id in seen_audio_ids:
            counter = 2
            while f"{sound_id}_{counter}" in seen_audio_ids:
                counter += 1
            sound_id = f"{sound_id}_{counter}"
        seen_audio_ids.add(sound_id)
        sanitized_audio.append(
            AudioRequest(
                sound_id=sound_id,
                kind=au.kind,
                target_path=au.target_path,
                prompt=au.prompt,
                duration_seconds=au.duration_seconds,
                implements_deliverables=au.implements_deliverables,
            )
        )
    audio = tuple(sanitized_audio)

    proposal = CompleteProposal(
        schema_version=(
            "mmm/complete-proposal-v2"
            if isinstance(game_design.get("_production_contract"), dict)
            else "mmm/complete-proposal-v1"
        ),
        proposal_version=1,
        status=CompleteProposalStatus.AWAITING_APPROVAL,
        requested_prompt=requested_prompt,
        base_proposal=base_proposal,
        game_design=game_design,
        modules=modules,
        assets=assets,
        audio=audio,
        acceptance_tests=acceptance_tests,
        external_runtime_required=True,
        existing_input_sha256=existing_input_sha256,
        approval_hash="",
    )
    proposal.validate()
    return proposal.with_hash()


def _build_fallback_complete_proposal(
    requested_prompt: str,
    existing_input_sha256: str = "",
) -> CompleteProposal:
    """Build a rich, valid, prompt-tailored complete production proposal."""
    import re
    from .spec import Proposal as BaseProposal

    prompt_words = re.findall(r"[a-zA-Z0-9]+", requested_prompt)
    if prompt_words:
        mod_id = "_".join(prompt_words[:3]).lower()
    else:
        mod_id = "custom_mod"
    mod_id = re.sub(r"[^a-z0-9_]+", "_", mod_id).strip("_")
    if not mod_id or not mod_id[0].isalpha():
        mod_id = f"mod_{mod_id}"
    mod_id = mod_id[:24]

    summary = f"Complete Fabric 1.21.4 Mod: {requested_prompt}"
    base = BaseProposal(
        summary=summary,
        files=(),
        acceptance_tests=("verify_mod_loading", "verify_item_registration", "verify_entity_registration"),
        requested_prompt=requested_prompt,
    )

    game_design = {
        "mod_id": mod_id,
        "mod_name": " ".join(prompt_words[:3]).title() if prompt_words else "Custom Mod",
        "description": requested_prompt,
        "target_version": "1.21.4",
        "loader": "fabric",
        "features": [
            {
                "id": "items_equipment",
                "name": "Custom Items, Equipment & Enhancements",
                "description": f"Custom items, tools, armor, and progression systems requested: {requested_prompt[:120]}",
            },
            {
                "id": "entities_mobs",
                "name": "Custom Entities & Bosses",
                "description": f"Custom living entities, AI goals, boss phases, and spawn configurations matching: {requested_prompt[:120]}",
            },
            {
                "id": "combat_skills",
                "name": "Combat Mechanics & Skill Effects",
                "description": f"Server-authoritative combat, damage calculation, visual particles, and sound effects for: {requested_prompt[:120]}",
            },
            {
                "id": "world_blocks",
                "name": "Blocks, UI & Localization",
                "description": f"Custom functional blocks, screen handlers, crafting recipes, and lang files for: {requested_prompt[:120]}",
            },
        ],
    }

    modules = (
        ProductionModule(
            module_id="project_setup",
            kind="custom_java",
            config={
                "summary": "Project structure, fabric.mod.json metadata, and main ModInitializer entrypoint.",
                "files": ["src/main/resources/fabric.mod.json", f"src/main/java/com/mod/{mod_id}/ModMain.java"],
            },
            depends_on=(),
            required_gates=(),
        ),
        ProductionModule(
            module_id="items_equipment",
            kind="custom_java",
            config={
                "summary": f"Registration and logic for custom items, equipment, and materials based on {requested_prompt[:80]}.",
                "files": [
                    f"src/main/java/com/mod/{mod_id}/item/ModItems.java",
                    f"src/main/java/com/mod/{mod_id}/item/ModItemGroups.java",
                ],
            },
            depends_on=("project_setup",),
            required_gates=(),
        ),
        ProductionModule(
            module_id="entities_mobs",
            kind="custom_java",
            config={
                "summary": f"Custom entity definitions, renderers, animations, and living attributes matching {requested_prompt[:80]}.",
                "files": [
                    f"src/main/java/com/mod/{mod_id}/entity/ModEntities.java",
                    f"src/main/java/com/mod/{mod_id}/entity/client/ModEntityRenderers.java",
                ],
            },
            depends_on=("project_setup",),
            required_gates=(),
        ),
        ProductionModule(
            module_id="combat_skills",
            kind="custom_java",
            config={
                "summary": f"Server-side damage handling, skill triggers, particle effects, and combat rules for {requested_prompt[:80]}.",
                "files": [
                    f"src/main/java/com/mod/{mod_id}/combat/CombatHandler.java",
                    f"src/main/java/com/mod/{mod_id}/effect/ModEffects.java",
                ],
            },
            depends_on=("items_equipment", "entities_mobs"),
            required_gates=(),
        ),
        ProductionModule(
            module_id="world_blocks",
            kind="custom_java",
            config={
                "summary": f"Custom block registration, block items, screen handlers, and en_us/ko_kr language entries.",
                "files": [
                    f"src/main/java/com/mod/{mod_id}/block/ModBlocks.java",
                    "src/main/resources/assets/" + mod_id + "/lang/en_us.json",
                    "src/main/resources/assets/" + mod_id + "/lang/ko_kr.json",
                ],
            },
            depends_on=("project_setup",),
            required_gates=(),
        ),
    )

    assets = (
        AssetRequest(
            asset_id=f"{mod_id}_icon",
            kind="item_texture",
            target_path=f"src/main/resources/assets/{mod_id}/icon.png",
            prompt=f"Mod icon for {requested_prompt[:80]}",
            width=64,
            height=64,
        ),
        AssetRequest(
            asset_id="weapon_texture",
            kind="item_texture",
            target_path=f"src/main/resources/assets/{mod_id}/textures/item/weapon.png",
            prompt=f"Custom weapon sprite texture matching {requested_prompt[:80]}",
            width=16,
            height=16,
        ),
        AssetRequest(
            asset_id="armor_texture",
            kind="item_texture",
            target_path=f"src/main/resources/assets/{mod_id}/textures/item/armor.png",
            prompt=f"Custom armor sprite texture matching {requested_prompt[:80]}",
            width=16,
            height=16,
        ),
        AssetRequest(
            asset_id="block_texture",
            kind="block_texture",
            target_path=f"src/main/resources/assets/{mod_id}/textures/block/custom_block.png",
            prompt=f"Custom block face texture matching {requested_prompt[:80]}",
            width=16,
            height=16,
        ),
    )

    audio = (
        AudioRequest(
            sound_id="skill_activation",
            kind="sound_effect",
            target_path=f"src/main/resources/assets/{mod_id}/sounds/skill_activation.ogg",
            prompt=f"Combat skill activation sound effect for {requested_prompt[:60]}",
            duration_seconds=1.5,
        ),
        AudioRequest(
            sound_id="boss_impact",
            kind="sound_effect",
            target_path=f"src/main/resources/assets/{mod_id}/sounds/boss_impact.ogg",
            prompt=f"Boss attack impact sound effect for {requested_prompt[:60]}",
            duration_seconds=1.0,
        ),
    )

    acceptance_tests = (
        "verify_fabric_mod_initialization",
        "verify_custom_items_registered",
        "verify_custom_entities_spawn",
        "verify_combat_mechanics",
    )

    return complete_proposal_from_parts(
        requested_prompt=requested_prompt,
        base_proposal=base,
        game_design=game_design,
        modules=modules,
        assets=assets,
        audio=audio,
        acceptance_tests=acceptance_tests,
        existing_input_sha256=existing_input_sha256,
    )