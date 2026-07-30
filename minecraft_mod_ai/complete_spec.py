from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .scale_policy import ScalePolicy
from .spec import Proposal, SpecValidationError, canonical_json


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


@dataclass(frozen=True)
class ProductionModule:
    module_id: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        if not _ID.fullmatch(self.module_id):
            raise SpecValidationError(f"Invalid production module id: {self.module_id!r}")
        if self.kind not in MODULE_KINDS:
            raise SpecValidationError(f"Unsupported production module kind: {self.kind!r}")
        if not isinstance(self.config, dict):
            raise SpecValidationError(f"Module config must be an object: {self.module_id}")
        try:
            encoded = json.dumps(self.config, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(f"Module config is not finite JSON: {self.module_id}") from exc
        if len(encoded) > policy.max_single_file_bytes:
            raise SpecValidationError(
                f"Module config exceeds the configured per-file resource policy: {self.module_id}"
            )
        for dependency in self.depends_on:
            if not _ID.fullmatch(dependency):
                raise SpecValidationError(
                    f"Invalid dependency {dependency!r} in module {self.module_id}"
                )
        if len(set(self.depends_on)) != len(self.depends_on):
            raise SpecValidationError(f"Duplicate dependency in module {self.module_id}")
        for gate in self.required_gates:
            if not isinstance(gate, str) or not gate.strip():
                raise SpecValidationError(f"Invalid gate in module {self.module_id}")


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
        if self.kind not in {"item", "block", "entity", "gui", "environment", "icon"}:
            raise SpecValidationError(f"Unsupported asset kind: {self.kind!r}")
        if not self.prompt.strip():
            raise SpecValidationError(f"Asset prompt is empty: {self.asset_id}")
        normalized = self.target_path.replace("\\", "/")
        if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise SpecValidationError(f"Unsafe asset target path: {self.target_path!r}")
        if type(self.width) is not int or type(self.height) is not int:
            raise SpecValidationError(f"Asset dimensions must be integers: {self.asset_id}")
        if not (1 <= self.width <= policy.max_texture_dimension):
            raise SpecValidationError(f"Asset width exceeds configured resource policy: {self.asset_id}")
        if not (1 <= self.height <= policy.max_texture_dimension):
            raise SpecValidationError(f"Asset height exceeds configured resource policy: {self.asset_id}")


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
            raise SpecValidationError(f"Unsupported audio kind: {self.kind!r}")
        values = (self.duration_seconds, self.frequency_hz, self.volume)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise SpecValidationError(f"Audio numeric fields are invalid: {self.sound_id}")
        if any(not math.isfinite(float(value)) for value in values):
            raise SpecValidationError(f"Audio numeric fields must be finite: {self.sound_id}")
        if not 0.001 <= float(self.duration_seconds) <= policy.max_audio_seconds:
            raise SpecValidationError(f"Audio duration exceeds configured resource policy: {self.sound_id}")
        if not 1.0 <= float(self.frequency_hz) <= 96_000.0:
            raise SpecValidationError(f"Invalid audio frequency: {self.sound_id}")
        if not 0.0 < float(self.volume) <= 4.0:
            raise SpecValidationError(f"Invalid audio volume: {self.sound_id}")
        if type(self.loop) is not bool:
            raise SpecValidationError(f"Audio loop must be boolean: {self.sound_id}")


@dataclass(frozen=True)
class CompleteProposal:
    schema_version: str
    proposal_version: int
    status: CompleteProposalStatus
    requested_prompt: str
    base_proposal: Proposal
    game_design: dict[str, Any]
    modules: tuple[ProductionModule, ...]
    world_ir: dict[str, Any] | None = None
    assets: tuple[AssetRequest, ...] = ()
    audio: tuple[AudioRequest, ...] = ()
    acceptance_tests: tuple[str, ...] = ()
    external_runtime_required: bool = True
    existing_input_sha256: str = ""
    approval_hash: str = ""

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        policy.validate()
        if self.schema_version != "mmm/complete-proposal-v1":
            raise SpecValidationError(f"Unsupported complete proposal schema: {self.schema_version}")
        if type(self.proposal_version) is not int or self.proposal_version < 1:
            raise SpecValidationError("proposal_version must be a positive integer.")
        if not self.requested_prompt.strip():
            raise SpecValidationError("requested_prompt must not be empty.")
        self.base_proposal.validate()
        if not isinstance(self.game_design, dict) or not self.game_design:
            raise SpecValidationError("game_design must be a non-empty object.")
        if not self.modules:
            raise SpecValidationError("A complete proposal must contain at least one production module.")

        ids: set[str] = set()
        for module in self.modules:
            module.validate(policy=policy)
            if module.module_id in ids:
                raise SpecValidationError(f"Duplicate production module: {module.module_id}")
            ids.add(module.module_id)
        for module in self.modules:
            missing = set(module.depends_on) - ids
            if missing:
                raise SpecValidationError(
                    f"Module {module.module_id} references missing dependencies: {sorted(missing)}"
                )
        self._validate_acyclic()

        for asset in self.assets:
            asset.validate(policy=policy)
        if len({asset.asset_id for asset in self.assets}) != len(self.assets):
            raise SpecValidationError("Asset IDs must be unique.")
        if len({asset.target_path.replace('\\', '/') for asset in self.assets}) != len(self.assets):
            raise SpecValidationError("Asset target paths must be unique.")
        for audio in self.audio:
            audio.validate(policy=policy)
        if len({audio.sound_id for audio in self.audio}) != len(self.audio):
            raise SpecValidationError("Audio IDs must be unique.")
        if self.world_ir is not None:
            if not isinstance(self.world_ir, dict):
                raise SpecValidationError("world_ir must be an object or null.")
            if self.world_ir.get("schema_version") != "mmm/world-ir-v1":
                raise SpecValidationError("world_ir must use mmm/world-ir-v1.")
        if not self.acceptance_tests or any(not value.strip() for value in self.acceptance_tests):
            raise SpecValidationError("At least one non-empty complete-production acceptance test is required.")
        if type(self.external_runtime_required) is not bool:
            raise SpecValidationError("external_runtime_required must be boolean.")
        if self.existing_input_sha256 and not _SHA.fullmatch(self.existing_input_sha256):
            raise SpecValidationError("existing_input_sha256 must be empty or a lowercase SHA-256 digest.")
        if self.approval_hash:
            if not _SHA.fullmatch(self.approval_hash):
                raise SpecValidationError("approval_hash must be a lowercase SHA-256 digest.")
            if self.approval_hash != self.calculate_hash():
                raise SpecValidationError("Complete proposal approval_hash does not match its payload.")

    def _validate_acyclic(self) -> None:
        outgoing: dict[str, list[str]] = {module.module_id: [] for module in self.modules}
        indegree = {module.module_id: len(module.depends_on) for module in self.modules}
        for module in self.modules:
            for dependency in module.depends_on:
                outgoing[dependency].append(module.module_id)
        ready = [node for node, degree in indegree.items() if degree == 0]
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
            cyclic = sorted(node for node, degree in indegree.items() if degree > 0)
            raise SpecValidationError(f"Production module dependency cycle detected: {cyclic[:20]}")

    def calculate_hash(self) -> str:
        payload = self.to_dict()
        payload["status"] = CompleteProposalStatus.AWAITING_APPROVAL.value
        payload["approval_hash"] = ""
        return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def with_hash(self) -> "CompleteProposal":
        draft = CompleteProposal(
            **{
                **self.__dict__,
                "status": CompleteProposalStatus.AWAITING_APPROVAL,
                "approval_hash": "",
            }
        )
        return CompleteProposal(**{**draft.__dict__, "approval_hash": draft.calculate_hash()})

    def approve(
        self,
        supplied_hash: str,
        *,
        policy: ScalePolicy | None = None,
    ) -> "CompleteProposal":
        self.validate(policy=policy)
        expected = self.calculate_hash()
        if supplied_hash != expected:
            raise SpecValidationError("Complete proposal approval hash mismatch.")
        return CompleteProposal(**{**self.__dict__, "status": CompleteProposalStatus.APPROVED})

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
            "world_ir",
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
                f"Invalid complete proposal fields; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        proposal = cls(
            schema_version=str(data["schema_version"]),
            proposal_version=int(data["proposal_version"]),
            status=CompleteProposalStatus(data["status"]),
            requested_prompt=str(data["requested_prompt"]),
            base_proposal=Proposal.from_dict(dict(data["base_proposal"])),
            game_design=dict(data["game_design"]),
            modules=tuple(
                ProductionModule(
                    module_id=str(item["module_id"]),
                    kind=str(item["kind"]),
                    config=dict(item.get("config", {})),
                    depends_on=tuple(str(value) for value in item.get("depends_on", ())),
                    required_gates=tuple(str(value) for value in item.get("required_gates", ())),
                )
                for item in data["modules"]
            ),
            world_ir=dict(data["world_ir"]) if data["world_ir"] is not None else None,
            assets=tuple(AssetRequest(**item) for item in data["assets"]),
            audio=tuple(AudioRequest(**item) for item in data["audio"]),
            acceptance_tests=tuple(str(value) for value in data["acceptance_tests"]),
            external_runtime_required=_strict_bool(
                data["external_runtime_required"],
                "external_runtime_required",
            ),
            existing_input_sha256=str(data["existing_input_sha256"]),
            approval_hash=str(data["approval_hash"]),
        )
        proposal.validate()
        return proposal


def _strict_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise SpecValidationError(f"{field_name} must be a JSON boolean.")
    return value


def complete_proposal_from_parts(
    *,
    requested_prompt: str,
    base_proposal: Proposal,
    game_design: dict[str, Any],
    modules: tuple[ProductionModule, ...],
    world_ir: dict[str, Any] | None = None,
    assets: tuple[AssetRequest, ...] = (),
    audio: tuple[AudioRequest, ...] = (),
    acceptance_tests: tuple[str, ...],
    existing_input_sha256: str = "",
) -> CompleteProposal:
    proposal = CompleteProposal(
        schema_version="mmm/complete-proposal-v1",
        proposal_version=1,
        status=CompleteProposalStatus.AWAITING_APPROVAL,
        requested_prompt=requested_prompt,
        base_proposal=base_proposal,
        game_design=game_design,
        modules=modules,
        world_ir=world_ir,
        assets=assets,
        audio=audio,
        acceptance_tests=acceptance_tests,
        external_runtime_required=True,
        existing_input_sha256=existing_input_sha256,
        approval_hash="",
    ).with_hash()
    proposal.validate()
    return proposal
