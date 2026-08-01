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
_RESOURCE_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")

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
            raise SpecValidationError(
                f"Invalid production module id: {self.module_id!r}"
            )
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

        ids: set[str] = set()
        for module in self.modules:
            module.validate(policy=policy)
            if module.module_id in ids:
                raise SpecValidationError(
                    f"Duplicate production module: {module.module_id}"
                )
            ids.add(module.module_id)
        for module in self.modules:
            missing = set(module.depends_on) - ids
            if missing:
                raise SpecValidationError(
                    f"Module {module.module_id} references missing dependencies: "
                    f"{sorted(missing)}"
                )
        self._validate_acyclic()

        for asset in self.assets:
            asset.validate(policy=policy)
        if len({asset.asset_id for asset in self.assets}) != len(self.assets):
            raise SpecValidationError("Asset IDs must be unique.")
        if (
            len(
                {
                    asset.target_path.replace("\\", "/")
                    for asset in self.assets
                }
            )
            != len(self.assets)
        ):
            raise SpecValidationError("Asset target paths must be unique.")

        for audio in self.audio:
            audio.validate(policy=policy)
        if len({audio.sound_id for audio in self.audio}) != len(self.audio):
            raise SpecValidationError("Audio IDs must be unique.")

        world_structure_ids: set[str] = set()
        if self.world_ir is not None:
            world_structure_ids = _validate_world_ir(
                self.world_ir,
                policy=policy,
            )
        custom_structure_ids = {
            module.module_id
            for module in self.modules
            if (
                module.kind == "custom_java"
                and module.config.get("requested_kind") == "structure"
            )
            or (
                module.kind == "structure"
                and module.config.get("implementation") == "custom"
            )
        }
        overlap = custom_structure_ids & world_structure_ids
        if overlap:
            raise SpecValidationError(
                "Custom structures may not also appear in built-in world_ir: "
                f"{sorted(overlap)}"
            )

        if (
            not self.acceptance_tests
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.acceptance_tests
            )
        ):
            raise SpecValidationError(
                "At least one non-empty complete-production acceptance test is required."
            )
        if len(set(self.acceptance_tests)) != len(self.acceptance_tests):
            raise SpecValidationError("Acceptance tests must be unique.")
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
            module.module_id: len(module.depends_on)
            for module in self.modules
        }
        for module in self.modules:
            for dependency in module.depends_on:
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
                "world_ir": self.world_ir,
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
                world_ir=(
                    dict(data["world_ir"])
                    if data["world_ir"] is not None
                    else None
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


def _validate_world_ir(
    ir: dict[str, Any],
    *,
    policy: ScalePolicy,
) -> set[str]:
    required = {
        "schema_version",
        "regions",
        "routes",
        "structures",
        "quests",
        "constraints",
    }
    if set(ir) != required or ir.get("schema_version") != "mmm/world-ir-v1":
        raise SpecValidationError("world_ir schema is invalid.")
    try:
        validate_canonical_json(ir)
    except (CanonicalJsonError, RecursionError) as exc:
        raise SpecValidationError(
            "world_ir must contain finite JSON values."
        ) from exc
    for key in ("regions", "routes", "structures", "quests", "constraints"):
        if not isinstance(ir[key], list):
            raise SpecValidationError(f"world_ir.{key} must be a list.")

    region_ids: set[str] = set()
    for region in ir["regions"]:
        if not isinstance(region, dict):
            raise SpecValidationError("Every world region must be an object.")
        region_id = str(region.get("id", ""))
        if not _ID.fullmatch(region_id) or region_id in region_ids:
            raise SpecValidationError(
                f"Invalid or duplicate world region: {region_id!r}"
            )
        region_ids.add(region_id)

    graph = {region_id: set() for region_id in region_ids}
    route_pairs: set[tuple[str, str]] = set()
    for route in ir["routes"]:
        if not isinstance(route, dict):
            raise SpecValidationError("Every world route must be an object.")
        left = str(route.get("from", ""))
        right = str(route.get("to", ""))
        if left not in region_ids or right not in region_ids or left == right:
            raise SpecValidationError(
                f"World route references invalid regions: {left!r}, {right!r}"
            )
        pair = tuple(sorted((left, right)))
        if pair in route_pairs:
            raise SpecValidationError(
                f"Duplicate world route: {pair[0]} <-> {pair[1]}"
            )
        route_pairs.add(pair)
        graph[left].add(right)
        graph[right].add(left)
    if graph and not _connected(graph):
        raise SpecValidationError("World region graph is disconnected.")

    structure_ids: set[str] = set()
    for structure in ir["structures"]:
        if not isinstance(structure, dict):
            raise SpecValidationError(
                "Every world structure must be an object."
            )
        structure_id = str(structure.get("id", ""))
        if not _ID.fullmatch(structure_id) or structure_id in structure_ids:
            raise SpecValidationError(
                f"Invalid or duplicate structure ID: {structure_id!r}"
            )
        structure_ids.add(structure_id)
        if structure.get("region_id") not in region_ids:
            raise SpecValidationError(
                f"Structure {structure_id} references an unknown region."
            )
        size = structure.get("size", [9, 6, 9])
        if (
            not isinstance(size, list)
            or len(size) != 3
            or any(type(value) is not int or value < 1 for value in size)
        ):
            raise SpecValidationError(
                f"Structure {structure_id} has an invalid size."
            )
        palette = structure.get(
            "palette",
            ["minecraft:stone_bricks", "minecraft:air"],
        )
        if not isinstance(palette, list) or not palette:
            raise SpecValidationError(
                f"Structure {structure_id} requires a palette."
            )
        for block in palette:
            if not isinstance(block, str) or not _RESOURCE_ID.fullmatch(block):
                raise SpecValidationError(
                    f"Structure {structure_id} has an invalid palette block: {block!r}"
                )
        biomes = structure.get("biomes", ["minecraft:plains"])
        if not isinstance(biomes, list) or not biomes:
            raise SpecValidationError(
                f"Structure {structure_id} requires at least one biome."
            )
        for biome in biomes:
            if not isinstance(biome, str) or not _RESOURCE_ID.fullmatch(biome):
                raise SpecValidationError(
                    f"Structure {structure_id} has an invalid biome: {biome!r}"
                )

    quest_ids: set[str] = set()
    for quest in ir["quests"]:
        if not isinstance(quest, dict):
            raise SpecValidationError(
                "Every world quest must be an object."
            )
        quest_id = str(quest.get("id", ""))
        if not _ID.fullmatch(quest_id) or quest_id in quest_ids:
            raise SpecValidationError(
                f"Invalid or duplicate world quest: {quest_id!r}"
            )
        quest_ids.add(quest_id)
        for field_name in ("start_region", "end_region"):
            region = quest.get(field_name)
            if region not in region_ids:
                raise SpecValidationError(
                    f"World quest {quest_id} references unknown {field_name}."
                )
        if not str(quest.get("objective", "")).strip():
            raise SpecValidationError(
                f"World quest {quest_id} objective is empty."
            )

    for constraint in ir["constraints"]:
        if not isinstance(constraint, (str, dict)):
            raise SpecValidationError(
                "World constraints must be strings or objects."
            )
    return structure_ids


def _connected(graph: dict[str, set[str]]) -> bool:
    if not graph:
        return True
    start = next(iter(graph))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(graph)


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
