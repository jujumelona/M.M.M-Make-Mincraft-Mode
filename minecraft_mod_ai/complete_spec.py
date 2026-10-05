from __future__ import annotations

import heapq
import json
import re
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import PurePosixPath
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
        "tag",
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
        "integration",
        "typed_host",
    }
)
ASSET_KINDS = frozenset({"item", "block", "entity", "gui", "environment", "icon"})
IMPLEMENTATION_KINDS = MODULE_KINDS


class CompleteProposalStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_user_approval"
    APPROVED = "approved"


@dataclass(frozen=True)
class ProductionModule(Mapping[str, Any]):
    module_id: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()

    def __getitem__(self, key: str) -> Any:
        if key == "plugin_id" or key == "module_id":
            return self.module_id
        if key == "kind":
            return self.kind
        if key == "config":
            return self.config
        if key == "depends_on":
            return self.depends_on
        if key == "required_gates":
            return self.required_gates
        if key in self.config:
            return self.config[key]
        if key == "requirement_refs":
            return [self.module_id]
        if key == "implementation_obligations":
            obligations = self.config.get("implementation_obligations")
            if obligations:
                return list(obligations)
            return [f"Implement {self.module_id}"]
        if key == "status":
            return "host_ready"
        if key == "capability":
            return self.kind
        if key == "reason":
            return self.config.get("reason", "")
        raise KeyError(key)

    def __iter__(self):
        seen = [
            "module_id",
            "plugin_id",
            "kind",
            "config",
            "depends_on",
            "required_gates",
            "requirement_refs",
            "implementation_obligations",
            "status",
            "capability",
            "reason",
        ]
        for k in self.config:
            if k not in seen:
                seen.append(k)
        return iter(seen)

    def __len__(self):
        return sum(1 for _ in self)

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        if not isinstance(self.module_id, str):
            raise SpecValidationError("Production module id must be a string.")
        if not isinstance(self.kind, str):
            raise SpecValidationError(
                f"Production module kind must be a string: {self.module_id!r}"
            )
        if not isinstance(self.depends_on, tuple):
            raise SpecValidationError(
                f"Module dependencies must use the canonical tuple representation: {self.module_id}"
            )
        if not isinstance(self.required_gates, tuple):
            raise SpecValidationError(
                f"Module required_gates must use the canonical tuple representation: {self.module_id}"
            )
        if not _ID.fullmatch(self.module_id):
            raise SpecValidationError(f"Invalid production module id: {self.module_id!r}")
        if self.kind not in MODULE_KINDS:
            raise SpecValidationError(f"Unsupported production module kind: {self.kind!r}")
        if not isinstance(self.config, dict):
            raise SpecValidationError(f"Module config must be an object: {self.module_id}")
        typed_plan_ir = self.config.get("typed_plan_ir")
        if self.kind == "typed_host":
            if not isinstance(typed_plan_ir, dict):
                raise SpecValidationError(
                    f"TYPED_HOST_PLAN_REQUIRED: {self.module_id}"
                )
        elif isinstance(typed_plan_ir, dict):
            raise SpecValidationError(
                f"TYPED_HOST_KIND_REQUIRED: {self.module_id}"
            )
        if self.kind == "integration" and self.config.get("integration_type") == "mmm_local_ai_sidecar":
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
        if self.kind == "integration" and self.config.get("integration_type") == "mmm_research_shard":
            from .research_ledger import (
                ResearchLedgerError,
                validate_research_shard_config,
            )

            try:
                validate_research_shard_config(
                    self.config,
                    module_id=self.module_id,
                )
            except ResearchLedgerError as exc:
                raise SpecValidationError(
                    f"Invalid reviewed research shard module {self.module_id}: {exc}"
                ) from exc
        implementation = self.config.get("implementation")
        if implementation is not None:
            raise SpecValidationError(
                "CUSTOM_JAVA_BACKEND_REMOVED: production modules must use typed_host, "
                "a supported deterministic native route, or the canonical artifact graph; "
                f"{self.module_id} supplied implementation={implementation!r}."
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
    """Canonical semantic resource request."""

    asset_id: str
    kind: str
    visual_description: str
    render_kind: str
    subject_id: str
    owner_module_id: str = ""
    container: str = "mod"
    requested_width: int | None = None
    requested_height: int | None = None
    variant_count: int = 1
    visual_spec: dict[str, Any] | None = None

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        from .resource_contracts import SUPPORTED_RENDER_KINDS

        policy = policy or ScalePolicy.from_environment()
        if not _ID.fullmatch(self.asset_id):
            raise SpecValidationError(f"Invalid asset id: {self.asset_id!r}")
        if self.kind not in ASSET_KINDS:
            raise SpecValidationError(f"Unsupported asset kind: {self.kind!r}")
        if not self.visual_description and self.visual_spec is None:
            raise SpecValidationError(f"Asset visual description is empty: {self.asset_id}")
        from .resource_visual_spec import resolve_visual_spec
        try:
            resolve_visual_spec(self.visual_spec, self.visual_description)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc
        if self.render_kind not in SUPPORTED_RENDER_KINDS:
            raise SpecValidationError(f"Unsupported asset render kind {self.render_kind!r}: {self.asset_id}")
        if not self.subject_id:
            raise SpecValidationError(f"Asset subject id is empty: {self.asset_id}")
        if self.container not in {"mod", "resource_pack"}:
            raise SpecValidationError(f"Unsupported asset container: {self.container!r}")
        if self.owner_module_id and not _ID.fullmatch(self.owner_module_id):
            raise SpecValidationError(f"Invalid asset owner module: {self.owner_module_id!r}")
        if type(self.variant_count) is not int or self.variant_count < 1:
            raise SpecValidationError(f"Asset variant_count must be positive: {self.asset_id}")
        if (self.requested_width is None) != (self.requested_height is None):
            raise SpecValidationError(f"Asset requested dimensions must be both supplied or both omitted: {self.asset_id}")
        for value, label in ((self.requested_width, "width"), (self.requested_height, "height")):
            if value is not None and (type(value) is not int or not 1 <= value <= policy.max_texture_dimension):
                raise SpecValidationError(f"Asset {label} exceeds configured resource policy: {self.asset_id}")


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
            raise SpecValidationError("proposal_version must be a positive integer.")
        if not isinstance(self.requested_prompt, str) or not self.requested_prompt.strip():
            raise SpecValidationError("requested_prompt must not be empty.")
        self.base_proposal.validate()
        if not isinstance(self.game_design, dict) or not self.game_design:
            raise SpecValidationError("game_design must be a non-empty object.")
        if self.base_proposal.spec.platform.host_facts_json:
            from .resolved_version_context import ResolvedVersionContext, VersionContextError

            resolved = self.base_proposal.spec.platform.version_context
            stored_payload = self.game_design.get("_resolved_version_context")
            if stored_payload is not None:
                if not isinstance(stored_payload, dict) or not stored_payload:
                    raise VersionContextError("INVALID_VERSION_CONTEXT")
                stored = ResolvedVersionContext.from_dict(stored_payload)
                resolved.assert_context(stored.context_id)

            bindings = self.game_design.get("_artifact_version_contexts")
            if bindings is not None:
                expected = {"module:" + module.module_id for module in self.modules}
                expected.update("asset:" + asset.asset_id for asset in self.assets)
                if not isinstance(bindings, dict) or set(bindings) != expected:
                    raise VersionContextError("ARTIFACT_CONTEXT_BINDING_MISSING")
                for identifier in bindings.values():
                    resolved.assert_context(identifier)
            for raw_job in self.game_design.get("_artifact_jobs", ()):
                if not isinstance(raw_job, Mapping):
                    raise VersionContextError("ARTIFACT_CONTEXT_BINDING_MISSING")
                resolved.assert_context(raw_job.get("context_id"))
        try:
            validate_canonical_json(self.game_design)
        except (CanonicalJsonError, RecursionError) as exc:
            raise SpecValidationError(
                "game_design must contain finite JSON values."
            ) from exc
        evidence_plan = self.game_design.get("_evidence_first_plan")
        retained_only = False
        if isinstance(evidence_plan, Mapping):
            retained_only = (
                not evidence_plan.get("gap_catalog")
                and not evidence_plan.get("tasks")
                and bool(evidence_plan.get("verified_provides"))
            )
        fixture = self.game_design.get("fixture")
        host_fixture_only = (
            self.game_design.get("mode") == "debug_fixture"
            and isinstance(fixture, Mapping)
            and fixture.get("deterministic") is True
            and isinstance(fixture.get("source_contract"), Mapping)
        )
        if not self.modules and not retained_only and not host_fixture_only:
            raise SpecValidationError(
                "A complete proposal must contain at least one production module."
            )

        bootstrap_module_ids = {
            content.content_id for content in self.base_proposal.spec.contents
        }
        if self.base_proposal.spec.boss is not None:
            boss_id = self.base_proposal.spec.boss.entity_id
            bootstrap_module_ids.update({boss_id, f"{boss_id}_spawn_egg"})
        module_bootstrap_collisions = sorted(
            bootstrap_module_ids
            & {module.module_id for module in self.modules}
        )
        if module_bootstrap_collisions:
            raise SpecValidationError(
                "Production modules may not duplicate bootstrap-owned content IDs: "
                + ", ".join(module_bootstrap_collisions[:20])
            )

        module_ids: set[str] = set()
        for module in self.modules:
            module.validate(policy=policy)
            if module.module_id in module_ids:
                raise SpecValidationError(
                    f"Duplicate production module id: {module.module_id}"
                )
            module_ids.add(module.module_id)
        for module in self.modules:
            missing = sorted(set(module.depends_on) - module_ids)
            if missing:
                raise SpecValidationError(
                    f"Module {module.module_id} references unknown dependencies: {missing[:20]}"
                )
            if module.module_id in module.depends_on:
                raise SpecValidationError(
                    f"Module {module.module_id} may not depend on itself."
                )
        self._validate_acyclic()

        raw_artifact_jobs = self.game_design.get("_artifact_jobs", ())
        if raw_artifact_jobs is None:
            raw_artifact_jobs = ()
        if (
            not isinstance(raw_artifact_jobs, (list, tuple))
            or isinstance(raw_artifact_jobs, (str, bytes, bytearray))
        ):
            raise SpecValidationError(
                "game_design._artifact_jobs must be an array when supplied."
            )
        from .artifact_job import (
            parse_artifact_jobs,
            validate_artifact_job_graph,
        )

        try:
            parsed_artifact_jobs = parse_artifact_jobs(raw_artifact_jobs)
            validate_artifact_job_graph(
                parsed_artifact_jobs,
                module_ids=module_ids,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecValidationError(
                f"Invalid canonical artifact job graph: {exc}"
            ) from exc

        artifact_owners = {
            job.owner_module
            for job in parsed_artifact_jobs
            if job.owner_module
        }

        from .platform_backend_contract import (
            deterministic_backend_capabilities,
            missing_production_backend_capabilities,
            native_production_route_available,
        )

        unroutable_modules = [
            f"{module.module_id}/{module.kind}"
            for module in self.modules
            if module.module_id not in artifact_owners
            and not native_production_route_available(module.kind, module.config)
        ]
        if unroutable_modules:
            raise SpecValidationError(
                "DETERMINISTIC_BACKEND_REQUIRED: production module has neither "
                "a native deterministic route nor an artifact owner: "
                + ", ".join(unroutable_modules[:20])
            )

        target_capabilities = deterministic_backend_capabilities(
            self.base_proposal.spec.platform
        )
        unsupported_native_modules: list[str] = []
        for module in self.modules:
            if module.module_id in artifact_owners:
                continue
            missing_backend = missing_production_backend_capabilities(
                target_capabilities,
                module.kind,
                module.config,
            )
            if missing_backend:
                unsupported_native_modules.append(
                    f"{module.module_id}/{module.kind}:"
                    f"{sorted(missing_backend)}"
                )
        if unsupported_native_modules:
            raise SpecValidationError(
                "DETERMINISTIC_BACKEND_REQUIRED: approved target cannot execute "
                "native production modules: "
                + ", ".join(unsupported_native_modules[:20])
            )

        asset_ids: set[str] = set()
        for asset in self.assets:
            asset.validate(policy=policy)
            if asset.asset_id in asset_ids:
                raise SpecValidationError(f"Duplicate asset id: {asset.asset_id}")
            if asset.owner_module_id and asset.owner_module_id not in module_ids:
                raise SpecValidationError(
                    f"Asset {asset.asset_id} references unknown owner module {asset.owner_module_id!r}"
                )
            asset_ids.add(asset.asset_id)

        if not self.acceptance_tests:
            raise SpecValidationError(
                "acceptance_tests must contain at least one test."
            )
        if len(self.acceptance_tests) != len(set(self.acceptance_tests)):
            raise SpecValidationError(
                "acceptance_tests must not contain duplicates."
            )
        for test in self.acceptance_tests:
            if not isinstance(test, str) or not test.strip():
                raise SpecValidationError(
                    "acceptance_tests must contain non-empty strings."
                )

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
                    self.modules,
                    self.acceptance_tests,
                    self.assets,
                    evidence_plan if isinstance(evidence_plan, Mapping) else None,
                )
            except ValueError as exc:
                raise SpecValidationError(
                    f"Invalid production contract: {exc}"
                ) from exc

        if type(self.external_runtime_required) is not bool:
            raise SpecValidationError("external_runtime_required must be boolean.")
        if self.existing_input_sha256 and not _SHA.fullmatch(self.existing_input_sha256):
            raise SpecValidationError(
                "existing_input_sha256 must be empty or a lowercase SHA-256 digest."
            )
        if (
            self.status is CompleteProposalStatus.APPROVED
            and not self.approval_hash
        ):
            raise SpecValidationError(
                "Complete proposal approved state requires its approval_hash integrity receipt."
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
            module.module_id: sum(
                1 for dependency in module.depends_on if dependency in outgoing
            )
            for module in self.modules
        }
        for module in self.modules:
            for dependency in module.depends_on:
                if dependency in outgoing:
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
            cyclic = sorted(
                node for node, degree in indegree.items() if degree > 0
            )
            raise SpecValidationError(
                f"Production module dependency cycle detected: {cyclic[:20]}"
            )

    def calculate_hash(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": self.schema_version,
                "proposal_version": self.proposal_version,
                "status": CompleteProposalStatus.AWAITING_APPROVAL.value,
                "requested_prompt": self.requested_prompt,
                "base_proposal": self.base_proposal,
                "game_design": deepcopy(self.game_design),
                "modules": self.modules,
                "assets": self.assets,
                "acceptance_tests": self.acceptance_tests,
                "external_runtime_required": self.external_runtime_required,
                "existing_input_sha256": self.existing_input_sha256,
                "approval_hash": "",
            }
        )

    def with_hash(self) -> CompleteProposal:
        draft = CompleteProposal(
            **{
                **self.__dict__,
                "status": CompleteProposalStatus.AWAITING_APPROVAL,
                "approval_hash": "",
            }
        )
        return CompleteProposal(
            **{**draft.__dict__, "approval_hash": draft.calculate_hash()}
        )

    def approve(
        self,
        supplied_hash: str,
        *,
        policy: ScalePolicy | None = None,
    ) -> CompleteProposal:
        self.validate(policy=policy)
        expected = self.calculate_hash()
        if supplied_hash != expected:
            raise SpecValidationError("Complete proposal approval hash mismatch.")
        return CompleteProposal(
            **{
                **self.__dict__,
                "status": CompleteProposalStatus.APPROVED,
                "approval_hash": expected,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_version": self.proposal_version,
            "status": self.status.value,
            "requested_prompt": self.requested_prompt,
            "base_proposal": self.base_proposal.to_dict(),
            "game_design": self.game_design,
            "modules": [
                {
                    "module_id": module.module_id,
                    "kind": module.kind,
                    "config": deepcopy(module.config),
                    "depends_on": list(module.depends_on),
                    "required_gates": list(module.required_gates),
                }
                for module in self.modules
            ],
            "assets": [asdict(asset) for asset in self.assets],
            "acceptance_tests": list(self.acceptance_tests),
            "external_runtime_required": self.external_runtime_required,
            "existing_input_sha256": self.existing_input_sha256,
            "approval_hash": self.approval_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompleteProposal:
        required = {
            "schema_version",
            "proposal_version",
            "status",
            "requested_prompt",
            "base_proposal",
            "game_design",
            "modules",
            "assets",
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
        if not isinstance(data["base_proposal"], dict):
            raise SpecValidationError("base_proposal must be a JSON object.")
        if not isinstance(data["game_design"], dict):
            raise SpecValidationError("game_design must be a JSON object.")
        if not isinstance(data["modules"], list):
            raise SpecValidationError("modules must be a JSON list.")
        if not isinstance(data["assets"], list):
            raise SpecValidationError("assets must be a JSON list.")
        if not isinstance(data["acceptance_tests"], list):
            raise SpecValidationError("acceptance_tests must be a JSON list.")
        try:
            proposal = cls(
                schema_version=_strict_string(data["schema_version"], "schema_version"),
                proposal_version=_strict_int(
                    data["proposal_version"], "proposal_version"
                ),
                status=CompleteProposalStatus(
                    _strict_string(data["status"], "status")
                ),
                requested_prompt=_strict_string(data["requested_prompt"], "requested_prompt"),
                base_proposal=Proposal.from_dict(deepcopy(data["base_proposal"])),
                game_design=deepcopy(data["game_design"]),
                modules=tuple(_module_from_dict(item) for item in data["modules"]),
                assets=tuple(_asset_from_dict(item) for item in data["assets"]),
                acceptance_tests=tuple(
                    _strict_string(value, "acceptance_tests[]")
                    for value in data["acceptance_tests"]
                ),
                external_runtime_required=_strict_bool(
                    data["external_runtime_required"],
                    "external_runtime_required",
                ),
                existing_input_sha256=_strict_string(
                    data["existing_input_sha256"], "existing_input_sha256"
                ),
                approval_hash=_strict_string(data["approval_hash"], "approval_hash"),
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
    expected = {"module_id", "kind", "config", "depends_on", "required_gates"}
    if set(value) != expected:
        raise SpecValidationError(
            "Invalid module fields; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    if not isinstance(value["config"], dict):
        raise SpecValidationError("Module config must be an object.")
    if not isinstance(value["depends_on"], list):
        raise SpecValidationError("Module depends_on must be a list.")
    if not isinstance(value["required_gates"], list):
        raise SpecValidationError("Module required_gates must be a list.")
    return ProductionModule(
        module_id=_strict_string(value["module_id"], "module.module_id"),
        kind=_strict_string(value["kind"], "module.kind"),
        config=deepcopy(value["config"]),
        depends_on=tuple(
            _strict_string(item, "module.depends_on[]")
            for item in value["depends_on"]
        ),
        required_gates=tuple(
            _strict_string(item, "module.required_gates[]")
            for item in value["required_gates"]
        ),
    )


def _asset_from_dict(value: Any) -> AssetRequest:
    if not isinstance(value, dict):
        raise SpecValidationError("Every asset must be an object.")
    expected = {
        "asset_id",
        "kind",
        "visual_description",
        "render_kind",
        "subject_id",
        "owner_module_id",
        "container",
        "requested_width",
        "requested_height",
        "variant_count",
        "visual_spec",
    }
    if set(value) != expected:
        raise SpecValidationError(
            "Invalid asset fields; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    visual_spec = value["visual_spec"]
    if visual_spec is not None and not isinstance(visual_spec, dict):
        raise SpecValidationError("asset.visual_spec must be an object or null.")
    return AssetRequest(
        asset_id=_strict_string(value["asset_id"], "asset.asset_id"),
        kind=_strict_string(value["kind"], "asset.kind"),
        visual_description=_strict_string(
            value["visual_description"], "asset.visual_description"
        ),
        render_kind=_strict_string(value["render_kind"], "asset.render_kind"),
        subject_id=_strict_string(value["subject_id"], "asset.subject_id"),
        owner_module_id=_strict_string(
            value["owner_module_id"], "asset.owner_module_id"
        ),
        container=_strict_string(value["container"], "asset.container"),
        requested_width=(
            None
            if value["requested_width"] is None
            else _strict_int(value["requested_width"], "asset.requested_width")
        ),
        requested_height=(
            None
            if value["requested_height"] is None
            else _strict_int(value["requested_height"], "asset.requested_height")
        ),
        variant_count=_strict_int(value["variant_count"], "asset.variant_count"),
        visual_spec=deepcopy(visual_spec),
    )


def _strict_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SpecValidationError(f"{field_name} must be a JSON string.")
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise SpecValidationError(f"{field_name} must be a JSON boolean.")
    return value


def _strict_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise SpecValidationError(f"{field_name} must be a JSON integer.")
    return value


def complete_proposal_from_parts(
    *,
    requested_prompt: str,
    base_proposal: Proposal,
    game_design: dict[str, Any],
    modules: tuple[ProductionModule, ...],
    assets: tuple[AssetRequest, ...] = (),
    acceptance_tests: tuple[str, ...],
    existing_input_sha256: str = "",
    external_runtime_required: bool = False,
) -> CompleteProposal:
    modules = tuple(modules)

    from .resource_contracts import derive_module_asset_specs

    supplied_assets = list(assets)
    supplied_assets.extend(
        AssetRequest(**row)
        for row in derive_module_asset_specs(
            modules,
            existing_asset_ids=[asset.asset_id for asset in supplied_assets],
        )
    )
    seen_asset_ids: set[str] = set()
    sanitized_assets: list[AssetRequest] = []
    valid_module_ids = {module.module_id for module in modules}
    for asset in supplied_assets:
        if asset.asset_id in seen_asset_ids:
            raise SpecValidationError(f"Duplicate semantic asset id: {asset.asset_id}")
        seen_asset_ids.add(asset.asset_id)
        if asset.owner_module_id and asset.owner_module_id not in valid_module_ids:
            raise SpecValidationError(
                f"Asset {asset.asset_id} references unknown owner module {asset.owner_module_id!r}"
            )
        asset.validate()
        sanitized_assets.append(asset)

    if base_proposal.spec.platform.host_facts_json:
        from .resolved_version_context import ResolvedVersionContext

        resolved = base_proposal.spec.platform.version_context
        prior = game_design.get("_resolved_version_context")
        if prior is not None:
            resolved.assert_context(ResolvedVersionContext.from_dict(prior).context_id)
        bindings = {"module:" + module.module_id: resolved.context_id for module in modules}
        bindings.update({"asset:" + asset.asset_id: resolved.context_id for asset in sanitized_assets})
        for identifier in game_design.get("_artifact_version_contexts", {}).values():
            resolved.assert_context(identifier)
        game_design = {**game_design, "_resolved_version_context": resolved.to_dict(),
                       "_artifact_version_contexts": bindings}

    schema_version = (
        "mmm/complete-proposal-v2"
        if isinstance(game_design.get("_production_contract"), dict)
        else "mmm/complete-proposal-v1"
    )
    proposal = CompleteProposal(
        schema_version=schema_version,
        proposal_version=1,
        status=CompleteProposalStatus.AWAITING_APPROVAL,
        requested_prompt=requested_prompt,
        base_proposal=base_proposal,
        game_design=game_design,
        modules=modules,
        assets=tuple(sanitized_assets),
        acceptance_tests=acceptance_tests,
        external_runtime_required=external_runtime_required,
        existing_input_sha256=existing_input_sha256,
        approval_hash="",
    )
    proposal.validate()
    return proposal.with_hash()


__all__ = [
    "ASSET_KINDS",
    "MODULE_KINDS",
    "AssetRequest",
    "CompleteProposal",
    "CompleteProposalStatus",
    "ProductionModule",
    "complete_proposal_from_parts",
]
