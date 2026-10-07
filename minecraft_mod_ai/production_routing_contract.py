from __future__ import annotations

"""Canonical production routing and ownership contract.

This module is the single authority for deciding whether an approved module is
owned by the artifact graph or by a reviewed native host backend.  Scheduling,
validation and resume fingerprints consume the same immutable snapshot instead
of re-deriving ownership from module kinds independently.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

from .artifact_job import (
    artifact_owner_module_ids,
    canonical_client_entrypoints,
    parse_artifact_jobs,
)
from .complete_spec import CompleteProposal, ProductionModule
from .platform_backend_contract import (
    deterministic_backend_capabilities,
    missing_production_backend_capabilities,
    native_production_stage,
    production_module_backend_capabilities,
)
from .spec import canonical_json


SCHEMA_VERSION = "mmm/production-routing-contract-v1"


class ProductionRoutingError(ValueError):
    """An approved proposal cannot be mapped to one deterministic producer route."""


@dataclass(frozen=True)
class ModuleProductionRoute:
    module_id: str
    kind: str
    owner: str
    stage: str
    required_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "kind": self.kind,
            "owner": self.owner,
            "stage": self.stage,
            "required_capabilities": list(self.required_capabilities),
        }


@dataclass(frozen=True)
class ProductionRoutingSnapshot:
    schema_version: str
    deterministic_module_kinds: tuple[str, ...]
    artifact_owner_module_ids: tuple[str, ...]
    artifact_client_entrypoints: tuple[str, ...]
    routes: tuple[ModuleProductionRoute, ...]
    contract_sha256: str

    @property
    def route_by_module_id(self) -> dict[str, ModuleProductionRoute]:
        return {route.module_id: route for route in self.routes}

    @property
    def native_module_ids(self) -> frozenset[str]:
        return frozenset(
            route.module_id for route in self.routes if route.owner == "native_host"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "deterministic_module_kinds": list(self.deterministic_module_kinds),
            "artifact_owner_module_ids": list(self.artifact_owner_module_ids),
            "artifact_client_entrypoints": list(self.artifact_client_entrypoints),
            "routes": [route.to_dict() for route in self.routes],
            "contract_sha256": self.contract_sha256,
        }


def _artifact_jobs(proposal: CompleteProposal) -> tuple[Any, ...]:
    game_design = proposal.game_design if isinstance(proposal.game_design, Mapping) else {}
    raw_jobs = game_design.get("_artifact_jobs", ())
    if raw_jobs is None:
        raw_jobs = ()
    if not isinstance(raw_jobs, (list, tuple)):
        raise ProductionRoutingError(
            "PRODUCTION_ROUTING_ARTIFACT_JOBS_INVALID: _artifact_jobs must be an array"
        )
    try:
        return parse_artifact_jobs(raw_jobs)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionRoutingError(
            f"PRODUCTION_ROUTING_ARTIFACT_JOBS_INVALID: {exc}"
        ) from exc


def compile_production_routing(
    proposal: CompleteProposal,
    *,
    modules: Sequence[ProductionModule] | None = None,
) -> ProductionRoutingSnapshot:
    """Compile the one approved producer/owner route for every selected module."""

    selected = tuple(proposal.modules) if modules is None else tuple(modules)
    module_ids = [module.module_id for module in selected]
    if len(set(module_ids)) != len(module_ids):
        raise ProductionRoutingError(
            "PRODUCTION_ROUTING_DUPLICATE_MODULE_ID"
        )

    jobs = _artifact_jobs(proposal)
    all_artifact_owners = artifact_owner_module_ids(jobs)
    selected_ids = frozenset(module_ids)
    artifact_owners = frozenset(all_artifact_owners & selected_ids)

    non_artifact_modules = tuple(
        module for module in selected if module.module_id not in artifact_owners
    )
    if non_artifact_modules:
        try:
            platform = proposal.base_proposal.spec.platform
        except AttributeError as exc:
            raise ProductionRoutingError(
                "PRODUCTION_ROUTING_PLATFORM_REQUIRED: native routes require "
                "the approved platform lock"
            ) from exc
        from .platform_catalog import adapter_for_lock_values

        adapter = adapter_for_lock_values(platform)
        available = deterministic_backend_capabilities(adapter)
    else:
        # Artifact-only validation does not need to resolve a native target.
        available = frozenset()

    routes: list[ModuleProductionRoute] = []
    for module in selected:
        if module.module_id in artifact_owners:
            routes.append(
                ModuleProductionRoute(
                    module_id=module.module_id,
                    kind=module.kind,
                    owner="artifact_graph",
                    stage="content",
                    required_capabilities=(),
                )
            )
            continue

        if module.kind == "typed_host":
            config = module.config if isinstance(module.config, Mapping) else {}
            if not isinstance(config.get("typed_plan_ir"), dict):
                raise ProductionRoutingError(
                    f"TYPED_HOST_PLAN_REQUIRED: {module.module_id}"
                )
        else:
            config = module.config if isinstance(module.config, Mapping) else {}
            if isinstance(config.get("typed_plan_ir"), dict):
                raise ProductionRoutingError(
                    f"TYPED_HOST_KIND_REQUIRED: {module.module_id}"
                )

        stage = native_production_stage(module.kind, config)
        if stage is None:
            raise ProductionRoutingError(
                "DETERMINISTIC_BACKEND_REQUIRED: unsupported production route "
                f"{module.module_id}/{module.kind}"
            )
        missing = missing_production_backend_capabilities(
            available,
            module.kind,
            config,
        )
        if missing:
            raise ProductionRoutingError(
                "DETERMINISTIC_BACKEND_REQUIRED: "
                f"{module.kind} requires missing backend capabilities "
                f"{sorted(missing)}"
            )
        required = tuple(
            sorted(production_module_backend_capabilities(module.kind, config))
        )
        routes.append(
            ModuleProductionRoute(
                module_id=module.module_id,
                kind=module.kind,
                owner="native_host",
                stage=stage,
                required_capabilities=required,
            )
        )

    body = {
        "schema_version": SCHEMA_VERSION,
        "deterministic_module_kinds": sorted(available),
        "artifact_owner_module_ids": sorted(artifact_owners),
        "artifact_client_entrypoints": list(canonical_client_entrypoints(jobs)),
        "routes": [route.to_dict() for route in routes],
    }
    digest = "sha256:" + hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()
    return ProductionRoutingSnapshot(
        schema_version=SCHEMA_VERSION,
        deterministic_module_kinds=tuple(sorted(available)),
        artifact_owner_module_ids=tuple(sorted(artifact_owners)),
        artifact_client_entrypoints=tuple(body["artifact_client_entrypoints"]),
        routes=tuple(routes),
        contract_sha256=digest,
    )


__all__ = [
    "ModuleProductionRoute",
    "ProductionRoutingError",
    "ProductionRoutingSnapshot",
    "SCHEMA_VERSION",
    "compile_production_routing",
]
