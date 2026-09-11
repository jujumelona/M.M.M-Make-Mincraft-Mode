from __future__ import annotations

"""Platform selection DTOs, parsing, proposal binding, and target decision lowering.

Actual optimisation lives in platform_evidence_pipeline through the canonical
platform_selection_pipeline. This module performs no search, retry, or ranking itself.
It is also the single lowering boundary from a verified PlatformSelection receipt into
the planner-facing target-decision receipt; downstream planners must not reconstruct
TargetContract coordinates themselves.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .module_identity import logical_module_id
from .platform_catalog import (
    adapter_for_target,
    adapters_for_version,
    discover_target_keys,
    executable_loaders,
    provider_for_loader,
)
from .platform_evidence_pipeline import PlatformOptimization, TargetResearchFn
from .spec import PlatformLock, Proposal, SpecValidationError, platform_receipt_sha256
from .target_contract import TargetContract, target_contract_from_mapping

_VERSION_RE = re.compile(r"(?<!\d)(1\.\d{1,2}(?:\.\d{1,2})?|\d{2,4}\.\d+(?:\.\d+)?)(?!\d)")
_ASCII_WORD = r"A-Za-z0-9_"
_FABRIC_RE = re.compile(rf"(?<![{_ASCII_WORD}])fabric(?![{_ASCII_WORD}])|패브릭", re.IGNORECASE)
_NEOFORGE_RE = re.compile(rf"(?<![{_ASCII_WORD}])neoforge(?![{_ASCII_WORD}])|네오포지", re.IGNORECASE)
_FORGE_RE = re.compile(rf"(?<![{_ASCII_WORD}])forge(?![{_ASCII_WORD}])|(?<!네오)포지", re.IGNORECASE)
_MIGRATION_RE = re.compile(
    r"마이그레이션|버전\s*(?:변경|업|올려|내려)|업데이트\s*해|포팅|이식|"
    r"migrat|port\s+(?:to|from)|upgrade\s+to|downgrade\s+to",
    re.IGNORECASE,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(dict.fromkeys(text for item in values if (text := str(item).strip())))


def _payload_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _planner_project_topology(
    module_ids: Sequence[Any],
    *,
    loaders: Sequence[Any] = (),
    source_sets: Sequence[Any] = (),
    gradle_project_paths: Sequence[Any] = (),
) -> dict[str, list[str]]:
    raw_ids = list(_strings(module_ids))
    logical_ids = [logical_module_id(item) for item in raw_ids]
    if len(logical_ids) != len(set(logical_ids)):
        raise SpecValidationError(
            "Project module topology collapses to duplicate logical module identities."
        )
    explicit_gradle_paths = list(_strings(gradle_project_paths))
    inferred_gradle_paths = [
        item for item in raw_ids if item == ":" or item.startswith(":")
    ]
    return {
        "module_ids": logical_ids,
        "gradle_project_paths": explicit_gradle_paths or inferred_gradle_paths,
        "loaders": list(_strings(loaders)),
        "source_sets": list(_strings(source_sets)),
    }


@dataclass(frozen=True)
class PlatformSelection:
    adapter: TargetContract
    source: str
    reason: str
    explicit_version: bool
    explicit_loader: bool
    preserved_existing_target: bool = False
    migration_requested: bool = False
    optimization: PlatformOptimization | None = None

    @property
    def lock(self) -> PlatformLock:
        return lock_from_adapter(self.adapter)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "mmm/platform-selection-v5",
            "adapter_id": self.adapter.adapter_id,
            "source": self.source,
            "reason": self.reason,
            "explicit_version": self.explicit_version,
            "explicit_loader": self.explicit_loader,
            "preserved_existing_target": self.preserved_existing_target,
            "migration_requested": self.migration_requested,
            "target": self.adapter.public_dict(),
        }
        if self.adapter.host_facts_json:
            payload["resolved_version_context"] = self.adapter.version_context.to_dict()
        if self.optimization is not None:
            payload["optimizer"] = self.optimization.to_dict()
        return payload


def compile_target_decision(
    selection_payload: Mapping[str, Any] | None,
    *,
    existing_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Lower one platform-selection receipt into the planner target decision.

    Target coordinates are never reconstructed field-by-field here.  A supplied target
    must deserialize through TargetContract and is then re-emitted with public_dict().
    Project topology and optimizer rejection metadata are decision-layer concerns and are
    derived here once so evidence planners only consume this receipt. Gradle project paths
    are preserved separately while planner-facing module IDs use the canonical logical
    module identity shared with task ownership.
    """

    raw = _mapping(selection_payload)
    raw_target = raw.get("target")
    if isinstance(raw_target, Mapping) and raw_target:
        adapter = target_contract_from_mapping(raw_target)
        target = adapter.public_dict()
        if adapter.host_facts_json:
            from .resolved_version_context import ResolvedVersionContext

            supplied_payload = raw.get("resolved_version_context")
            if supplied_payload is not None:
                if not isinstance(supplied_payload, Mapping) or not supplied_payload:
                    raise ValueError("resolved_version_context must be a non-empty mapping when supplied")
                supplied_context = ResolvedVersionContext.from_dict(supplied_payload)
                adapter.version_context.assert_context(supplied_context.context_id)
    else:
        target = {
            "minecraft_version": "unresolved",
            "loader": "unresolved",
            "source_api_family": "unresolved",
        }

    policy = (
        "preserve"
        if raw.get("preserved_existing_target")
        else "migrate"
        if raw.get("migration_requested")
        else "new"
    )
    optimizer = _mapping(raw.get("optimizer"))
    inventory = _mapping(existing_inventory)
    inventory_target = _mapping(inventory.get("target"))
    inventory_modules = (
        inventory.get("modules") if isinstance(inventory.get("modules"), list) else []
    )
    topology_modules = [
        item
        for item in inventory_modules
        if isinstance(item, Mapping)
        and not (
            len(inventory_modules) > 1
            and str(item.get("module_id") or "") == ":"
            and not _strings(item.get("source_sets"))
        )
    ]
    raw_module_ids = [
        str(item.get("module_id") or "")
        for item in topology_modules
        if str(item.get("module_id") or "")
    ]
    project_topology = _planner_project_topology(
        raw_module_ids,
        loaders=_strings(inventory_target.get("loaders")),
        source_sets=sorted(
            {
                str(source_set)
                for item in topology_modules
                for source_set in _strings(item.get("source_sets"))
            }
        ),
        gradle_project_paths=raw_module_ids,
    )
    supplied_topology = _mapping(raw.get("project_topology"))
    if supplied_topology:
        project_topology = _planner_project_topology(
            _strings(supplied_topology.get("module_ids")),
            loaders=_strings(supplied_topology.get("loaders")),
            source_sets=_strings(supplied_topology.get("source_sets")),
            gradle_project_paths=_strings(
                supplied_topology.get("gradle_project_paths")
            ),
        )

    rejected: list[dict[str, Any]] = []
    candidates = optimizer.get("candidates")
    if isinstance(candidates, list):
        selected_key = (
            str(target.get("minecraft_version") or ""),
            str(target.get("loader") or "").casefold(),
        )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            candidate_target = _mapping(candidate.get("target"))
            key = (
                str(candidate_target.get("minecraft_version") or ""),
                str(candidate_target.get("loader") or "").casefold(),
            )
            if key != selected_key:
                rejected.append(
                    {
                        "target": candidate_target,
                        "total_expected_cost": candidate.get("total_expected_cost"),
                        "reason": "ranked_below_selected_after_hard_gates_and_verified_reuse",
                    }
                )

    resolved = bool(
        str(target.get("minecraft_version") or "").strip().casefold()
        not in {"", "unresolved"}
        and str(target.get("loader") or "").strip().casefold() not in {"", "unresolved"}
    )
    result: dict[str, Any] = {
        "policy": policy,
        "coordinates": target,
        "hard_gate_status": "passed" if resolved else "deferred",
        "preserved_existing_target": bool(raw.get("preserved_existing_target")),
        "migration_requested": bool(raw.get("migration_requested")),
        "decision_reason": str(
            raw.get("reason") or optimizer.get("selection_basis") or "host target input"
        ),
        "rejected_alternatives": rejected,
        "project_topology": project_topology,
        "evidence_refs": [f"platform-selection:{_payload_sha(raw)}"] if raw else [],
        "decision_sha256": "",
    }
    result["decision_sha256"] = _payload_sha(result)
    return result


def lock_from_adapter(adapter: TargetContract) -> PlatformLock:
    """Copy the already-validated provider receipt without re-resolving it.

    PlatformLock is the immutable execution boundary.  Dropping the extended receipt
    fields here used to turn a fully verified adapter back into a legacy/partial lock,
    which then failed native-name and receipt validation downstream.
    """

    adapter.validate()
    lock = PlatformLock(
        adapter_id=adapter.adapter_id,
        edition=adapter.edition,
        loader=adapter.loader,
        minecraft_version=adapter.minecraft_version,
        java_version=adapter.java_version,
        yarn_mappings=adapter.yarn_mappings,
        mappings_kind=adapter.mappings_kind,
        mappings_version=adapter.mappings_version,
        fabric_loader=adapter.fabric_loader,
        fabric_api=adapter.fabric_api,
        fabric_loom=adapter.fabric_loom,
        gradle=adapter.gradle,
        gradle_sha256=adapter.gradle_sha256,
        gradle_distribution_url=(
            f"https://services.gradle.org/distributions/gradle-{adapter.gradle}-bin.zip"
        ),
        data_pack_version=adapter.data_pack_version,
        resource_pack_version=adapter.resource_pack_version,
        resource_pack_format=adapter.resource_pack_format,
        release_metadata_url=adapter.release_metadata_url,
        source_api_family=adapter.source_api_family,
        deterministic_module_kinds=tuple(sorted(adapter.deterministic_module_kinds)),
        host_facts_json=adapter.host_facts_json,
    )
    lock = replace(lock, receipt_sha256=platform_receipt_sha256(lock))
    lock.validate()
    return lock


def resolve_platform(
    prompt: str,
    *,
    design: dict[str, Any] | None = None,
    module_kinds: Iterable[str] = (),
    existing_version: str | None = None,
    existing_loader: str | None = None,
    router: Any | None = None,
    target_research_fn: TargetResearchFn | None = None,
) -> PlatformSelection:
    """Compatibility entrypoint routed only to the canonical fail-closed selector."""

    del router
    from .platform_selection_pipeline import resolve_platform_fail_closed

    return resolve_platform_fail_closed(
        prompt,
        design=design,
        module_kinds=module_kinds,
        existing_version=existing_version,
        existing_loader=existing_loader,
        target_research_fn=target_research_fn,
    )


def _optimized_selection(
    optimization: PlatformOptimization,
    *,
    source: str,
    explicit_version: bool,
    explicit_loader: bool,
    migration_requested: bool,
) -> PlatformSelection:
    adapter = optimization.selected
    evidence = optimization.evidence
    return PlatformSelection(
        adapter=adapter,
        source=source,
        reason=(
            f"Verified evidence selected {adapter.minecraft_version}/{adapter.loader}: "
            f"reuse {evidence.reuse_coverage}/{len(evidence.requested_capabilities)}, "
            f"residual {evidence.residual_cost}, dependency closure "
            f"{'complete' if evidence.dependency_closure_complete else 'incomplete'}."
        ),
        explicit_version=explicit_version,
        explicit_loader=explicit_loader,
        migration_requested=migration_requested,
        optimization=optimization,
    )


def _exact_adapter(version: str, loader: str) -> TargetContract:
    try:
        return adapter_for_target(version, loader)
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc


def _existing_adapter(version: str, loader: str | None) -> TargetContract:
    if loader and str(loader).strip():
        return _exact_adapter(str(version), str(loader))
    candidates = adapters_for_version(str(version))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SpecValidationError(
            f"기존 프로젝트 Minecraft {version} target을 실행할 provider가 없습니다."
        )
    raise SpecValidationError(
        "기존 프로젝트 loader를 식별할 수 없고 같은 Minecraft 버전에 여러 실행 "
        "provider가 존재합니다. 기존 target을 추측하지 않습니다."
    )


def retarget_proposal(proposal: Proposal, selection: PlatformSelection) -> Proposal:
    from .knowledge import evidence_for_target, evidence_snapshot_hash

    spec = replace(proposal.spec, platform=selection.lock)
    evidence_terms: list[str] = [proposal.requested_prompt, "project build metadata dependency"]
    if proposal.spec.boss is not None:
        evidence_terms.append("boss entity gametest runtime server")
    if proposal.spec.contents:
        evidence_terms.append("item block recipe data generation resource")
    evidence = evidence_for_target(
        " ".join(evidence_terms),
        minecraft_version=selection.adapter.minecraft_version,
    )
    assumptions = tuple(
        value
        for value in proposal.assumptions
        if not ("Minecraft Java Edition" in value or "Minecraft " in value and "Fabric" in value)
    ) + (
        (
            f"Target: Minecraft Java {selection.adapter.minecraft_version}, "
            f"{selection.adapter.loader}, Java {selection.adapter.java_version}. "
            f"{selection.reason}"
        ),
    )
    return replace(
        proposal,
        spec=spec,
        assumptions=assumptions,
        evidence_sources=evidence,
        evidence_snapshot_hash=evidence_snapshot_hash(evidence),
        approval_hash="",
    ).with_hash()


def supported_target_summary() -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    for loader, version in discover_target_keys(limit_per_loader=32):
        result.append(
            {
                "minecraft_version": version,
                "loader": loader,
                "provider": provider_for_loader(loader).provider_id,
            }
        )
    return tuple(result)


def _explicit_minecraft_version(prompt: str) -> str | None:
    matches = _VERSION_RE.findall(prompt)
    if not matches:
        return None
    return list(dict.fromkeys(matches))[-1]


def _explicit_loader(prompt: str) -> str | None:
    found: list[str] = []
    if _FABRIC_RE.search(prompt):
        found.append("fabric")
    if _NEOFORGE_RE.search(prompt):
        found.append("neoforge")
    if _FORGE_RE.search(prompt):
        found.append("forge")
    unique = list(dict.fromkeys(found))
    if len(unique) > 1:
        raise SpecValidationError(f"하나의 프로젝트에 여러 로더가 동시에 명시되었습니다: {unique}")
    return unique[0] if unique else None


def _require_supported_kinds(
    adapter: TargetContract,
    module_kinds: Iterable[str],
    *,
    explicit: bool,
) -> None:
    if adapter.host_facts_json:
        context = adapter.version_context
        for kind in module_kinds:
            context.require_capability(str(kind))
        return
    if adapter.source_api_family == "fabric_live_ai":
        return
    kinds = {str(value).strip() for value in module_kinds if str(value).strip()}
    unsupported = sorted(kinds - adapter.deterministic_module_kinds)
    if not unsupported:
        return
    prefix = "명시한 target" if explicit else "선택된 target"
    raise SpecValidationError(
        f"{prefix} {adapter.minecraft_version}/{adapter.loader}의 deterministic legacy "
        f"generator가 지원하지 않는 종류가 있습니다: {unsupported}."
    )


def executable_target_names() -> tuple[str, ...]:
    return tuple(
        f"{version}/{loader}"
        for loader in executable_loaders()
        for candidate_loader, version in discover_target_keys(loader=loader, limit_per_loader=32)
        if candidate_loader == loader
    )


__all__ = [
    "PlatformSelection",
    "compile_target_decision",
    "executable_target_names",
    "lock_from_adapter",
    "resolve_platform",
    "retarget_proposal",
    "supported_target_summary",
]
