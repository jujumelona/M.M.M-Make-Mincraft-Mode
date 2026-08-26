from __future__ import annotations

"""Install research-backed reuse/version planning and fixed pixel-asset production.

Pre-bootstrap installation is deliberate: existing runtime contracts must wrap the
upgraded raw asset producer so Qwen GPU handoff, model parking and recovery keep one
owner. Post-bootstrap installation only adds verified-result promotion.
"""

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

from .component_registry import persist_promotions, promotion_records
from .resource_asset_production import (
    attach_generation_plan,
    bind_reuse_plan,
    install_prebootstrap_asset_runtime,
)
from .reuse_planner import (
    decompose_capability_graph,
    optimize_platform_and_reuse,
    plan_fixed_target,
)
from .source_transplant import materialize_source_slices

_MATERIALIZE_LOCK = threading.RLock()
_MATERIALIZE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_MATERIALIZE_KEY_LOCKS: dict[tuple[str, str], tuple[threading.Lock, int]] = {}
_TRANSPORT_CONFIG_KEYS = frozenset(
    {
        "_approved_reuse_plan",
        "_owned_reuse_plan",
        "_reuse_materialization",
        "_donor_source_excerpts",
    }
)


def install_prebootstrap() -> None:
    """Upgrade owner boundaries before runtime contracts capture/wrap them."""

    from . import complete_planner, custom_module_generator, platform_resolver

    _install_joint_platform_optimizer(platform_resolver)
    _install_reuse_aware_resolver(platform_resolver)
    _install_complete_planning_handoff(complete_planner)
    _install_reuse_materialization(custom_module_generator)
    install_prebootstrap_asset_runtime()


def install_postbootstrap() -> None:
    """Add only promotion after the normal runtime bootstrap has finished."""

    from . import complete_orchestrator

    _install_verified_promotion(complete_orchestrator)


def _install_joint_platform_optimizer(resolver: Any) -> None:
    """Select automatic/new targets only through evidence-first reuse evaluation.

    Existing-project preservation remains owned by ``platform_resolver``. Every
    automatic/new target flows through this hook without a Minecraft-version filter.
    An explicit ``MMM_ECOSYSTEM_DISCOVERY=off`` opt-out keeps the canonical host
    optimizer available, but an evidence-first optimization failure must not silently
    select a target through the legacy optimizer.
    """

    current = resolver._optimize
    if getattr(current, "_mmm_joint_reuse_optimizer", False):
        return

    @wraps(current)
    def optimize(
        prompt: str,
        *,
        design: dict[str, Any] | None,
        module_kinds: Sequence[str],
        loader_constraint: str | None = None,
        version_constraint: str | None = None,
        target_research_fn: Any | None = None,
    ):
        del version_constraint

        def base_optimizer():
            return current(
                prompt,
                design=design,
                module_kinds=module_kinds,
                loader_constraint=loader_constraint,
                version_constraint=None,
                target_research_fn=target_research_fn,
            )

        discovery_mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
        if discovery_mode == "off":
            return base_optimizer()
        joint = optimize_platform_and_reuse(
            prompt,
            design=design,
            module_kinds=module_kinds,
            loader_constraint=loader_constraint,
            version_constraint=None,
            target_research_fn=target_research_fn,
        )
        result = joint.base_optimization
        object.__setattr__(result, "_mmm_reuse_plan", joint.selected_plan.to_dict())
        return result

    optimize._mmm_joint_reuse_optimizer = True
    resolver._optimize = optimize


def _install_reuse_aware_resolver(resolver: Any) -> None:
    current = resolver.resolve_platform
    if getattr(current, "_mmm_joint_reuse_platform", False):
        return

    original_to_dict = resolver.PlatformSelection.to_dict
    if not getattr(original_to_dict, "_mmm_reuse_payload", False):

        @wraps(original_to_dict)
        def to_dict(self: Any) -> dict[str, Any]:
            payload = original_to_dict(self)
            target = payload.get("target")
            if isinstance(target, dict):
                target["resource_pack_format"] = int(self.adapter.resource_pack_format)
            plan = getattr(self, "_mmm_reuse_plan", None)
            if isinstance(plan, Mapping):
                payload["reuse_plan"] = dict(plan)
            return payload

        to_dict._mmm_reuse_payload = True
        resolver.PlatformSelection.to_dict = to_dict

    @wraps(current)
    def resolve_platform(
        prompt: str,
        *,
        design: dict[str, Any] | None = None,
        module_kinds=(),
        existing_version: str | None = None,
        existing_loader: str | None = None,
        router: Any | None = None,
        target_research_fn: Any | None = None,
    ):
        text = str(prompt or "")
        kinds = tuple(str(value).strip() for value in module_kinds if str(value).strip())
        selection = current(
            text,
            design=design,
            module_kinds=kinds,
            existing_version=existing_version,
            existing_loader=existing_loader,
            router=router,
            target_research_fn=target_research_fn,
        )
        attached = (
            getattr(selection.optimization, "_mmm_reuse_plan", None)
            if selection.optimization is not None
            else None
        )
        if _plan_matches_adapter(attached, selection.adapter):
            plan_payload = dict(attached)
        else:
            graph = decompose_capability_graph(text, design=design, module_kinds=kinds)
            evidence = selection.optimization.evidence if selection.optimization is not None else None
            allow_network = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower() != "off"
            plan_payload = plan_fixed_target(
                selection.adapter,
                capabilities=graph.nodes,
                design=design,
                platform_evidence=evidence,
                allow_network=allow_network,
                capability_graph=graph.to_dict(),
            ).to_dict()
        object.__setattr__(selection, "_mmm_reuse_plan", plan_payload)
        return selection

    resolve_platform._mmm_joint_reuse_platform = True
    resolver.resolve_platform = resolve_platform


def _plan_matches_adapter(plan: Any, adapter: Any) -> bool:
    if not isinstance(plan, Mapping):
        return False
    target = plan.get("target")
    if not isinstance(target, Mapping):
        return False
    return (
        str(target.get("minecraft_version") or "") == str(adapter.minecraft_version)
        and str(target.get("loader") or "").casefold() == str(adapter.loader).casefold()
    )


def _install_complete_planning_handoff(complete_planner: Any) -> None:
    cls = complete_planner.CompleteGameDesignPlanner
    current = cls._plan_in_session
    if getattr(current, "_mmm_reuse_asset_planning", False):
        return

    @wraps(current)
    def plan_in_session(self: Any, *args: Any, **kwargs: Any):
        proposal = current(self, *args, **kwargs)
        proposal = bind_reuse_plan(proposal)
        return attach_generation_plan(self.router, proposal)

    plan_in_session._mmm_reuse_asset_planning = True
    cls._plan_in_session = plan_in_session


def _install_reuse_materialization(custom_module_generator: Any) -> None:
    cls = custom_module_generator.CustomModuleGenerator
    current = cls.generate
    if getattr(current, "_mmm_source_transplant_materialization", False):
        return

    @wraps(current)
    def generate(self: Any, project_root: str | Path, *, module: Any, **kwargs: Any):
        config = module.config if isinstance(module.config, Mapping) else {}
        reuse_plan = config.get("_owned_reuse_plan")
        if isinstance(reuse_plan, Mapping) and reuse_plan.get("capabilities"):
            receipt = _materialize_once(project_root, reuse_plan)
            donor_context = _materialized_donor_context(receipt)
            decisions = reuse_plan.get("capabilities")
            fresh: list[str] = []
            adapters: list[str] = []
            compact_decisions: list[dict[str, str]] = []
            if isinstance(decisions, Sequence) and not isinstance(decisions, (str, bytes)):
                for item in decisions:
                    if not isinstance(item, Mapping):
                        continue
                    capability = str(item.get("capability") or "").strip()
                    mode = str(item.get("mode") or "").strip()
                    source_id = str(item.get("source_id") or "").strip()
                    if capability:
                        compact_decisions.append(
                            {"capability": capability, "mode": mode, "source_id": source_id}
                        )
                    if capability and mode == "fresh":
                        fresh.append(capability)
                    elif capability and mode in {"adapt", "source_transplant"}:
                        adapters.append(capability)
            model_config = {
                key: value
                for key, value in dict(config).items()
                if key not in _TRANSPORT_CONFIG_KEYS
            }
            model_config.update(
                {
                    "_reuse_decisions": compact_decisions,
                    "_source_transplant_context": donor_context,
                    "_fresh_only_capabilities": fresh,
                    "_adapter_capabilities": adapters,
                    "_generation_rule": (
                        "Generate only the capabilities listed in _fresh_only_capabilities and "
                        "the minimal project-local integration/adaptation work listed in "
                        "_adapter_capabilities. A source_transplant decision is pinned donor "
                        "evidence, not a completed project implementation: use its source context "
                        "and implement only the residual integration delta. Read pinned donor source "
                        "on demand with read_reuse_source instead of asking the model to recreate it. Reuse approved "
                        "same-project/MMM/library capabilities exactly as the compact reuse "
                        "decisions specify; do not reimplement them."
                    ),
                }
            )
            module = replace(module, config=model_config)
        return current(self, project_root, module=module, **kwargs)

    generate._mmm_source_transplant_materialization = True
    cls.generate = generate


def _materialize_once(project_root: str | Path, reuse_plan: Mapping[str, Any]) -> dict[str, Any]:
    root = str(Path(project_root).expanduser().resolve())
    encoded = json.dumps(reuse_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = (root, hashlib.sha256(encoded).hexdigest())
    with _MATERIALIZE_LOCK:
        cached = _MATERIALIZE_CACHE.get(key)
        if cached is not None:
            return dict(cached)
        tracked = _MATERIALIZE_KEY_LOCKS.get(key)
        if tracked is None:
            key_lock = threading.Lock()
            users = 0
        else:
            key_lock, users = tracked
        _MATERIALIZE_KEY_LOCKS[key] = (key_lock, users + 1)
    try:
        with key_lock:
            with _MATERIALIZE_LOCK:
                cached = _MATERIALIZE_CACHE.get(key)
            if cached is not None:
                return dict(cached)
            receipt = materialize_source_slices(root, reuse_plan)
            with _MATERIALIZE_LOCK:
                _MATERIALIZE_CACHE[key] = dict(receipt)
                while len(_MATERIALIZE_CACHE) > 64:
                    _MATERIALIZE_CACHE.pop(next(iter(_MATERIALIZE_CACHE)))
            return dict(receipt)
    finally:
        with _MATERIALIZE_LOCK:
            tracked = _MATERIALIZE_KEY_LOCKS.get(key)
            if tracked is not None:
                tracked_lock, users = tracked
                if users <= 1:
                    _MATERIALIZE_KEY_LOCKS.pop(key, None)
                else:
                    _MATERIALIZE_KEY_LOCKS[key] = (tracked_lock, users - 1)


def _materialized_donor_context(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose only donor manifests; source text is fetched on demand by read_reuse_source."""
    context: list[dict[str, Any]] = []
    donors = receipt.get("donors")
    if not isinstance(donors, Sequence) or isinstance(donors, (str, bytes)):
        return context
    for donor in donors:
        if not isinstance(donor, Mapping):
            continue
        files = donor.get("files")
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
            continue
        for item in files:
            if not isinstance(item, Mapping):
                continue
            context.append(
                {
                    "repository": donor.get("repository"),
                    "commit_sha": donor.get("commit_sha"),
                    "license_id": donor.get("license_id"),
                    "capability": donor.get("capability"),
                    "path": item.get("path"),
                    "sha256": item.get("sha256"),
                    "size_bytes": item.get("size_bytes"),
                    "read_tool": "read_reuse_source",
                }
            )
    return context

def _install_verified_promotion(orchestrator: Any) -> None:
    cls = orchestrator.CompleteProductionOrchestrator
    current = cls.execute
    if getattr(current, "_mmm_verified_reuse_promotion", False):
        return

    @wraps(current)
    def execute(self: Any, proposal: Any, *args: Any, **kwargs: Any):
        result = current(self, proposal, *args, **kwargs)
        if result.release_ready and not result.unresolved_gates:
            parsed = proposal
            if not isinstance(parsed, orchestrator.CompleteProposal):
                parsed = orchestrator.CompleteProposal.from_dict(parsed)
            records = promotion_records(proposal=parsed, result=result)
            if records:
                persist_promotions(result.project_root, records)
        return result

    execute._mmm_verified_reuse_promotion = True
    cls.execute = execute


__all__ = ["install_postbootstrap", "install_prebootstrap"]
