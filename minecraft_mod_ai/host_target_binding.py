from __future__ import annotations

"""Host-owned existing-project and platform target binding.

This module deliberately contains no gameplay planning, model generation, repair,
or implementation fallback. It binds immutable host observations to an authored
Typed PlanIR production request.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from . import central_research
from .model_concurrency import planning_work_unit_timeout_seconds
from .platform_resolver import retarget_proposal
from .spec import Proposal, SpecValidationError


def _bounded_future_result(future: Any, *, operation: str) -> Any:
    timeout = planning_work_unit_timeout_seconds()
    try:
        return future.result(timeout=timeout)
    except TypeError as exc:
        raise SpecValidationError(
            f"{operation} future does not support bounded result(timeout=...)"
        ) from exc
    except TimeoutError as exc:
        cancel = getattr(future, "cancel", None)
        if callable(cancel):
            cancel()
        raise SpecValidationError(
            f"{operation} exceeded the {timeout:.3f}s host work-unit deadline"
        ) from exc


def bind_existing_project(
    router: Any,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(design)
    existing_report = getattr(router, "_mmm_existing_project_report", None)
    if isinstance(existing_report, Mapping):
        result["_existing_project_report"] = dict(existing_report)

    existing_inventory = getattr(router, "_mmm_existing_project_inventory", None)
    inventory_future = getattr(
        router,
        "_mmm_existing_project_inventory_future",
        None,
    )
    if existing_inventory is None and hasattr(inventory_future, "result"):
        inventory = _bounded_future_result(
            inventory_future,
            operation="existing project inventory",
        )
        validate = getattr(inventory, "validate", None)
        to_dict = getattr(inventory, "to_dict", None)
        if callable(validate) and callable(to_dict):
            validate()
            existing_inventory = to_dict()
            router._mmm_existing_project_inventory = existing_inventory

    if isinstance(existing_inventory, Mapping):
        from .project_inventory import validate_project_inventory_payload

        inventory_payload = validate_project_inventory_payload(existing_inventory)
        router._mmm_existing_project_inventory = inventory_payload
        result.update({
            "_existing_project_inventory": inventory_payload,
            "_existing_snapshot": inventory_payload,
            "_component_catalog": dict(
                inventory_payload.get("component_catalog") or {}
            ),
        })
    return result


def bind_platform(
    router: Any,
    prompt: str,
    design: Mapping[str, Any],
    base_proposal: Proposal,
    *,
    module_kinds: Iterable[str] = (),
) -> tuple[dict[str, Any], Proposal]:
    from .platform_selection_pipeline import resolve_platform_fail_closed
    from .platform_target_research import target_research_callback

    existing_version = getattr(router, "_mmm_existing_minecraft_version", None)
    existing_loader = getattr(router, "_mmm_existing_loader", None)
    requested_version = getattr(router, "_mmm_requested_minecraft_version", None)
    requested_loader = getattr(router, "_mmm_requested_loader", None)

    effective_prompt = str(prompt)
    if requested_version and str(requested_version) not in effective_prompt:
        effective_prompt += (
            f"\n[HOST_TARGET_CONSTRAINT Minecraft {requested_version}]"
        )
    if (
        requested_loader
        and str(requested_loader).casefold() not in effective_prompt.casefold()
    ):
        effective_prompt += f"\n[HOST_LOADER_CONSTRAINT {requested_loader}]"

    research_brief = design.get("_research_brief")
    if not isinstance(research_brief, dict):
        research_brief = central_research.normalize_research_brief(
            prompt,
            dict(design),
        )

    target_research = target_research_callback(research_brief)
    selection = resolve_platform_fail_closed(
        effective_prompt,
        design=dict(design),
        module_kinds=module_kinds,
        existing_version=existing_version,
        existing_loader=existing_loader,
        target_research_fn=target_research,
    )
    proposal = retarget_proposal(base_proposal, selection)
    proposal.validate()

    selection_dict = selection.to_dict()
    if selection.migration_requested and existing_version:
        selection_dict["migration_from"] = {
            "minecraft_version": str(existing_version),
            "loader": str(existing_loader or "unknown").strip().casefold(),
        }

    target = dict(selection_dict["target"])
    bound_brief = {**research_brief, "_mmm_platform_target": target}

    platform_evidence: Mapping[str, Any] | None = None
    if selection.optimization is not None:
        deep = selection.optimization.evidence.deep_research
        if isinstance(deep, Mapping):
            platform_evidence = dict(deep)
    if platform_evidence is None:
        value = target_research(selection.adapter)
        if isinstance(value, Mapping):
            platform_evidence = dict(value)
    if platform_evidence is None:
        platform_evidence = {
            "schema_version": "mmm/platform-evidence-host-v1",
            "status": "host_target_resolved",
            "target": target,
        }

    result = {
        **dict(design),
        "_platform_selection": selection_dict,
        "_platform_evidence": dict(platform_evidence),
        "_research_brief": bound_brief,
    }
    return result, proposal


__all__ = ["bind_existing_project", "bind_platform"]
