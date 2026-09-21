"""Pass a saved design to the implementation agent without planning it again."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .authored_plan import AuthoredPlan
from .complete_spec import (
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .planning_pipeline import PlanningPipeline
from .spec import ModSpec, Proposal, ProposalStatus
from .target_contract import TargetContractError, target_coordinates_from_mapping

_TARGET_KEYS = ("minecraft_version", "loader", "mappings")


def _bound_target(design: Mapping[str, Any]) -> dict[str, str]:
    """Return the complete host-selected target, or no target when none exists.

    The platform selector owns the target. Decode its receipt through the same
    contract used by generation, including native names and mapping receipt objects.
    Older saved designs without a selection may still carry standalone coordinates.
    """
    candidates: list[Mapping[str, Any]] = [design]
    for key in ("platform", "target", "toolchain", "build", "existing_project"):
        value = design.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)

    selection = design.get("_platform_selection")
    if isinstance(selection, Mapping) and "target" in selection:
        target = selection["target"]
        if not isinstance(target, Mapping):
            raise ValueError("Saved authored platform selection target must be an object.")
        # Never let stale design/existing-project fields override the selected target,
        # including when the selected receipt is invalid.
        coordinates = target_coordinates_from_mapping(target)
        return {key: getattr(coordinates, key) for key in _TARGET_KEYS}
    if isinstance(selection, Mapping):
        candidates.append(selection)

    first_error: TargetContractError | None = None
    for candidate in candidates:
        if not any(candidate.get(key) not in (None, "") for key in (
            *_TARGET_KEYS, "mappings_version", "yarn_mappings",
        )):
            continue
        try:
            coordinates = target_coordinates_from_mapping(candidate)
        except TargetContractError as exc:
            if first_error is None:
                first_error = exc
            continue
        return {key: getattr(coordinates, key) for key in _TARGET_KEYS}

    if first_error is not None:
        raise first_error
    return {}


def compile_authored_design(
    router: Any, plan: AuthoredPlan, *, existing_input_sha256: str = ""
) -> CompleteProposal:
    # These are host project coordinates, not inferred gameplay or placeholder content.
    mod_id = "authored_" + plan.calculate_hash()[:12]
    acceptance = (
        "Implement the behaviors in the saved authored design and exercise them in Minecraft.",
        "Build the project and verify that the mod loads and runs without errors.",
    )
    base = Proposal(
        schema_version="minecraft-mod-ai/proposal-v1",
        proposal_version=1,
        status=ProposalStatus.AWAITING_APPROVAL,
        requested_prompt=plan.requested_prompt,
        spec=ModSpec(
            mod_id=mod_id,
            mod_name="Authored Minecraft Mod",
            package_name=f"ai.minecraft.generated.{mod_id}",
            version="1.0.0",
            summary=plan.requested_prompt,
            contents=(),
        ),
        assumptions=(), exclusions=(), deferred_requests=(),
        acceptance_tests=acceptance, evidence_sources=(),
    )
    design = {"authored_plan": plan.to_dict()}
    # Bind the actual build toolchain and existing project only. Never enter prepare(),
    # requirement extraction, design validation, or the old PlanIR compiler.
    binding = PlanningPipeline(router)
    design = binding._bind_existing_project(design)
    design, base, _, _ = binding._bind_platform(plan.requested_prompt, design, base)

    # The saved-plan route previously preserved the binding only in game_design while
    # its production module dropped the target triple. Official RAG validates the
    # production-side host contract, so make the bound target explicit at that boundary.
    target = _bound_target(design)
    design = {**design, **target}
    module_config: dict[str, Any] = {
        "implementation": "custom",
        "authored_plan": plan.to_dict(),
        **target,
    }
    if not (existing_input_sha256 or plan.existing_input_sha256):
        # New authored projects already have a host-generated stable package. Carry it
        # into mutation authority so a small coder cannot invent unrelated Java roots
        # across independent authored fragments.
        module_config["authored_java_package"] = base.spec.package_name

    return complete_proposal_from_parts(
        requested_prompt=plan.requested_prompt,
        base_proposal=base,
        game_design=design,
        modules=(ProductionModule(
            module_id="authored_design",
            kind="custom_java",
            config=module_config,
            required_gates=("project build",),
        ),),
        acceptance_tests=acceptance,
        existing_input_sha256=existing_input_sha256 or plan.existing_input_sha256,
    )
