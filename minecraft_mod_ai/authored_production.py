"""Pass a saved design to the implementation agent without planning it again."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .authored_plan import AuthoredPlan
from .complete_spec import CompleteProposal, ProductionModule, complete_proposal_from_parts
from .planning_pipeline import PlanningPipeline
from .spec import ModSpec, Proposal, ProposalStatus


_TARGET_KEYS = ("minecraft_version", "loader", "mappings")


def _require_bound_target(design: Mapping[str, Any]) -> dict[str, str]:
    """Return the host-selected platform triple produced by ``_bind_platform``.

    Saved authored designs bypass the normal planning path, so the production module
    must carry the same host-selected target contract that normal production receives.
    Do not infer or default any of these values here: an incomplete binding is an
    upstream contract violation and must fail before an RAG worker is submitted.
    """
    candidates: list[Mapping[str, Any]] = [design]
    for key in ("platform", "target", "toolchain", "build", "existing_project"):
        value = design.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)

    for candidate in candidates:
        if all(candidate.get(key) not in (None, "") for key in _TARGET_KEYS):
            return {key: str(candidate[key]) for key in _TARGET_KEYS}

    raise ValueError(
        "Saved authored production requires the host-selected minecraft_version, "
        "loader and mappings returned by platform binding."
    )


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
    target = _require_bound_target(design)
    design = {**design, **target}

    return complete_proposal_from_parts(
        requested_prompt=plan.requested_prompt,
        base_proposal=base,
        game_design=design,
        modules=(ProductionModule(
            module_id="authored_design",
            kind="custom_java",
            config={
                "implementation": "custom",
                "authored_plan": plan.to_dict(),
                **target,
            },
        ),),
        acceptance_tests=acceptance,
        existing_input_sha256=existing_input_sha256 or plan.existing_input_sha256,
    )
