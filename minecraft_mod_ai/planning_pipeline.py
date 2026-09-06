from __future__ import annotations

"""Prompt-first compiler-owned planning pipeline.

The production order is now: prompt state -> grounded reference/scope research ->
researched requirements -> grounded implementation research -> detailed plan -> legacy
catalog/proposal lowering -> target binding. Raw prompt text is never compiled directly
into implementation/search tasks.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

from . import central_research
from .model_router import ModelRouter
from .planner import _proposal_from_model_data
from .platform_resolver import retarget_proposal
from .spec import Proposal, SpecValidationError


class PlanningStage(str, Enum):
    REQUEST = "request"
    RESEARCH = "research"
    DESIGN = "design"
    PRE_RETRIEVAL_PLAN = "pre_retrieval_plan"
    PLATFORM = "platform"
    EVIDENCE = "evidence"


class PlanningStageError(SpecValidationError):
    def __init__(
        self,
        stage: PlanningStage,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        self.stage = stage
        self.cause_type = type(cause).__name__ if cause is not None else ""
        suffix = f" ({self.cause_type}: {cause})" if cause is not None else ""
        super().__init__(f"planning host invariant {stage.value}: {message}{suffix}")


@dataclass(frozen=True)
class PlanningArtifacts:
    planning_state: dict[str, Any]
    game_design: dict[str, Any]
    base_proposal: Proposal
    research_brief: dict[str, Any]
    technical_evidence: dict[str, Any]


class PlanningPipeline:
    """Compile an authored request into an implementation-ready grounded contract."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self.planning_state: dict[str, Any] | None = None

    def prepare(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_state: Mapping[str, Any] | None = None,
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> PlanningArtifacts:
        if not str(prompt).strip():
            raise PlanningStageError(PlanningStage.REQUEST, "prompt is empty")

        from .planning_state_pipeline import prepare_planning_state

        if existing_state is None and self.planning_state is not None:
            if self.planning_state.get("original_prompt") == prompt:
                existing_state = self.planning_state

        def save_state(value: dict[str, Any]) -> None:
            self.planning_state = deepcopy(value)
            if checkpoint is not None:
                checkpoint(deepcopy(value))

        try:
            planning_state = prepare_planning_state(
                self.router, prompt, existing_state=existing_state, checkpoint=save_state
            )
        except Exception as exc:
            raise PlanningStageError(
                PlanningStage.RESEARCH,
                "prompt-first research state did not reach code-ready coverage",
                cause=exc,
            ) from exc

        game_design, base_proposal = self._semantic_design(
            prompt,
            planning_state=planning_state,
            media_paths=media_paths,
        )
        game_design, base_proposal, research_brief, platform_evidence = self._bind_platform(
            prompt,
            game_design,
            base_proposal,
        )
        technical_evidence = self._validated_evidence(platform_evidence)
        return PlanningArtifacts(
            planning_state=planning_state,
            game_design=game_design,
            base_proposal=base_proposal,
            research_brief=research_brief,
            technical_evidence=technical_evidence,
        )

    def _semantic_design(
        self,
        prompt: str,
        *,
        planning_state: Mapping[str, Any],
        media_paths: Sequence[str | Path],
    ) -> tuple[dict[str, Any], Proposal]:
        """Lower the already researched state into existing proposal/design shapes."""
        del media_paths
        from . import agentic_research_game_design as host_design
        from .planning_authority import (
            authoritative_request_scope,
            build_authoritative_request_catalog,
        )
        from .reuse_planner import compile_pre_retrieval_plan

        request_catalog = build_authoritative_request_catalog(
            prompt,
            self.router,
            planning_state=planning_state,
        )
        with authoritative_request_scope(
            prompt,
            request_catalog,
            planning_state=planning_state,
        ):
            design = host_design.generate_sectioned_game_design(
                self.router,
                prompt,
                research={"planning_state_sha256": planning_state.get("state_sha256")},
            )
            design = host_design.validate_ready_design(
                prompt,
                host_design.canonical_game_design(design),
            )
            design = self._bind_existing_project(design)
            design = {
                **design,
                "_evidence_request_catalog": request_catalog,
                "_planning_state": dict(planning_state),
            }
            # This reuse plan is now compiled only after two research passes and the
            # detailed-plan coverage gate. It is no longer a raw-prompt pre-research plan.
            pre_retrieval_plan = compile_pre_retrieval_plan(prompt, design)
            design = {**design, "_pre_retrieval_plan": pre_retrieval_plan}
            research_brief = central_research.normalize_research_brief(prompt, design)
            design = {
                **design,
                "_research_brief": research_brief,
                "_planning_authority": {
                    "owner": "prompt_first_grounded_state_machine",
                    "planning_state_sha256": planning_state.get("state_sha256", ""),
                    "request_catalog_sha256": request_catalog.get("catalog_sha256", ""),
                    "raw_prompt_compiler": False,
                    "model_generated_json": True,
                },
            }

        build_slice = host_design.deterministic_bootstrap(prompt, design)
        proposal = _proposal_from_model_data(prompt, build_slice)
        if proposal.requested_prompt != prompt:
            proposal = replace(
                proposal,
                requested_prompt=prompt,
                approval_hash="",
            ).with_hash()
        return design, proposal

    def _bind_existing_project(self, design: dict[str, Any]) -> dict[str, Any]:
        existing_report = getattr(self.router, "_mmm_existing_project_report", None)
        if isinstance(existing_report, Mapping):
            design = {**design, "_existing_project_report": dict(existing_report)}

        existing_inventory = getattr(self.router, "_mmm_existing_project_inventory", None)
        inventory_future = getattr(
            self.router,
            "_mmm_existing_project_inventory_future",
            None,
        )
        if existing_inventory is None and hasattr(inventory_future, "result"):
            inventory = inventory_future.result()
            validate = getattr(inventory, "validate", None)
            to_dict = getattr(inventory, "to_dict", None)
            if callable(validate) and callable(to_dict):
                validate()
                existing_inventory = to_dict()
                self.router._mmm_existing_project_inventory = existing_inventory

        if isinstance(existing_inventory, Mapping):
            from .project_inventory import validate_project_inventory_payload

            inventory_payload = validate_project_inventory_payload(existing_inventory)
            self.router._mmm_existing_project_inventory = inventory_payload
            design = {
                **design,
                "_existing_project_inventory": inventory_payload,
                "_existing_snapshot": inventory_payload,
                "_component_catalog": dict(inventory_payload.get("component_catalog") or {}),
            }
        return design

    def _bind_platform(
        self,
        prompt: str,
        design: dict[str, Any],
        base_proposal: Proposal,
    ) -> tuple[dict[str, Any], Proposal, dict[str, Any], dict[str, Any]]:
        from .platform_selection_pipeline import resolve_platform_fail_closed
        from .platform_target_research import target_research_callback

        existing_version = getattr(self.router, "_mmm_existing_minecraft_version", None)
        existing_loader = getattr(self.router, "_mmm_existing_loader", None)
        requested_version = getattr(self.router, "_mmm_requested_minecraft_version", None)
        requested_loader = getattr(self.router, "_mmm_requested_loader", None)
        effective_prompt = str(prompt)
        if requested_version and str(requested_version) not in effective_prompt:
            effective_prompt += f"\n[HOST_TARGET_CONSTRAINT Minecraft {requested_version}]"
        if requested_loader and str(requested_loader).casefold() not in effective_prompt.casefold():
            effective_prompt += f"\n[HOST_LOADER_CONSTRAINT {requested_loader}]"

        research_brief = design.get("_research_brief")
        if not isinstance(research_brief, dict):
            research_brief = central_research.normalize_research_brief(prompt, design)
        target_research = target_research_callback(research_brief)
        selection = resolve_platform_fail_closed(
            effective_prompt,
            design=design,
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

        from .grounded_source_reuse import build_repository_reuse_plan

        reuse_design = {**design, "_platform_selection": selection_dict}
        selection_dict["reuse_plan"] = build_repository_reuse_plan(reuse_design)
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
                "schema_version": "mmm/platform-evidence-local-fallback-v1",
                "status": "host_target_resolved_without_external_evidence",
                "target": target,
            }

        result = {
            **design,
            "_platform_selection": selection_dict,
            "_platform_evidence": dict(platform_evidence),
            "_research_brief": bound_brief,
        }
        return result, proposal, bound_brief, dict(platform_evidence)

    @staticmethod
    def _validated_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {
                "schema_version": "mmm/technical-evidence-local-fallback-v1",
                "status": "host_only",
            }
        payload = dict(value)
        if not str(payload.get("schema_version") or "").strip():
            payload["schema_version"] = "mmm/technical-evidence-host-normalized-v1"
        return payload


__all__ = ["PlanningArtifacts", "PlanningPipeline", "PlanningStage", "PlanningStageError"]
