from __future__ import annotations

"""Prompt-first compiler-owned planning pipeline.

The production order is now: prompt state -> grounded reference/scope research ->
researched requirements -> grounded implementation research -> detailed plan -> legacy
catalog/proposal lowering -> target binding. Raw prompt text is never compiled directly
into implementation/search tasks.
"""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from . import central_research
from .model_router import ModelRouter
from .planner import _proposal_from_model_data
from .platform_resolver import retarget_proposal
from .root_cause_trace import traced_callable
from .spec import Proposal, SpecValidationError

if TYPE_CHECKING:
    from .planning_state_pipeline import DetailSectionApplicabilityResolver

_T = TypeVar("_T")


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


def _host_operation(operation: str, callback: Callable[[], _T]) -> _T:
    """Route every major planning host boundary through one trace implementation."""

    return traced_callable(callback, stage="planning", operation=operation)()


class PlanningPipeline:
    """Compile an authored request into an implementation-ready grounded contract."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self.planning_state: dict[str, Any] | None = None
        self.design_progress: dict[str, Any] = {}

    def prepare(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_state: Mapping[str, Any] | None = None,
        checkpoint: Callable[[dict[str, Any]], None] | None = None,
        detail_section_applicability_resolver: DetailSectionApplicabilityResolver | None = None,
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
            planning_state = _host_operation(
                "prepare_planning_state",
                lambda: prepare_planning_state(
                    self.router,
                    prompt,
                    existing_state=existing_state,
                    checkpoint=save_state,
                    detail_section_applicability_resolver=(
                        detail_section_applicability_resolver
                    ),
                ),
            )
        except Exception as exc:
            raise PlanningStageError(
                PlanningStage.RESEARCH,
                "prompt-first research state did not reach code-ready coverage",
                cause=exc,
            ) from exc

        try:
            game_design, base_proposal = _host_operation(
                "semantic_design",
                lambda: self._semantic_design(
                    prompt,
                    planning_state=planning_state,
                    media_paths=media_paths,
                ),
            )
        except PlanningStageError:
            raise
        except Exception as exc:
            raise PlanningStageError(
                PlanningStage.DESIGN,
                "researched state could not be lowered into the semantic design",
                cause=exc,
            ) from exc

        try:
            game_design, base_proposal, research_brief, platform_evidence = _host_operation(
                "bind_target_contract",
                lambda: self._bind_platform(prompt, game_design, base_proposal),
            )
        except Exception as exc:
            raise PlanningStageError(
                PlanningStage.PLATFORM,
                "canonical target contract could not be bound",
                cause=exc,
            ) from exc

        technical_evidence = _host_operation(
            "validate_technical_evidence",
            lambda: self._validated_evidence(platform_evidence),
        )
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

        request_catalog = _host_operation(
            "build_authoritative_request_catalog",
            lambda: build_authoritative_request_catalog(
                prompt,
                self.router,
                planning_state=planning_state,
            ),
        )
        with authoritative_request_scope(
            prompt,
            request_catalog,
            planning_state=planning_state,
        ):
            from .atomic_design_pipeline import compile_atomic_design

            atomic_design = _host_operation(
                "compile_atomic_design_slots",
                lambda: compile_atomic_design(
                    prompt,
                    self.router,
                    research={
                        "planning_state_sha256": planning_state.get("state_sha256"),
                        "goal": planning_state.get("goal"),
                        "known": planning_state.get("known", []),
                        "references": planning_state.get("references", []),
                        "evidence": planning_state.get("evidence", []),
                    },
                    request_catalog=request_catalog,
                    progress=self.design_progress,
                ),
            )
            design = dict(atomic_design)
            design = _host_operation(
                "validate_ready_design",
                lambda: host_design.validate_ready_design(
                    prompt,
                    host_design.canonical_game_design(design),
                ),
            )
            design = _host_operation(
                "bind_existing_project",
                lambda: self._bind_existing_project(design),
            )
            design = {
                **design,
                "_evidence_request_catalog": request_catalog,
                "_planning_state": dict(planning_state),
                "_design_slots": atomic_design.get("_design_slots", {}),
                "_atomic_facts": atomic_design.get("_implementation_facts", []),
                "_atomic_modules": atomic_design.get("modules", []),
                "_atomic_assets": atomic_design.get("assets", []),
                "_content_entities": atomic_design.get("_content_entities", []),
                "_content_relations": atomic_design.get("_content_relations", []),
                "_research_facts": atomic_design.get("_research_facts", []),
            }
            try:
                pre_retrieval_plan = _host_operation(
                    "compile_pre_retrieval_plan",
                    lambda: compile_pre_retrieval_plan(prompt, design),
                )
            except Exception as exc:
                raise PlanningStageError(
                    PlanningStage.PRE_RETRIEVAL_PLAN,
                    "semantic design could not be lowered into the pre-retrieval plan",
                    cause=exc,
                ) from exc
            design = {**design, "_pre_retrieval_plan": pre_retrieval_plan}
            research_brief = _host_operation(
                "normalize_research_brief",
                lambda: central_research.normalize_research_brief(prompt, design),
            )
            design = {
                **design,
                "_research_brief": research_brief,
                "_planning_authority": {
                    "owner": "atomic_design_slot_pipeline",
                    "planning_state_sha256": planning_state.get("state_sha256", ""),
                    "request_catalog_sha256": request_catalog.get("catalog_sha256", ""),
                    "raw_prompt_compiler": False,
                    "model_generated_json": False,
                },
            }

        build_slice = _host_operation(
            "deterministic_bootstrap",
            lambda: host_design.deterministic_bootstrap(prompt, design),
        )
        if design.get("_content_entities"):
            # All content is owned by the graph; bootstrap must not invent another item/block.
            build_slice["contents"] = []
            build_slice["deferred_capabilities"] = []
        proposal = _host_operation(
            "lower_model_data_to_proposal",
            lambda: _proposal_from_model_data(prompt, build_slice),
        )
        if design.get("_content_entities"):
            proposal = replace(proposal, spec=replace(proposal.spec, boss=None),
                               deferred_requests=(), approval_hash="").with_hash()
            from .complete_spec import AssetRequest
            assets = []
            for asset in design.get("_atomic_assets", ()):
                if not isinstance(asset, AssetRequest):
                    raise PlanningStageError(PlanningStage.DESIGN, "atomic asset must be typed")
                assets.append(asset)
            design = {**design, "assets": assets, "_atomic_assets": assets}
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
        status = str(payload.get("status") or "").strip().casefold()
        if status in {"unavailable", "error", "failed"}:
            raise PlanningStageError(
                PlanningStage.EVIDENCE,
                f"technical evidence is explicitly unavailable ({status})",
            )
        if not str(payload.get("schema_version") or "").strip():
            payload["schema_version"] = "mmm/technical-evidence-host-normalized-v1"
        return payload


__all__ = ["PlanningArtifacts", "PlanningPipeline", "PlanningStage", "PlanningStageError"]
