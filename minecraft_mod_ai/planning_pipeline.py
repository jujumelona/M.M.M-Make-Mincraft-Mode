from __future__ import annotations

"""Canonical fail-closed planning pipeline.

The small language model performs bounded semantic synthesis only. Host code owns stage
ordering, request authority, platform resolution, evidence binding, and state validity.
No failed stage is converted into a heuristic/sentinel artifact that can advance.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

from . import central_research
from .model_router import ModelRouter
from .planner import _proposal_from_model_data
from .platform_resolver import retarget_proposal
from .spec import Proposal, SpecValidationError

_T = TypeVar("_T")


class PlanningStage(str, Enum):
    REQUEST = "request"
    DESIGN = "design"
    PRE_RETRIEVAL_PLAN = "pre_retrieval_plan"
    PLATFORM = "platform"
    EVIDENCE = "evidence"


class PlanningStageError(SpecValidationError):
    """Typed stage failure; callers must stop rather than synthesize a fallback artifact."""

    def __init__(self, stage: PlanningStage, message: str, *, cause: BaseException | None = None) -> None:
        self.stage = stage
        self.cause_type = type(cause).__name__ if cause is not None else ""
        suffix = f" ({self.cause_type}: {cause})" if cause is not None else ""
        super().__init__(f"planning stage {stage.value} failed: {message}{suffix}")


@dataclass(frozen=True)
class PlanningArtifacts:
    game_design: dict[str, Any]
    base_proposal: Proposal
    research_brief: dict[str, Any]
    technical_evidence: dict[str, Any]


def _stage(stage: PlanningStage, operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except PlanningStageError:
        raise
    except Exception as exc:
        raise PlanningStageError(stage, "stage contract was not satisfied", cause=exc) from exc


class PlanningPipeline:
    """Compiler-like planning owner used by CompleteGameDesignPlanner."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def prepare(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
    ) -> PlanningArtifacts:
        if not str(prompt).strip():
            raise PlanningStageError(PlanningStage.REQUEST, "prompt is empty")

        game_design, base_proposal = _stage(
            PlanningStage.DESIGN,
            lambda: self._semantic_design(prompt, media_paths=media_paths),
        )
        game_design, base_proposal, research_brief, platform_evidence = _stage(
            PlanningStage.PLATFORM,
            lambda: self._bind_platform(prompt, game_design, base_proposal),
        )
        technical_evidence = _stage(
            PlanningStage.EVIDENCE,
            lambda: self._validated_evidence(platform_evidence),
        )
        return PlanningArtifacts(
            game_design=game_design,
            base_proposal=base_proposal,
            research_brief=research_brief,
            technical_evidence=technical_evidence,
        )

    def _semantic_design(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path],
    ) -> tuple[dict[str, Any], Proposal]:
        from . import game_design as gd
        from .planning_authority import (
            authoritative_request_scope,
            build_authoritative_request_catalog,
        )
        from .reuse_planner import compile_pre_retrieval_plan

        request_catalog = build_authoritative_request_catalog(prompt, self.router)
        with authoritative_request_scope(prompt, request_catalog):
            page_budget = gd._request_page_bytes(self.router)
            request_pages = gd._lossless_request_pages(
                prompt,
                max_json_text_bytes=page_budget,
            )
            if not request_pages:
                raise PlanningStageError(PlanningStage.REQUEST, "request paging produced no pages")
            if len(request_pages) == 1:
                design = self._strict_generate_once(
                    authoritative_prompt=prompt,
                    design_prompt=prompt,
                    media_paths=media_paths,
                    system_prompt=gd._system_prompt(),
                )
            else:
                design = self._strict_sharded_design(
                    prompt,
                    request_pages=request_pages,
                    media_paths=media_paths,
                    page_budget=page_budget,
                )
            design = gd._validate_ready_design(prompt, design)
            design = self._bind_existing_project(design)
            design = {**design, "_evidence_request_catalog": request_catalog}

            pre_retrieval_plan = _stage(
                PlanningStage.PRE_RETRIEVAL_PLAN,
                lambda: compile_pre_retrieval_plan(prompt, design),
            )
            design = {**design, "_pre_retrieval_plan": pre_retrieval_plan}
            research_brief = central_research.normalize_research_brief(prompt, design)
            design = {**design, "_research_brief": research_brief}

        build_slice = gd._deterministic_bootstrap(prompt, design)
        proposal = _proposal_from_model_data(prompt, build_slice)
        if proposal.requested_prompt != prompt:
            proposal = replace(
                proposal,
                requested_prompt=prompt,
                approval_hash="",
            ).with_hash()
        return design, proposal

    def _strict_generate_once(
        self,
        *,
        authoritative_prompt: str,
        design_prompt: str,
        media_paths: Sequence[str | Path],
        system_prompt: str,
        precollected_research: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from . import agentic_research_game_design as agentic
        from . import game_design as gd

        if agentic.supports_agentic_research_router(self.router):
            from .pre_design_research_pipeline import collect_design_research

            research = (
                dict(precollected_research)
                if isinstance(precollected_research, Mapping)
                else collect_design_research(self.router, design_prompt)
            )
            design = agentic.generate_sectioned_game_design(
                gd,
                self.router,
                design_prompt,
                media_paths=media_paths,
                research=research,
            )
            canonical = gd._canonical_game_design(design)
            result: dict[str, Any] = {
                **canonical,
                "_pre_design_research": research,
                "_research_brief": dict(research.get("research_brief") or {}),
            }
            return gd._validate_ready_design(design_prompt, result)

        authority = gd._active_authority()
        effective_system_prompt = system_prompt
        if authority is not None:
            _authority_prompt, ledger = authority
            effective_system_prompt = gd._augment_system_prompt(system_prompt, ledger)
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": authoritative_prompt},
        ]
        raw = self.router.generate_text(
            "planner",
            messages,
            media_paths=media_paths,
            response_format="json",
            response_schema=gd._GAME_DESIGN_RESPONSE_SCHEMA,
            enable_tools=False,
        )
        candidate = gd._extract_model_design(str(raw))
        if not candidate:
            raise PlanningStageError(
                PlanningStage.DESIGN,
                "planner returned no parseable game_design object",
            )
        result = gd._merge_model_design(gd._game_design_skeleton(design_prompt), candidate)
        return gd._validate_ready_design(design_prompt, result)

    def _strict_sharded_design(
        self,
        prompt: str,
        *,
        request_pages: tuple[str, ...],
        media_paths: Sequence[str | Path],
        page_budget: int,
    ) -> dict[str, Any]:
        from . import agentic_research_game_design as agentic
        from . import game_design as gd

        page_research: tuple[Mapping[str, Any], ...] = ()
        if agentic.supports_agentic_research_router(self.router):
            from .pre_design_research_pipeline import collect_design_research

            page_count = len(request_pages)
            page_research = tuple(
                collect_design_research(
                    self.router,
                    page_text,
                    trace_metadata={
                        "request_page_index": page_index,
                        "request_page_count": page_count,
                    },
                )
                for page_index, page_text in enumerate(request_pages)
            )

        prompt_bytes = prompt.encode("utf-8")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        page_designs: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        byte_offset = 0
        char_offset = 0
        authority = gd._active_authority()

        for page_index, page_text in enumerate(request_pages):
            encoded_page = page_text.encode("utf-8")
            byte_start = byte_offset
            byte_offset += len(encoded_page)
            char_start = char_offset
            char_offset += len(page_text)
            receipt = {
                "page_index": page_index,
                "page_count": len(request_pages),
                "byte_start": byte_start,
                "byte_end": byte_offset,
                "byte_length": len(encoded_page),
                "content_sha256": hashlib.sha256(encoded_page).hexdigest(),
            }
            request = {
                "schema_version": gd._REQUEST_PAGE_SCHEMA,
                "full_request": {
                    "sha256": prompt_sha256,
                    "byte_length": len(prompt_bytes),
                    "page_count": len(request_pages),
                },
                "page": receipt,
                "authoritative_request_text": page_text,
            }
            token = gd._activate_page_authority(
                authority,
                page_text=page_text,
                char_start=char_start,
                char_end=char_offset,
            )
            try:
                design = self._strict_generate_once(
                    authoritative_prompt=json.dumps(
                        request,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    design_prompt=page_text,
                    media_paths=media_paths if page_index == 0 else (),
                    system_prompt=gd._sharded_design_system_prompt(),
                    precollected_research=(
                        page_research[page_index] if page_research else None
                    ),
                )
            finally:
                gd._reset_page_authority(token)
            page_designs.append(design)
            receipts.append({**receipt, "design_sha256": gd._json_sha256(design)})

        if byte_offset != len(prompt_bytes) or "".join(request_pages) != prompt:
            raise PlanningStageError(
                PlanningStage.REQUEST,
                "authoritative request paging failed its lossless host check",
            )

        merged = gd._merge_game_design_pages(page_designs)
        chain = hashlib.sha256(
            f"{gd._REQUEST_INGESTION_SCHEMA}:{prompt_sha256}".encode()
        )
        for receipt in receipts:
            chain.update(
                json.dumps(
                    receipt,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        ingestion = {
            "schema_version": gd._REQUEST_INGESTION_SCHEMA,
            "prompt_sha256": prompt_sha256,
            "prompt_byte_length": len(prompt_bytes),
            "page_count": len(request_pages),
            "page_json_text_max_bytes": page_budget,
            "pages": receipts,
            "chain_sha256": chain.hexdigest(),
            "authority": {
                "semantic_input": "user_request",
                "receipts": "host_computed",
                "model_output": "descriptive_values_only",
                "execution_authority": False,
            },
        }
        result: dict[str, Any] = {**merged, "_request_ingestion": ingestion}
        if page_research:
            result["_pre_design_research"] = gd._sharded_research_ledger(
                prompt,
                request_pages,
                page_research,
            )
        return result

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
            if not callable(validate):
                raise PlanningStageError(
                    PlanningStage.REQUEST,
                    "existing-project inventory did not return a validated host object",
                )
            validate()
            to_dict = getattr(inventory, "to_dict", None)
            if not callable(to_dict):
                raise PlanningStageError(
                    PlanningStage.REQUEST,
                    "existing-project inventory cannot be bound to planning",
                )
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
                "_component_catalog": dict(
                    inventory_payload.get("component_catalog") or {}
                ),
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
            raise PlanningStageError(
                PlanningStage.PLATFORM,
                "semantic design did not produce a research brief",
            )
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
        # Convert only host-grounded GitHub evidence into executable donor source.
        # This runs after the target is frozen and before PlanIR compilation so the
        # small coder receives code bytes, not another repository-search problem.
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
            if not isinstance(value, Mapping):
                raise PlanningStageError(
                    PlanningStage.PLATFORM,
                    "target research returned a non-object evidence receipt",
                )
            platform_evidence = dict(value)

        pre_design = design.get("_pre_design_research")
        if isinstance(pre_design, dict):
            deterministic = pre_design.get("deterministic")
            if isinstance(deterministic, dict):
                deterministic = {
                    **deterministic,
                    "official_rag": dict(platform_evidence),
                }
            else:
                deterministic = {"official_rag": dict(platform_evidence)}
            pre_design = {
                **pre_design,
                "research_brief": bound_brief,
                "deterministic": deterministic,
            }

        result = {
            **design,
            "_platform_selection": selection_dict,
            "_platform_evidence": dict(platform_evidence),
            "_research_brief": bound_brief,
        }
        if isinstance(pre_design, dict):
            result["_pre_design_research"] = pre_design
        return result, proposal, bound_brief, dict(platform_evidence)

    @staticmethod
    def _validated_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise PlanningStageError(
                PlanningStage.EVIDENCE,
                "technical evidence is not an object",
            )
        payload = dict(value)
        if payload.get("status") == "unavailable":
            raise PlanningStageError(
                PlanningStage.EVIDENCE,
                "technical evidence is explicitly unavailable",
            )
        schema = str(payload.get("schema_version") or "").strip()
        if not schema:
            raise PlanningStageError(
                PlanningStage.EVIDENCE,
                "technical evidence has no schema_version",
            )
        return payload


__all__ = [
    "PlanningArtifacts",
    "PlanningPipeline",
    "PlanningStage",
    "PlanningStageError",
]
