"""Public conversational API for notebooks, Python programs, and services."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .conversation import merge_design_brief
from .pipeline import PipelineResult
from .planner import OpenAICompatiblePlanner, Planner
from .routed_planner import RoutedPlanner
from .scalable_pipeline import ScalableMinecraftModPipeline
from .spec import Proposal, SpecValidationError
from .webui import (
    _buildable,
    _clarification_questions,
    _is_approval_message,
    _merge_brief,
    _new_execution_root,
    _render_plan,
)

if TYPE_CHECKING:
    from .complete_orchestrator import CompleteExecutionOptions, CompletePipelineResult
    from .complete_spec import CompleteProposal

SUPPORTED_MINECRAFT_VERSIONS = ("1.20.1",)


def supported_minecraft_versions() -> tuple[str, ...]:
    return SUPPORTED_MINECRAFT_VERSIONS


@dataclass(frozen=True)
class ChatReply:
    message: str
    ready_to_build: bool
    questions: tuple[str, ...]
    proposal: Proposal = field(repr=False)

    @property
    def buildable(self) -> bool:
        return self.ready_to_build


class ModAISession:
    """Stateful plan -> revise -> scalable build API."""

    def __init__(
        self,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        planner: Planner | None = None,
        model_profile: str = "t4_local",
        existing_input: str | Path | None = None,
    ) -> None:
        requested_version = minecraft_version.strip()
        if requested_version not in SUPPORTED_MINECRAFT_VERSIONS:
            supported = ", ".join(SUPPORTED_MINECRAFT_VERSIONS)
            raise SpecValidationError(
                f"지원하지 않는 Minecraft 버전입니다: {requested_version!r}. "
                f"현재 검증된 버전: {supported}"
            )
        self.minecraft_version = requested_version
        self.output_root = Path(output_root)
        self.existing_input = (
            Path(existing_input) if existing_input is not None else None
        )
        self.pipeline = ScalableMinecraftModPipeline(
            planner=planner or RoutedPlanner(profile=model_profile)
        )
        self.brief = ""
        self.proposal: Proposal | None = None

    @classmethod
    def with_local_model(
        cls,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        existing_input: str | Path | None = None,
        model_profile: str = "t4_local",
    ) -> "ModAISession":
        return cls(
            output_root=output_root,
            minecraft_version=minecraft_version,
            existing_input=existing_input,
            model_profile=model_profile,
        )

    @classmethod
    def with_openai_compatible_model(
        cls,
        *,
        endpoint: str,
        model: str,
        api_key: str = "",
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        existing_input: str | Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> "ModAISession":
        planner = OpenAICompatiblePlanner(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return cls(
            output_root=output_root,
            minecraft_version=minecraft_version,
            planner=planner,
            existing_input=existing_input,
        )

    def plan(self, prompt: str) -> ChatReply:
        self.brief = _merge_brief(self.brief, prompt)
        self.proposal = self.pipeline.plan(self.brief)
        return self._reply(self.proposal)

    def revise(self, feedback: str) -> ChatReply:
        if self.proposal is None:
            return self.plan(feedback)
        self.brief = merge_design_brief(self.brief, feedback)
        self.proposal = self.pipeline.plan(self.brief)
        return self._reply(self.proposal)

    def load_plan(self, path: str | Path) -> ChatReply:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.proposal = Proposal.from_dict(payload)
        self.brief = self.proposal.requested_prompt
        return self._reply(self.proposal)

    def save_plan(self, path: str | Path) -> Path:
        if self.proposal is None:
            raise SpecValidationError("저장할 플랜이 없습니다.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.proposal.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def build(
        self,
        candidate: ChatReply | Proposal | None = None,
        *,
        source_only: bool = False,
        build: bool = True,
        run_name: str = "mod-build",
    ) -> PipelineResult:
        proposal = candidate.proposal if isinstance(candidate, ChatReply) else candidate
        proposal = proposal or self.proposal
        if proposal is None:
            raise SpecValidationError("먼저 플랜을 생성해 주세요.")
        if not _buildable(proposal):
            questions = _clarification_questions(proposal)
            raise SpecValidationError(
                "플랜이 아직 제작 준비 상태가 아닙니다. " + " / ".join(questions)
            )
        execution_root = _new_execution_root(self.output_root, run_name)
        return self.pipeline.execute(
            proposal,
            approval_hash=proposal.approval_hash,
            output_root=execution_root,
            source_only=source_only,
            build=build,
            existing_input=self.existing_input,
        )

    def _reply(self, proposal: Proposal) -> ChatReply:
        questions = tuple(_clarification_questions(proposal))
        ready = _buildable(proposal)
        return ChatReply(
            message=_render_plan(proposal),
            ready_to_build=ready,
            questions=questions,
            proposal=proposal,
        )


@dataclass(frozen=True)
class CompleteChatReply:
    message: str
    approval_hash: str = field(repr=False)
    complete_proposal: "CompleteProposal" = field(repr=False)

    @property
    def buildable(self) -> bool:
        return bool(self.approval_hash)


class CompleteModAISession:
    """Stateful full-production plan -> revise -> build API."""

    def __init__(
        self,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        model_profile: str = "t4_local",
        existing_input: str | Path | None = None,
        fast_mode: bool = False,
    ) -> None:
        from .complete_orchestrator import CompleteProductionOrchestrator

        self.output_root = Path(output_root)
        self.minecraft_version = minecraft_version.strip()
        self.model_profile = model_profile
        self.existing_input = Path(existing_input) if existing_input is not None else None
        self.orchestrator = CompleteProductionOrchestrator(
            output_root=self.output_root,
            model_profile=model_profile,
        )
        self.orchestrator._fast_mode = fast_mode
        self.brief = ""
        self.complete_proposal: "CompleteProposal | None" = None

    def plan(
        self,
        prompt: str,
        *,
        media_paths: tuple[str | Path, ...] = (),
    ) -> CompleteChatReply:
        self.brief = _merge_brief(self.brief, prompt)
        self.complete_proposal = self.orchestrator.plan(
            self.brief,
            media_paths=media_paths,
            minecraft_version=self.minecraft_version,
            existing_input=self.existing_input,
        )
        return self._reply(self.complete_proposal)

    def revise(
        self,
        feedback: str,
        *,
        media_paths: tuple[str | Path, ...] = (),
    ) -> CompleteChatReply:
        if self.complete_proposal is None:
            return self.plan(feedback, media_paths=media_paths)
        self.brief = merge_design_brief(self.brief, feedback)
        self.complete_proposal = self.orchestrator.plan(
            self.brief,
            media_paths=media_paths,
            minecraft_version=self.minecraft_version,
            existing_input=self.existing_input,
        )
        return self._reply(self.complete_proposal)

    def load_plan(self, path: str | Path) -> CompleteChatReply:
        from .complete_spec import CompleteProposal

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.complete_proposal = CompleteProposal.from_dict(payload)
        self.brief = self.complete_proposal.requested_prompt
        return self._reply(self.complete_proposal)

    def save_plan(self, path: str | Path) -> Path:
        if self.complete_proposal is None:
            raise SpecValidationError("저장할 플랜이 없습니다.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.complete_proposal.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    def build(
        self,
        candidate: CompleteChatReply | "CompleteProposal | None" = None,
        *,
        run_name: str = "complete-run",
        source_only: bool = False,
        options: "CompleteExecutionOptions | None" = None,
    ) -> "CompletePipelineResult":
        from .complete_orchestrator import CompleteExecutionOptions
        from .complete_spec import CompleteProposal

        proposal = (
            candidate.complete_proposal
            if isinstance(candidate, CompleteChatReply)
            else candidate
        )
        proposal = proposal or self.complete_proposal
        if proposal is None:
            raise SpecValidationError("먼저 완전 제작 플랜을 생성해 주세요.")
        if not isinstance(proposal, CompleteProposal):
            raise SpecValidationError("잘못된 완전 제작 플랜 형식입니다.")
        if not proposal.approval_hash:
            raise SpecValidationError("승인 해시가 없는 플랜은 제작할 수 없습니다.")
        execution_root = _new_execution_root(self.output_root, run_name)
        execution_options = options or CompleteExecutionOptions(source_only=source_only)
        return self.orchestrator.execute(
            proposal,
            output_root=execution_root,
            options=execution_options,
            existing_input=self.existing_input,
        )

    @staticmethod
    def _reply(proposal: "CompleteProposal") -> CompleteChatReply:
        return CompleteChatReply(
            message=_render_plan(proposal.base_proposal),
            approval_hash=proposal.approval_hash,
            complete_proposal=proposal,
        )


__all__ = [
    "ChatReply",
    "CompleteChatReply",
    "CompleteModAISession",
    "ModAISession",
    "SUPPORTED_MINECRAFT_VERSIONS",
    "supported_minecraft_versions",
]
