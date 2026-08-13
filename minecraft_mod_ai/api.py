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
        profile: str = "t4_local",
        model_id: str | None = None,
    ) -> "ModAISession":
        if model_id is not None:
            raise SpecValidationError(
                "Direct model_id overrides are disabled. Add the model to "
                "config/model_registry.yaml and select its profile."
            )
        return cls(
            output_root=output_root,
            minecraft_version=minecraft_version,
            existing_input=existing_input,
            model_profile=profile,
        )

    @classmethod
    def with_openai_compatible_api(
        cls,
        *,
        base_url: str,
        model: str,
        api_key: str,
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        existing_input: str | Path | None = None,
        timeout_seconds: int = 90,
    ) -> "ModAISession":
        return cls(
            output_root=output_root,
            minecraft_version=minecraft_version,
            existing_input=existing_input,
            planner=OpenAICompatiblePlanner(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            ),
        )

    @classmethod
    def with_openai_compatible_env(
        cls,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "MMM_API_KEY",
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        existing_input: str | Path | None = None,
        timeout_seconds: int = 90,
    ) -> "ModAISession":
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise SpecValidationError(
                f"환경 변수 {api_key_env}에 외부 AI API 키가 없습니다."
            )
        return cls.with_openai_compatible_api(
            base_url=base_url,
            model=model,
            api_key=api_key,
            output_root=output_root,
            minecraft_version=minecraft_version,
            existing_input=existing_input,
            timeout_seconds=timeout_seconds,
        )

    def plan(self, prompt: str) -> ChatReply:
        self.reset()
        return self.chat(prompt)

    def revise(self, message: str) -> ChatReply:
        return self.chat(message)

    def chat(self, message: str) -> ChatReply:
        message = message.strip()
        if not message:
            raise SpecValidationError("대화 내용을 입력해 주세요.")
        if _is_approval_message(message) and self.proposal is not None:
            return self._reply(self.proposal)
        updated_brief = _merge_brief(self.brief, message)
        proposal = self.pipeline.plan(
            updated_brief,
            existing_input=self.existing_input,
        )
        self.brief = updated_brief
        self.proposal = proposal
        return self._reply(proposal)

    def build(
        self,
        candidate: ChatReply | Proposal | None = None,
        *,
        source_only: bool = False,
        output_root: str | Path | None = None,
    ) -> PipelineResult:
        if isinstance(candidate, ChatReply):
            proposal = candidate.proposal
        elif isinstance(candidate, Proposal):
            proposal = candidate
        elif candidate is None:
            proposal = self.proposal
        else:
            raise TypeError(
                "candidate must be ChatReply, Proposal, or None."
            )
        if proposal is None:
            raise SpecValidationError(
                "먼저 대화로 계획을 만들어 주세요."
            )
        questions = _clarification_questions(
            proposal.requested_prompt,
            proposal,
        )
        if not _buildable(proposal, questions):
            raise SpecValidationError(
                "아직 정하지 않았거나 구현과 연결되지 않은 내용이 있습니다. "
                "대화에서 필요한 내용을 더 정해 주세요."
            )
        base_output = (
            Path(output_root)
            if output_root is not None
            else self.output_root
        )
        return self.pipeline.execute(
            proposal,
            approval_hash=proposal.approval_hash,
            output_root=_new_execution_root(base_output),
            build=not source_only,
            run_gametest=not source_only,
            existing_input=self.existing_input,
        )

    def reset(self) -> None:
        self.brief = ""
        self.proposal = None

    @staticmethod
    def _reply(proposal: Proposal) -> ChatReply:
        questions = _clarification_questions(
            proposal.requested_prompt,
            proposal,
        )
        return ChatReply(
            message=_render_plan(proposal, questions),
            ready_to_build=_buildable(proposal, questions),
            questions=questions,
            proposal=proposal,
        )


@dataclass(frozen=True)
class CompleteChatReply:
    """Natural-language complete-production plan with hidden execution state."""

    message: str
    approval_hash: str = field(repr=False)
    complete_proposal: "CompleteProposal" = field(repr=False)

    @property
    def ready_to_build(self) -> bool:
        return True


class CompleteModAISession:
    """Default complete plan -> approve -> full production API."""

    def __init__(
        self,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        model_profile: str = "t4_local",
        existing_input: str | Path | None = None,
        fast_mode: bool = False,
        kv_cache_quant: str = "q4_0",
    ) -> None:
        import os
        from .complete_orchestrator import CompleteProductionOrchestrator
        from .complete_planner import CompleteGameDesignPlanner
        from .model_router import ModelRouter

        if minecraft_version != "1.20.1":
            raise SpecValidationError(
                "Complete production is pinned to Minecraft Java 1.20.1 Fabric."
            )
        self.output_root = Path(output_root)
        self.model_profile = model_profile
        self.fast_mode = fast_mode
        self.kv_cache_quant = kv_cache_quant
        os.environ["MMM_KV_CACHE_QUANT"] = kv_cache_quant
        self.existing_input = (
            Path(existing_input) if existing_input is not None else None
        )
        self.router = ModelRouter(profile=model_profile)
        if fast_mode:
            print("⚡ [Fast Mode Activated] 선택한 모델로 초소형 간이 제작/검토 모드를 실행합니다 (1~2분 완주).", flush=True)
            for role_name in ("planner", "coder", "researcher", "coder_safe", "visual_critic"):
                try:
                    cfg = self.router.registry.role(model_profile, role_name)
                    if hasattr(cfg, "max_context"):
                        cfg.max_context = min(cfg.max_context, 8192)
                    if hasattr(cfg, "max_new_tokens"):
                        cfg.max_new_tokens = min(cfg.max_new_tokens, 1024)
                except Exception:
                    pass

        self.planner = CompleteGameDesignPlanner(self.router)
        self.orchestrator = CompleteProductionOrchestrator(
            workspace_root=self.output_root,
            profile=model_profile,
            router_factory=lambda: self.router,
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
        self.reset()
        return self.chat(prompt, media_paths=media_paths)

    def revise(
        self,
        message: str,
        *,
        media_paths: tuple[str | Path, ...] = (),
    ) -> CompleteChatReply:
        return self.chat(message, media_paths=media_paths)

    def chat(
        self,
        message: str,
        *,
        media_paths: tuple[str | Path, ...] = (),
    ) -> CompleteChatReply:
        import hashlib

        try:
            updated_brief = merge_design_brief(self.brief, message)
        except ValueError as exc:
            raise SpecValidationError("대화 내용을 입력해 주세요.") from exc
        existing_hash = ""
        if self.existing_input is not None:
            if not self.existing_input.is_file():
                raise FileNotFoundError(self.existing_input)
            existing_hash = "sha256:" + hashlib.sha256(
                self.existing_input.read_bytes()
            ).hexdigest()
        proposal = self.planner.plan(
            updated_brief,
            media_paths=media_paths,
            existing_input_sha256=existing_hash,
        )
        self.brief = updated_brief
        self.complete_proposal = proposal
        self.save_plan()
        from .plan_render import render_complete_plan

        return CompleteChatReply(
            message=render_complete_plan(
                requested_prompt=proposal.requested_prompt,
                game_design=proposal.game_design,
                modules=proposal.modules,
                acceptance_tests=proposal.acceptance_tests,
            ),
            approval_hash=proposal.calculate_hash(),
            complete_proposal=proposal,
        )

    def save_plan(self, target_path: str | Path | None = None) -> Path:

        if self.complete_proposal is None:
            raise SpecValidationError("No complete proposal to save.")
        path = (
            Path(target_path)
            if target_path is not None
            else (self.output_root / "proposal.json")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.complete_proposal.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"💾 [Session] Plan saved to: {path}", flush=True)
        return path

    def load_plan(self, source_path: str | Path | None = None) -> CompleteChatReply:
        from .complete_spec import CompleteProposal
        from .plan_render import render_complete_plan

        path = (
            Path(source_path)
            if source_path is not None
            else (self.output_root / "proposal.json")
        )
        if not path.is_file():
            raise FileNotFoundError(f"No saved proposal JSON found at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        proposal = CompleteProposal.from_dict(data)
        self.complete_proposal = proposal
        self.brief = proposal.requested_prompt
        print(f"📂 [Session] Existing plan successfully loaded from: {path}", flush=True)

        return CompleteChatReply(
            message=render_complete_plan(
                requested_prompt=proposal.requested_prompt,
                game_design=proposal.game_design,
                modules=proposal.modules,
                acceptance_tests=proposal.acceptance_tests,
            ),
            approval_hash=proposal.calculate_hash(),
            complete_proposal=proposal,
        )

    def reset(self) -> None:
        self.brief = ""
        self.complete_proposal = None

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

        if isinstance(candidate, CompleteChatReply):
            proposal = candidate.complete_proposal
        elif isinstance(candidate, CompleteProposal):
            proposal = candidate
        elif candidate is None:
            proposal = self.complete_proposal
        else:
            raise TypeError(
                "candidate must be CompleteChatReply, CompleteProposal or None."
            )
        if proposal is None:
            raise SpecValidationError(
                "Create a complete plan before building."
            )
        selected = options or CompleteExecutionOptions(
            source_only=source_only
        )
        if source_only and not selected.source_only:
            selected = CompleteExecutionOptions(
                **{
                    **selected.__dict__,
                    "source_only": True,
                }
            )
        return self.orchestrator.execute(
            proposal,
            approval_hash=proposal.calculate_hash(),
            run_name=run_name,
            options=selected,
            existing_input=self.existing_input,
        )


__all__ = [
    "ChatReply",
    "CompleteChatReply",
    "CompleteModAISession",
    "ModAISession",
    "SUPPORTED_MINECRAFT_VERSIONS",
    "supported_minecraft_versions",
]
