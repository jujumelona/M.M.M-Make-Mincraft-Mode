"""Public conversational API for notebooks, Python programs, and services."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import MinecraftModPipeline, PipelineResult
from .planner import (
    HeuristicPlanner,
    LocalTransformersPlanner,
    OpenAICompatiblePlanner,
    Planner,
)
from .spec import Proposal, SpecValidationError
from .webui import (
    _buildable,
    _clarification_questions,
    _is_approval_message,
    _merge_brief,
    _new_execution_root,
    _render_plan,
)


SUPPORTED_MINECRAFT_VERSIONS = ("1.20.1",)


def supported_minecraft_versions() -> tuple[str, ...]:
    """Return the exact Minecraft versions backed by a validated build profile."""

    return SUPPORTED_MINECRAFT_VERSIONS


@dataclass(frozen=True)
class ChatReply:
    """One natural-language planning turn.

    ``proposal`` is available for programmatic inspection, but callers never
    need to display or manually approve its internal JSON/hash.
    """

    message: str
    ready_to_build: bool
    questions: tuple[str, ...]
    proposal: Proposal = field(repr=False)

    @property
    def buildable(self) -> bool:
        """Compatibility alias for code that prefers the shorter name."""

        return self.ready_to_build


class ModAISession:
    """Stateful plan -> revise -> build interface.

    The notebook UI is only one client of the same package. This class can be
    imported from any Colab, Python process, or service without importing
    Gradio at module import time.
    """

    def __init__(
        self,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        planner: Planner | None = None,
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
        self.pipeline = MinecraftModPipeline(planner=planner or HeuristicPlanner())
        self.brief = ""
        self.proposal: Proposal | None = None

    @classmethod
    def with_local_model(
        cls,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str = "1.20.1",
        existing_input: str | Path | None = None,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    ) -> "ModAISession":
        return cls(
            output_root=output_root,
            minecraft_version=minecraft_version,
            existing_input=existing_input,
            planner=LocalTransformersPlanner(model_id=model_id),
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
        """Create a session backed by an explicitly configured HTTPS API."""

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
        """Read an API key at runtime without putting it in source code."""

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
        """Start a new plan from a natural-language request."""

        self.reset()
        return self.chat(prompt)

    def revise(self, message: str) -> ChatReply:
        """Add a natural-language correction or missing requirement."""

        return self.chat(message)

    def chat(self, message: str) -> ChatReply:
        """Continue the requirements conversation without writing files."""

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
        """Build the current reviewed plan in a collision-free run directory."""

        if isinstance(candidate, ChatReply):
            proposal = candidate.proposal
        elif isinstance(candidate, Proposal):
            proposal = candidate
        elif candidate is None:
            proposal = self.proposal
        else:
            raise TypeError("candidate must be ChatReply, Proposal, or None.")

        if proposal is None:
            raise SpecValidationError("먼저 대화로 계획을 만들어 주세요.")
        questions = _clarification_questions(proposal.requested_prompt, proposal)
        if not _buildable(proposal, questions):
            raise SpecValidationError(
                "아직 정하지 않았거나 구현과 연결되지 않은 내용이 있습니다. "
                "대화에서 필요한 내용을 더 정해 주세요."
            )

        base_output = Path(output_root) if output_root is not None else self.output_root
        return self.pipeline.execute(
            proposal,
            approval_hash=proposal.approval_hash,
            output_root=_new_execution_root(base_output),
            build=not source_only,
            run_gametest=not source_only,
            existing_input=self.existing_input,
        )

    def reset(self) -> None:
        """Clear only the conversation state; generated runs remain untouched."""

        self.brief = ""
        self.proposal = None

    @staticmethod
    def _reply(proposal: Proposal) -> ChatReply:
        questions = _clarification_questions(proposal.requested_prompt, proposal)
        return ChatReply(
            message=_render_plan(proposal, questions),
            ready_to_build=_buildable(proposal, questions),
            questions=questions,
            proposal=proposal,
        )


__all__ = [
    "ChatReply",
    "ModAISession",
    "SUPPORTED_MINECRAFT_VERSIONS",
    "supported_minecraft_versions",
]
