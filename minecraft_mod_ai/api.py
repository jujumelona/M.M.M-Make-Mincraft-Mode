"""Public conversational API for notebooks, Python programs, and services."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

SUPPORTED_MINECRAFT_VERSIONS: tuple[str, ...] = ()


def supported_minecraft_versions(*, loader: str | None = None) -> tuple[str, ...]:
    from .platform_catalog import supported_minecraft_versions as discover

    return discover(loader=loader)


def _normalize_target_value(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"auto", "automatic"}:
        return None
    return text


def _validate_requested_target(
    minecraft_version: str | None,
    loader: str | None,
) -> tuple[str | None, str | None]:
    from .platform_catalog import (
        adapter_for_target,
        adapters_for_version,
        provider_for_loader,
    )

    version = _normalize_target_value(minecraft_version)
    normalized_loader = _normalize_target_value(loader)
    if normalized_loader is not None:
        normalized_loader = normalized_loader.casefold()
        try:
            provider_for_loader(normalized_loader)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc
    if version is not None and normalized_loader is not None:
        try:
            adapter_for_target(version, normalized_loader)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc
    elif version is not None and not adapters_for_version(version):
        raise SpecValidationError(
            f"Minecraft {version!r}을 실행할 수 있는 platform provider가 없습니다."
        )
    return version, normalized_loader


def _attach_target_constraints(
    owner: Any,
    *,
    minecraft_version: str | None,
    loader: str | None,
) -> None:
    if minecraft_version is not None:
        owner._mmm_requested_minecraft_version = minecraft_version
    if loader is not None:
        owner._mmm_requested_loader = loader


def _attach_existing_target(owner: Any, existing_input: Path | None) -> None:
    if existing_input is None or not existing_input.is_file():
        return
    from .importer import inspect_existing_project_archive

    report = inspect_existing_project_archive(existing_input)
    observed_archive_sha256 = str(report.archive_sha256)
    if not observed_archive_sha256.startswith("sha256:"):
        raise SpecValidationError(
            "Existing-project inspection did not return a qualified archive SHA-256."
        )
    if report.minecraft_version:
        owner._mmm_existing_minecraft_version = report.minecraft_version
    if report.loader:
        owner._mmm_existing_loader = report.loader
    report_payload = report.to_dict()
    report_payload["source"] = str(existing_input)
    if report_payload.get("archive_sha256") != observed_archive_sha256:
        raise SpecValidationError(
            "Existing-project report is not bound to its observed archive SHA-256."
        )
    owner._mmm_existing_archive_sha256 = observed_archive_sha256
    owner._mmm_existing_platform_report = report_payload
    owner._mmm_existing_project_report = report_payload

    # Build the deeper symbol/resource/test/dependency inventory while the
    # independent semantic-design call is running.  The target-selection wrapper
    # joins this future before it can decide reuse or platform coordinates.
    future: Future[Any] = Future()

    def inspect_inventory() -> None:
        try:
            from .project_inventory import inspect_existing_archive_inventory

            inventory = inspect_existing_archive_inventory(existing_input)
            inventory.validate()
            if inventory.source_sha256 != observed_archive_sha256:
                raise SpecValidationError(
                    "Existing-project ZIP changed between platform inspection and "
                    "background project inventory."
                )
            future.set_result(inventory)
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(
        target=inspect_inventory,
        name="mmm-existing-project-inventory",
        daemon=True,
    ).start()
    owner._mmm_existing_project_inventory_future = future


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verified_existing_input_sha256(
    owner: Any,
    existing_input: Path,
    *,
    await_inventory: bool = False,
) -> str:
    """Return the session's one observed archive digest or fail on any drift.

    The importer report owns the observation. Planning, the asynchronous deep
    inventory, and the proposal all bind to that exact digest; none is allowed to
    silently replace it with a later observation of the path.
    """

    path = existing_input.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    expected = str(getattr(owner, "_mmm_existing_archive_sha256", ""))
    if not expected.startswith("sha256:") or len(expected) != 71:
        raise SpecValidationError(
            "Existing-project session has no immutable observed archive SHA-256. "
            "Create a new session for this ZIP."
        )
    report = getattr(owner, "_mmm_existing_project_report", None)
    if not isinstance(report, dict) or report.get("archive_sha256") != expected:
        raise SpecValidationError(
            "Existing-project report disagrees with the session archive SHA-256."
        )
    if _sha256_file(path) != expected:
        raise SpecValidationError(
            "Existing-project ZIP changed after the session observed it. "
            "Create a new session for the changed ZIP."
        )

    future = getattr(owner, "_mmm_existing_project_inventory_future", None)
    if isinstance(future, Future) and (await_inventory or future.done()):
        try:
            inventory = future.result()
        except BaseException as exc:
            raise SpecValidationError(
                f"Existing-project inventory could not be bound to the observed ZIP: {exc}"
            ) from exc
        if str(getattr(inventory, "source_sha256", "")) != expected:
            raise SpecValidationError(
                "Existing-project inventory disagrees with the session archive SHA-256."
            )
    return expected


def _is_colab_drive_path(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    drive_root = Path("/content/drive").resolve()
    return resolved == drive_root or drive_root in resolved.parents


def _complete_workspace_root(output_root: Path) -> Path:
    configured = os.environ.get("MMM_COMPLETE_WORKSPACE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    resolved_output = output_root.expanduser().resolve()
    if _is_colab_drive_path(resolved_output):
        return Path("/content/mmm-work").resolve()
    return resolved_output


def _configure_persistent_autotune_cache(output_root: Path, model_profile: str) -> None:
    if os.environ.get("MMM_LLAMA_AUTOTUNE_CACHE", "").strip():
        return
    resolved_output = output_root.expanduser().resolve()
    if not _is_colab_drive_path(resolved_output):
        return
    profile_key = hashlib.sha256(model_profile.encode("utf-8")).hexdigest()[:16]
    cache = resolved_output / ".cache" / "llama-autotune" / f"{profile_key}.json"
    os.environ["MMM_LLAMA_AUTOTUNE_CACHE"] = str(cache)


def _atomic_copy(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{hashlib.sha256(str(source).encode()).hexdigest()[:8]}"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


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
        minecraft_version: str | None = None,
        loader: str | None = None,
        planner: Planner | None = None,
        model_profile: str = "t4_local",
        existing_input: str | Path | None = None,
    ) -> None:
        version, loader_id = _validate_requested_target(minecraft_version, loader)
        self.minecraft_version = version
        self.loader = loader_id
        self.output_root = Path(output_root)
        self.existing_input = Path(existing_input) if existing_input is not None else None
        selected_planner = planner or RoutedPlanner(profile=model_profile)
        _attach_target_constraints(
            selected_planner,
            minecraft_version=version,
            loader=loader_id,
        )
        _attach_existing_target(selected_planner, self.existing_input)
        self.pipeline = ScalableMinecraftModPipeline(planner=selected_planner)
        self.brief = ""
        self.proposal: Proposal | None = None

    @classmethod
    def with_local_model(
        cls,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str | None = None,
        loader: str | None = None,
        existing_input: str | Path | None = None,
        profile: str = "t4_local",
        model_id: str | None = None,
    ) -> ModAISession:
        if model_id is not None:
            raise SpecValidationError(
                "Direct model_id overrides are disabled. Add the model to "
                "config/model_registry.yaml and select its profile."
            )
        return cls(
            output_root=output_root,
            minecraft_version=minecraft_version,
            loader=loader,
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
        minecraft_version: str | None = None,
        loader: str | None = None,
        existing_input: str | Path | None = None,
    ) -> ModAISession:
        return cls(
            output_root=output_root,
            minecraft_version=minecraft_version,
            loader=loader,
            existing_input=existing_input,
            planner=OpenAICompatiblePlanner(
                base_url=base_url,
                model=model,
                api_key=api_key,
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
        minecraft_version: str | None = None,
        loader: str | None = None,
        existing_input: str | Path | None = None,
    ) -> ModAISession:
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
            loader=loader,
            existing_input=existing_input,
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


@dataclass(frozen=True)
class CompleteChatReply:
    """Natural-language complete-production plan with hidden execution state."""

    message: str
    approval_hash: str = field(repr=False)
    complete_proposal: CompleteProposal = field(repr=False)

    @property
    def ready_to_build(self) -> bool:
        return True


class CompleteModAISession:
    """Default complete plan -> approve -> full production API."""

    def __init__(
        self,
        *,
        output_root: str | Path = "mmm-output",
        minecraft_version: str | None = None,
        loader: str | None = None,
        model_profile: str = "t4_local",
        existing_input: str | Path | None = None,
        fast_mode: bool = False,
        kv_cache_quant: str = "q4_0",
    ) -> None:
        from .complete_orchestrator import CompleteProductionOrchestrator
        from .complete_planner import CompleteGameDesignPlanner
        from .model_router import ModelRouter

        version, loader_id = _validate_requested_target(minecraft_version, loader)
        self.minecraft_version = version
        self.loader = loader_id
        self.output_root = Path(output_root)
        self.workspace_root = _complete_workspace_root(self.output_root)
        self.model_profile = model_profile
        _configure_persistent_autotune_cache(self.output_root, model_profile)
        self.fast_mode = fast_mode
        self.kv_cache_quant = kv_cache_quant
        os.environ["MMM_KV_CACHE_QUANT"] = kv_cache_quant
        self.existing_input = Path(existing_input) if existing_input is not None else None
        self.router = ModelRouter(profile=model_profile)
        _attach_target_constraints(
            self.router,
            minecraft_version=version,
            loader=loader_id,
        )
        _attach_existing_target(self.router, self.existing_input)
        if fast_mode:
            print(
                "⚡ [Fast Mode Activated] 검색·작업 스케줄링만 빠르게 조정하며 "
                "모델 컨텍스트와 출력 한도는 줄이지 않습니다.",
                flush=True,
            )
        self.planner = CompleteGameDesignPlanner(self.router)
        self.orchestrator = CompleteProductionOrchestrator(
            workspace_root=self.workspace_root,
            profile=model_profile,
            router_factory=lambda: self.router,
        )
        self.orchestrator._fast_mode = fast_mode
        self.brief = ""
        self.complete_proposal: CompleteProposal | None = None

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
        try:
            updated_brief = merge_design_brief(self.brief, message)
        except ValueError as exc:
            raise SpecValidationError("대화 내용을 입력해 주세요.") from exc
        existing_hash = ""
        if self.existing_input is not None:
            existing_hash = _verified_existing_input_sha256(
                self.router,
                self.existing_input,
            )

        proposal = self.planner.plan(
            updated_brief,
            media_paths=media_paths,
            existing_input_sha256=existing_hash,
        )
        if self.existing_input is not None:
            rebound = _verified_existing_input_sha256(
                self.router,
                self.existing_input,
                await_inventory=True,
            )
            if rebound != existing_hash or proposal.existing_input_sha256 != existing_hash:
                raise SpecValidationError(
                    "Complete proposal is not bound to the session's observed existing-project ZIP."
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
            else self.output_root / "proposal.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.complete_proposal.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def load_plan(self, source_path: str | Path | None = None) -> CompleteChatReply:
        from .complete_spec import CompleteProposal
        from .plan_render import render_complete_plan

        path = (
            Path(source_path)
            if source_path is not None
            else self.output_root / "proposal.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"No saved proposal JSON found at {path}")
        proposal = CompleteProposal.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        self.complete_proposal = proposal
        self.brief = proposal.requested_prompt
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

    def _persist_result_artifacts(
        self,
        result: CompletePipelineResult,
        *,
        run_name: str,
    ) -> CompletePipelineResult:
        if self.workspace_root == self.output_root.expanduser().resolve():
            return result
        run_label = Path(run_name).name
        if not run_label or run_label in {".", ".."}:
            run_label = "complete-run"
        persistent_root = self.output_root.expanduser().resolve() / "runs" / run_label
        updates: dict[str, str | None] = {}
        for field_name in ("release_zip", "jar_path", "work_ledger_path"):
            raw = getattr(result, field_name, None)
            if not raw:
                continue
            source = Path(str(raw)).expanduser().resolve()
            if not source.is_file():
                continue
            destination = persistent_root / source.name
            _atomic_copy(source, destination)
            updates[field_name] = str(destination)
        return replace(result, **updates) if updates else result

    def build(
        self,
        candidate: CompleteChatReply | CompleteProposal | None = None,
        *,
        run_name: str = "complete-run",
        source_only: bool = False,
        options: CompleteExecutionOptions | None = None,
    ) -> CompletePipelineResult:
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
            raise SpecValidationError("Create a complete plan before building.")
        selected = options or CompleteExecutionOptions(source_only=source_only)
        if source_only and not selected.source_only:
            selected = CompleteExecutionOptions(
                **{**selected.__dict__, "source_only": True}
            )
        result = self.orchestrator.execute(
            proposal,
            approval_hash=proposal.calculate_hash(),
            run_name=run_name,
            options=selected,
            existing_input=self.existing_input,
        )
        return self._persist_result_artifacts(result, run_name=run_name)


__all__ = [
    "SUPPORTED_MINECRAFT_VERSIONS",
    "ChatReply",
    "CompleteChatReply",
    "CompleteModAISession",
    "ModAISession",
    "supported_minecraft_versions",
]
