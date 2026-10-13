"""Public conversational API for notebooks, Python programs, and services."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .authored_plan import AuthoredPlan
from .conversation import merge_design_brief
from .platform_backend_contract import deterministic_backend_capabilities
from .model_concurrency import planning_work_unit_timeout_seconds
from .spec import SpecValidationError

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
    if minecraft_version is not None and loader is not None:
        try:
            from .platform_catalog import adapter_for_target

            adapter = adapter_for_target(minecraft_version, loader)
            owner._mmm_target_adapter = adapter
            owner._mmm_deterministic_module_kinds = deterministic_backend_capabilities(adapter)
        except Exception:
            pass


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
            inventory = future.result(timeout=planning_work_unit_timeout_seconds())
        except TimeoutError as exc:
            future.cancel()
            raise SpecValidationError(
                "Existing-project inventory exceeded the planning work-unit deadline."
            ) from exc
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



def _saved_proposal_from_data(data: dict[str, Any]) -> CompleteProposal | AuthoredPlan:
    schema_version = data.get("schema_version")
    if schema_version == "mmm/authored-plan-v2":
        return AuthoredPlan.from_dict(data)
    if schema_version == "mmm/authored-plan-v1":
        raise ValueError(
            "AUTHORED_PLAN_V1_REMOVED: production requires the canonical typed "
            "mmm/authored-plan-v2 handoff."
        )
    from .complete_spec import CompleteProposal

    return CompleteProposal.from_dict(data)


def _proposal_message(proposal: CompleteProposal | AuthoredPlan) -> str:
    if isinstance(proposal, AuthoredPlan):
        return proposal.text
    from .plan_render import render_complete_plan

    return render_complete_plan(
        requested_prompt=proposal.requested_prompt,
        game_design=proposal.game_design,
        modules=proposal.modules,
        acceptance_tests=proposal.acceptance_tests,
        quality_contract=proposal.quality_contract,
    )


def _loaded_proposal_message(proposal: CompleteProposal | AuthoredPlan) -> str:
    if isinstance(proposal, AuthoredPlan):
        return proposal.text
    from .plan_render import render_complete_plan
    return render_complete_plan(
        requested_prompt=proposal.requested_prompt,
        game_design=proposal.game_design,
        modules=proposal.modules,
        acceptance_tests=proposal.acceptance_tests,
    )


def _validate_internal_engine_preflight() -> None:
    from .internal_package_preflight import (
        InternalPackagePreflightError,
        validate_internal_package_integrity,
    )

    try:
        validate_internal_package_integrity()
        # Validate the packaged prompts before the first local-model request.
        # Previously a YAML rule parsed as a dict failed only after many calls.
        from .task_template_catalog import RUNTIME_TEMPLATE_ROOT
        from .template_contract_validation import validate_catalog_rule_types

        validate_catalog_rule_types(RUNTIME_TEMPLATE_ROOT)
    except (InternalPackagePreflightError, ValueError) as exc:
        raise SpecValidationError(
            f"MMM internal package preflight failed before execution: {exc}"
        ) from exc


def _bound_platform_for_preflight(
    session: "CompleteModAISession",
    proposal: Any,
) -> Any | None:
    base = getattr(proposal, "base_proposal", None)
    spec = getattr(base, "spec", None)
    platform = getattr(spec, "platform", None)
    if platform is not None:
        return platform

    adapter = getattr(session.router, "_mmm_target_adapter", None)
    if adapter is not None:
        return adapter

    version = getattr(session.router, "_mmm_existing_minecraft_version", None)
    loader = getattr(session.router, "_mmm_existing_loader", None)
    if version and loader:
        from .platform_catalog import adapter_for_target

        return adapter_for_target(str(version), str(loader))
    return None


def _production_proposal(
    session: "CompleteModAISession",
    proposal: CompleteProposal | AuthoredPlan,
) -> CompleteProposal:
    if isinstance(proposal, AuthoredPlan):
        existing_hash = ""
        if session.existing_input is not None:
            existing_hash = _verified_existing_input_sha256(
                session.router, session.existing_input, await_inventory=True,
            )
        compiled = session.planner.compile_for_production(
            proposal,
            media_paths=proposal.media_paths,
            existing_input_sha256=existing_hash,
        )
    else:
        compiled = proposal

    return compiled


@contextmanager
def _production_target_scope(session: "CompleteModAISession"):
    """Keep inferred production bindings local to one build, including failures."""
    missing = object()
    snapshots = []
    for owner, names in (
        (getattr(session, "router", None), (
            "_mmm_requested_minecraft_version", "_mmm_requested_loader",
            "_mmm_target_adapter", "_mmm_deterministic_module_kinds",
            "_mmm_production_platform_selection",
        )),
        (getattr(session, "planner", None), ("adapter", "deterministic_module_kinds")),
    ):
        if owner is None:
            continue
        snapshots.extend((owner, name, getattr(owner, name, missing)) for name in names)
    try:
        yield
    finally:
        for owner, name, value in snapshots:
            if value is missing:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, value)


def _prepare_production_scaffold(
    session: "CompleteModAISession", proposal: Any, platform: Any | None,
) -> Any | None:
    """Create the real fresh-project scaffold before any production model work."""
    if session.existing_input is not None or getattr(proposal, "existing_input_sha256", ""):
        return platform
    from .platform_catalog import adapter_for_lock_values
    from .target_contract import TargetContract
    from .platform_live_execution_contract import _uses_official_scaffold, prepare_official_fabric_project
    from .complete_orchestrator_support import CompleteProductionError

    if isinstance(proposal, AuthoredPlan):
        from .authored_production import authored_base_proposal

        base = authored_base_proposal(proposal)
        if platform is None:
            from .host_target_binding import bind_platform
            from .typed_platform_ir import PLATFORM_HOST_KINDS

            _design, base = bind_platform(
                session.router, proposal.requested_prompt,
                {"authored_plan": proposal.to_dict()}, base,
                module_kinds=tuple(
                    str(row["kind"]) for row in proposal.typed_plan_ir.get("platform_modules", ())
                    if isinstance(row, dict) and row.get("kind")
                    and row["kind"] not in PLATFORM_HOST_KINDS
                ),
            )
            platform = base.spec.platform
        adapter = platform if isinstance(platform, TargetContract) else adapter_for_lock_values(platform)
        from .platform_resolver import PlatformSelection, retarget_proposal

        selection = PlatformSelection(
            adapter=adapter, source="production_scaffold",
            reason="Prepare the selected target before production model work.",
            explicit_version=True, explicit_loader=True,
        )
        base = retarget_proposal(base, selection)
        # AUTO binds once before authoring. Later compilation must use the same
        # complete receipt for which the scaffold and model backend were prepared.
        session.router._mmm_production_platform_selection = selection
        session.router._mmm_requested_minecraft_version = adapter.minecraft_version
        session.router._mmm_requested_loader = adapter.loader
        session.router._mmm_target_adapter = adapter
        session.router._mmm_deterministic_module_kinds = deterministic_backend_capabilities(adapter)
        session.planner.adapter = adapter
        session.planner.deterministic_module_kinds = deterministic_backend_capabilities(adapter)
    else:
        if platform is None:
            return None
        base = proposal.base_proposal
        adapter = platform if isinstance(platform, TargetContract) else adapter_for_lock_values(platform)
    if not _uses_official_scaffold(adapter):
        return platform

    cache_key = hashlib.sha256(json.dumps(
        base.to_dict()["spec"], sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    from types import SimpleNamespace

    print(f"FABRIC_PROJECT_PREPARE: minecraft={adapter.minecraft_version} before production model work", flush=True)
    root = prepare_official_fabric_project(
        session.orchestrator, SimpleNamespace(base_proposal=base),
        run_root=session.workspace_root / ".cache/fabric-scaffolds" / cache_key,
        adapter=adapter, error_type=CompleteProductionError,
    )
    session.orchestrator._prepared_fabric_project = root
    print(f"FABRIC_PROJECT_READY: minecraft={adapter.minecraft_version} project={root}", flush=True)
    return base.spec.platform


def _preflight_production_construction(
    session: "CompleteModAISession", platform: Any,
) -> None:
    """Prepare the actual small-model compiler context before production planning."""
    from .fixed_template_generation import _adapter_name

    if _adapter_name(session.router, "coder") != "llama_cpp":
        return
    from .platform_catalog import adapter_for_lock_values
    from .target_contract import TargetContract, target_symbol_namespace

    adapter = platform if isinstance(platform, TargetContract) else adapter_for_lock_values(platform)
    # Existing projects still use the same target semantics; their classpath is
    # inspected after import. Fresh official scaffolds are available right now.
    # Match the execution snapshot's raw fields without admitting a host bundle
    # yet, so native mapping contradictions cannot disappear in a public view.
    adapter.validate()
    target_symbol_namespace(asdict(adapter))
    if session.existing_input is not None:
        return
    from .platform_live_execution_contract import _uses_official_scaffold

    if not _uses_official_scaffold(adapter):
        return
    from .complete_orchestrator_support import CompleteProductionError

    resolved = adapter.version_context
    target = resolved.to_dict()["target"]
    root = getattr(session.orchestrator, "_prepared_fabric_project", None)
    if root is None:
        raise CompleteProductionError("CONSTRUCTION_PREPARED_PROJECT_REQUIRED")
    print(
        f"CONSTRUCTION_TARGET_PREFLIGHT: minecraft={resolved.minecraft} before production model work",
        flush=True,
    )
    from .construction_classpath import resolve_fabric_compile_classpath
    from .construction_runtime import prepare_construction_context
    from tempfile import TemporaryDirectory

    root = Path(root).resolve()

    def ignore_runtime_outputs(directory: str, names: list[str]) -> list[str]:
        # Only root build outputs are caches; a source package named "build"
        # is part of the approved scaffold and must remain available.
        return [name for name in names if name in {".gradle", "build"}] if Path(directory) == root else []

    try:
        # Gradle inspection can write caches/reports. Keep the reusable approved
        # scaffold immutable, and never hand temporary paths to real generation.
        with TemporaryDirectory(prefix="mmm-construction-preflight-") as directory:
            probe_root = Path(directory) / "project"
            shutil.copytree(root, probe_root, ignore=ignore_runtime_outputs)
            classpath = resolve_fabric_compile_classpath(str(probe_root), java_version=resolved.java)
            context = prepare_construction_context(
                {}, java_classpath=classpath, project_root=probe_root, target_override=target,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        raise CompleteProductionError(
            "Construction target preflight failed before production model work: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print(
        f"CONSTRUCTION_TARGET_READY: minecraft={resolved.minecraft} "
        f"namespace={context['namespace']} java={resolved.java} "
        f"symbols={len(context['symbols'])} context_id={resolved.context_id}",
        flush=True,
    )


@dataclass(frozen=True)
class CompleteChatReply:
    """Natural-language complete-production plan with hidden execution state."""

    message: str
    approval_hash: str = field(repr=False)
    complete_proposal: CompleteProposal | AuthoredPlan = field(repr=False)

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
        # Retrieval must index the generated/existing mod workspace, never whichever
        # directory happened to launch the MMM engine process.
        self.router._mmm_workspace_root = str(self.workspace_root.resolve())
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
        target_adapter = getattr(self.router, "_mmm_target_adapter", None)
        self.planner = CompleteGameDesignPlanner(
            self.router,
            adapter=target_adapter,
            deterministic_module_kinds=(
                deterministic_backend_capabilities(target_adapter)
                if target_adapter is not None
                else None
            ),
        )
        self.orchestrator = CompleteProductionOrchestrator(
            workspace_root=self.workspace_root,
            profile=model_profile,
            router_factory=lambda: self.router,
        )
        self.orchestrator._fast_mode = fast_mode
        self.brief = ""
        self.complete_proposal: CompleteProposal | AuthoredPlan | None = None

    def plan(
        self,
        prompt: str,
        *,
        media_paths: tuple[str | Path, ...] = (),
        save_plan_path: str | Path | None = None,
    ) -> CompleteChatReply:
        self.reset()
        return self.chat(
            prompt, media_paths=media_paths, save_plan_path=save_plan_path,
        )

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
        save_plan_path: str | Path | None = None,
    ) -> CompleteChatReply:
        _validate_internal_engine_preflight()
        try:
            updated_brief = merge_design_brief(self.brief, message)
        except ValueError as exc:
            raise SpecValidationError("대화 내용을 입력해 주세요.") from exc
        existing_input_sha256 = ""
        if self.existing_input is not None:
            existing_input_sha256 = _verified_existing_input_sha256(
                self.router,
                self.existing_input,
                await_inventory=True,
            )
        proposal = self.planner.plan(
            updated_brief,
            media_paths=media_paths,
            existing_input_sha256=existing_input_sha256,
        )
        self.brief = updated_brief
        self.complete_proposal = proposal
        self.save_plan(save_plan_path)
        return CompleteChatReply(
            message=_loaded_proposal_message(proposal),
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
        path = (
            Path(source_path)
            if source_path is not None
            else self.output_root / "proposal.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"No saved proposal JSON found at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        proposal = _saved_proposal_from_data(data)
        self.complete_proposal = proposal
        self.brief = proposal.requested_prompt
        return CompleteChatReply(
            message=_loaded_proposal_message(proposal),
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
        candidate: CompleteChatReply | CompleteProposal | AuthoredPlan | None = None,
        *,
        run_name: str = "complete-run",
        source_only: bool = False,
        options: CompleteExecutionOptions | None = None,
    ) -> CompletePipelineResult:
        from .complete_orchestrator import CompleteExecutionOptions
        from .complete_spec import CompleteProposal

        _validate_internal_engine_preflight()

        if isinstance(candidate, CompleteChatReply):
            proposal = candidate.complete_proposal
        elif isinstance(candidate, (CompleteProposal, AuthoredPlan)):
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

        with _production_target_scope(self):
            bound_platform = _bound_platform_for_preflight(self, proposal)
            bound_platform = _prepare_production_scaffold(self, proposal, bound_platform)
            if bound_platform is not None:
                from .complete_preflight_contract import (
                    validate_platform_toolchain_preflight,
                )

                validate_platform_toolchain_preflight(bound_platform, selected)
                _preflight_production_construction(self, bound_platform)

            proposal = _production_proposal(self, proposal)
        # Colab's reusable launch options may have been created before the
        # approved plan was compiled. A plan with entity/boss/NPC geometry must
        # not silently disable host-mandatory Blockbench verification.
        required_entity_ids = tuple(
            str(module.module_id)
            for module in proposal.modules
            if module.kind in {"entity", "boss", "npc"}
        )
        if required_entity_ids and not selected.source_only and not selected.run_blockbench:
            selected = replace(selected, run_blockbench=True)
            print(
                "BLOCKBENCH_REQUIRED_AUTO_ENABLED: modules="
                + ",".join(required_entity_ids[:12]),
                flush=True,
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
    "CompleteChatReply",
    "CompleteModAISession",
    "supported_minecraft_versions",
]
