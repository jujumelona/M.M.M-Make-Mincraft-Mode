from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .audio_generator import generate_audio_assets
from .complete_orchestrator_services import (
    blockbench_review,
    generate_assets,
    package_source_only,
    run_playtest,
    runtime_profile,
    visual_review,
)
from .complete_orchestrator_support import (
    CompleteProductionError,
    _external_gates,
    _handled_module_ids,
    _jar_path,
    _locate_existing_fabric_root,
    _module_dict,
    _normalize_modules,
    _system_groups,
)
from .complete_spec import CompleteProposal, CompleteProposalStatus, ProductionModule
from .custom_module_generator import CustomModuleGenerator
from .extended_content_generator import generate_extended_content
from .generator import FabricProjectGenerator
from .geckolib_generator import generate_geckolib_entity_assets
from .importer import inspect_existing_project_archive
from .java_lsp import JavaLanguageService
from .model_router import ModelRouter
from .project_edit import inspect_fabric_project
from .project_index import ProjectIndex
from .publisher import (
    build_distribution_metadata,
    package_distribution_bundle,
    publish_curseforge,
    publish_modrinth,
)
from .repair_engine import RepairEngine
from .resource_tuning import tune_gradle_resources
from .runner import GradleRunner
from .runtime_manager import MinecraftRuntimeManager
from .scalable_validator import ScalableProjectValidator
from .scalable_world_compiler import compile_scalable_world_ir
from .scale_policy import ScalePolicy
from .spec import SpecValidationError
from .system_pack_generator import generate_system_pack
from .validator import validate_jar


@dataclass(frozen=True)
class CompleteExecutionOptions:
    source_only: bool = False
    run_jdt: bool = True
    run_gametest: bool = True
    auto_repair: bool = True
    max_repair_attempts: int | None = None
    run_blockbench: bool = True
    run_runtime: bool = True
    run_client: bool = True
    run_mineflayer: bool = True
    run_visual_review: bool = True
    cleanup_runtime: bool = True
    eula_accepted: bool = False
    server_launcher: str | None = None
    screenshot_paths: tuple[str, ...] = ()
    playtest_actions: tuple[dict[str, Any], ...] = ()
    publish_provider: str | None = None
    publish_project_id: str | None = None
    changelog: str = "Generated and verified by M.M.M"
    gradle_heap_mb: int | None = None
    server_memory_mb: int | None = None

    def validate(self, *, policy: ScalePolicy | None = None) -> None:
        policy = policy or ScalePolicy.from_environment()
        if self.max_repair_attempts is not None and (
            type(self.max_repair_attempts) is not int
            or self.max_repair_attempts < 1
        ):
            raise CompleteProductionError(
                "max_repair_attempts must be null or a positive integer."
            )
        if self.publish_provider not in {None, "modrinth", "curseforge"}:
            raise CompleteProductionError(
                "publish_provider must be modrinth or curseforge."
            )
        if self.publish_provider and not self.publish_project_id:
            raise CompleteProductionError(
                "publish_project_id is required when publishing."
            )
        if self.source_only and self.publish_provider:
            raise CompleteProductionError(
                "Source-only execution cannot publish a binary release."
            )
        if self.gradle_heap_mb is not None and self.gradle_heap_mb < 512:
            raise CompleteProductionError(
                "gradle_heap_mb must be at least 512 when supplied."
            )
        if self.server_memory_mb is not None and self.server_memory_mb < 1024:
            raise CompleteProductionError(
                "server_memory_mb must be at least 1024 when supplied."
            )
        if (
            self.gradle_heap_mb is not None
            and self.gradle_heap_mb > policy.gradle_max_heap_mb
        ):
            raise CompleteProductionError(
                "gradle_heap_mb exceeds MMM_GRADLE_MAX_HEAP_MB host policy."
            )
        if (
            self.server_memory_mb is not None
            and self.server_memory_mb > policy.runtime_max_heap_mb
        ):
            raise CompleteProductionError(
                "server_memory_mb exceeds MMM_RUNTIME_MAX_HEAP_MB host policy."
            )


@dataclass(frozen=True)
class CompletePipelineResult:
    schema_version: str
    status: str
    project_root: str
    release_zip: str | None
    jar_path: str | None
    complete_proposal_hash: str
    source_validation: dict[str, Any]
    build_report: dict[str, Any] | None
    jar_validation: dict[str, Any] | None
    module_receipts: tuple[dict[str, Any], ...]
    world_receipt: dict[str, Any] | None
    asset_receipt: dict[str, Any] | None
    audio_receipt: dict[str, Any] | None
    blockbench_receipts: tuple[dict[str, Any], ...]
    runtime_receipt: dict[str, Any] | None
    playtest_receipt: dict[str, Any] | None
    visual_receipt: dict[str, Any] | None
    distribution_receipt: dict[str, Any] | None
    unresolved_gates: tuple[str, ...]
    release_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompleteProductionOrchestrator:
    """Approved request -> sharded source -> repair -> runtime -> release."""

    def __init__(
        self,
        *,
        workspace_root: str | Path = "mmm-output",
        profile: str = "t4_local",
        router_factory: Callable[[], ModelRouter] | None = None,
        policy: ScalePolicy | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.router_factory = router_factory or (
            lambda: ModelRouter(profile=profile)
        )
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    def execute(
        self,
        proposal: CompleteProposal | dict[str, Any],
        *,
        approval_hash: str,
        run_name: str,
        options: CompleteExecutionOptions | None = None,
        existing_input: str | Path | None = None,
    ) -> CompletePipelineResult:
        options = options or CompleteExecutionOptions()
        options.validate(policy=self.policy)
        parsed = (
            proposal
            if isinstance(proposal, CompleteProposal)
            else CompleteProposal.from_dict(proposal)
        )
        parsed.validate(policy=self.policy)
        approved = parsed.approve(approval_hash)
        if approved.status is not CompleteProposalStatus.APPROVED:
            raise SpecValidationError(
                "Complete proposal approval did not complete."
            )

        run_root = self._new_run_root(run_name)
        router: ModelRouter | None = None
        module_receipts: list[dict[str, Any]] = []
        blockbench_receipts: list[dict[str, Any]] = []
        world_receipt: dict[str, Any] | None = None
        asset_receipt: dict[str, Any] | None = None
        audio_receipt: dict[str, Any] | None = None
        runtime_receipt: dict[str, Any] | None = None
        playtest_receipt: dict[str, Any] | None = None
        visual_receipt: dict[str, Any] | None = None
        distribution_receipt: dict[str, Any] | None = None
        unresolved: list[str] = []

        project_root = self._prepare_project(
            approved,
            run_root=run_root,
            existing_input=existing_input,
        )
        self._write_complete_approval(project_root, approved)
        base = approved.base_proposal
        spec = base.spec
        ordered, collision_receipts = _normalize_modules(
            approved.modules,
            spec,
        )
        module_receipts.extend(collision_receipts)

        extended = generate_extended_content(
            project_root=project_root,
            mod_id=spec.mod_id,
            package_name=spec.package_name,
            modules=ordered,
            policy=self.policy,
        )
        if extended.get("status") != "SKIPPED":
            module_receipts.append(extended)

        for pack_id, pack_modules in _system_groups(ordered).items():
            module_receipts.append(
                generate_system_pack(
                    project_root=project_root,
                    pack_id=pack_id,
                    mod_id=spec.mod_id,
                    package_name=spec.package_name,
                    config={
                        "modules": [
                            _module_dict(item) for item in pack_modules
                        ]
                    },
                )
            )

        entity_modules = [
            item
            for item in ordered
            if item.kind in {"entity", "boss", "npc"}
        ]
        for module in entity_modules:
            config = module.config
            behavior_default = (
                "npc" if module.kind == "npc" else "hostile_melee"
            )
            receipt = generate_geckolib_entity_assets(
                project_root=project_root,
                mod_id=spec.mod_id,
                package_name=spec.package_name,
                entity_id=module.module_id,
                texture_width=int(config.get("texture_width", 64)),
                texture_height=int(config.get("texture_height", 64)),
                max_health=float(config.get("max_health", 80.0)),
                attack_damage=float(config.get("attack_damage", 8.0)),
                movement_speed=float(config.get("movement_speed", 0.27)),
                follow_range=float(config.get("follow_range", 40.0)),
                archetype=str(config.get("archetype", "biped")),
                behavior=str(
                    config.get("behavior", behavior_default)
                ),
                entity_width=float(config.get("entity_width", 0.8)),
                entity_height=float(config.get("entity_height", 2.0)),
                spawn_group=(
                    str(config.get("spawn_group"))
                    if config.get("spawn_group")
                    else None
                ),
                custom_bones=(
                    config.get("custom_bones")
                    if isinstance(config.get("custom_bones"), list)
                    else None
                ),
                policy=self.policy,
            )
            module_receipts.append(receipt)
            if options.run_blockbench and not options.source_only:
                blockbench_receipts.append(
                    self._blockbench_review(receipt, run_root)
                )
            elif options.run_blockbench:
                unresolved.append(
                    f"blockbench:{module.module_id}:not-run-in-source-only-mode"
                )

        if approved.world_ir is not None:
            world_output = run_root / "world-compiled"
            world_receipt = compile_scalable_world_ir(
                approved.world_ir,
                mod_id=spec.mod_id,
                output_root=world_output,
                package_world_zip=True,
                policy=self.policy,
            )
            self._merge_world_resources(
                world_output,
                project_root,
                spec.mod_id,
            )

        if approved.assets:
            router = router or self.router_factory()
            asset_receipt = self._generate_assets(
                router,
                approved,
                project_root,
                run_root,
            )
        if approved.audio:
            audio_receipt = generate_audio_assets(
                project_root=project_root,
                mod_id=spec.mod_id,
                package_name=spec.package_name,
                requests=approved.audio,
            )

        handled = _handled_module_ids(ordered)
        if approved.world_ir is None:
            handled -= {
                module.module_id
                for module in ordered
                if module.kind == "structure"
            }
        if not approved.audio:
            handled -= {
                module.module_id
                for module in ordered
                if module.kind == "audio"
            }
        custom_modules = [
            module
            for module in ordered
            if module.module_id not in handled
        ]
        for module in custom_modules:
            router = router or self.router_factory()
            module_receipts.append(
                CustomModuleGenerator(
                    router,
                    policy=self.policy,
                ).generate(
                    project_root,
                    module=module,
                    minecraft_version=spec.platform.minecraft_version,
                    loader=spec.platform.loader,
                    mappings=spec.platform.yarn_mappings,
                )
            )

        index = ProjectIndex(project_root, policy=self.policy)
        heap_receipt = tune_gradle_resources(
            project_root,
            module_count=len(ordered),
            source_file_count=len(index.files),
            policy=self._policy_with_heap_override(
                options.gradle_heap_mb
            ),
        )
        module_receipts.append(
            {
                "schema_version": "mmm/resource-tuning-v1",
                **heap_receipt,
            }
        )
        ProjectIndex(project_root, policy=self.policy).write_manifest()

        source_report = ScalableProjectValidator(
            policy=self.policy
        ).validate(project_root, spec).to_dict()
        if source_report.get("status") != "PASS":
            raise CompleteProductionError(
                "Generated complete project failed deterministic validation."
            )

        if options.run_jdt and not options.source_only:
            try:
                jdt_receipt = JavaLanguageService().diagnostics(
                    project_root,
                    timeout_seconds=90,
                )
            except Exception as exc:
                jdt_receipt = {
                    "status": "UNAVAILABLE",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            module_receipts.append(
                {
                    "schema_version": "mmm/jdt-gate-v1",
                    **jdt_receipt,
                }
            )
            if jdt_receipt.get("status") == "UNAVAILABLE":
                raise CompleteProductionError(
                    "JDT Language Server is required for a fully verified build: "
                    + str(jdt_receipt.get("error", "unavailable"))
                )
            errors = [
                item
                for item in jdt_receipt.get("diagnostics", [])
                if isinstance(item, dict)
                and int(item.get("severity", 1)) <= 2
            ]
            if errors and not options.auto_repair:
                raise CompleteProductionError(
                    "JDT reported errors and automatic repair is disabled."
                )

        if options.source_only:
            release = self._package_source_only(
                run_root,
                project_root,
                approved,
            )
            unresolved.extend(_external_gates(approved, options))
            return CompletePipelineResult(
                schema_version="mmm/complete-pipeline-result-v2",
                status="SOURCE_READY",
                project_root=str(project_root),
                release_zip=release,
                jar_path=None,
                complete_proposal_hash=approved.calculate_hash(),
                source_validation=source_report,
                build_report=None,
                jar_validation=None,
                module_receipts=tuple(module_receipts),
                world_receipt=world_receipt,
                asset_receipt=asset_receipt,
                audio_receipt=audio_receipt,
                blockbench_receipts=tuple(blockbench_receipts),
                runtime_receipt=None,
                playtest_receipt=None,
                visual_receipt=None,
                distribution_receipt=None,
                unresolved_gates=tuple(sorted(set(unresolved))),
                release_ready=False,
            )

        cache = run_root / ".cache/gradle"
        build = GradleRunner(cache).build(
            project_root,
            run_gametest=options.run_gametest,
        ).to_dict()
        if build.get("status") != "PASS" and options.auto_repair:
            router = router or self.router_factory()
            repair = RepairEngine(
                router=router,
                gradle_cache=cache,
                policy=self.policy,
            ).repair(
                project_root,
                run_gametest=options.run_gametest,
                max_attempts=options.max_repair_attempts,
            )
            module_receipts.append(
                {
                    "schema_version": "mmm/repair-receipt-v2",
                    **repair,
                }
            )
            build = GradleRunner(cache).build(
                project_root,
                run_gametest=options.run_gametest,
            ).to_dict()
        if build.get("status") != "PASS":
            raise CompleteProductionError(
                "Gradle/GameTest failed after the repair loop."
            )

        jar_path = _jar_path(build)
        jar_validation = validate_jar(jar_path, spec).to_dict()
        if jar_validation.get("status") != "PASS":
            raise CompleteProductionError(
                "Built JAR failed independent validation."
            )

        runtime_manager: MinecraftRuntimeManager | None = None
        try:
            if options.run_runtime:
                if not options.server_launcher:
                    raise CompleteProductionError(
                        "server_launcher is required for complete runtime verification."
                    )
                if not options.eula_accepted:
                    raise CompleteProductionError(
                        "Explicit Minecraft EULA acceptance is required."
                    )
                memory = (
                    options.server_memory_mb
                    or self.policy.runtime_heap_mb(
                        module_count=len(ordered),
                        entity_count=len(entity_modules),
                        structure_count=(
                            len(approved.world_ir.get("structures", []))
                            if approved.world_ir
                            else 0
                        ),
                    )
                )
                runtime_config = self._runtime_profile(
                    run_root,
                    memory,
                )
                runtime_manager = MinecraftRuntimeManager(
                    run_root,
                    config_path=runtime_config,
                )
                launcher_source = Path(
                    options.server_launcher
                ).expanduser().resolve()
                if (
                    not launcher_source.is_file()
                    or launcher_source.is_symlink()
                ):
                    raise CompleteProductionError(
                        "server_launcher must be a regular file."
                    )
                launcher_copy = (
                    run_root
                    / "integration-inputs/fabric-server-launch.jar"
                )
                launcher_copy.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copy2(launcher_source, launcher_copy)
                prepared = runtime_manager.prepare_instance(
                    "complete-integration",
                    mod_jar=jar_path,
                    server_launcher=launcher_copy,
                    eula_accepted=True,
                )
                server = runtime_manager.start_server(
                    timeout_seconds=180
                )
                client = (
                    runtime_manager.start_client()
                    if options.run_client
                    else None
                )
                runtime_receipt = {
                    "prepared": prepared,
                    "server": server,
                    "client": client,
                    "server_memory_mb": memory,
                }
            else:
                unresolved.append("runtime:not-requested")

            if options.run_mineflayer:
                if not options.run_runtime:
                    raise CompleteProductionError(
                        "Mineflayer requires the disposable runtime."
                    )
                playtest_receipt = self._run_playtest(
                    options.playtest_actions
                )
            else:
                unresolved.append("mineflayer:not-requested")

            if options.run_visual_review:
                if not options.screenshot_paths:
                    raise CompleteProductionError(
                        "Visual review requires explicit runtime screenshot paths."
                    )
                router = router or self.router_factory()
                visual_receipt = self._visual_review(
                    router,
                    approved,
                    options.screenshot_paths,
                )
                if visual_receipt.get("status") != "PASS":
                    raise CompleteProductionError(
                        "VisualCritic rejected the runtime screenshots."
                    )
            else:
                unresolved.append("visual-review:not-requested")
        finally:
            if runtime_manager is not None and options.cleanup_runtime:
                cleanup = runtime_manager.cleanup()
                runtime_receipt = {
                    **(runtime_receipt or {}),
                    "cleanup": cleanup,
                }

        from .mcp_tools import MMMToolService

        tool_service = MMMToolService(
            workspace_root=run_root,
            profile=self.profile,
        )
        release_result = tool_service.package_release(
            str(project_root),
            base.to_dict(),
            base.calculate_hash(),
            output_zip="releases/complete-release.zip",
            jar_path=str(jar_path),
        )
        release_zip = str(release_result["release_zip"])

        metadata = build_distribution_metadata(
            jar_path=jar_path,
            mod_id=spec.mod_id,
            version=spec.version,
            name=spec.mod_name,
            changelog=options.changelog,
        )
        bundle = package_distribution_bundle(
            metadata,
            output_zip=(
                run_root / "releases/distribution-bundle.zip"
            ),
            source_zip=release_zip,
        )
        distribution_receipt = {
            "metadata": metadata,
            "bundle": bundle,
        }
        if options.publish_provider == "modrinth":
            distribution_receipt["publish"] = publish_modrinth(
                metadata,
                project_id=str(options.publish_project_id),
            )
        elif options.publish_provider == "curseforge":
            distribution_receipt["publish"] = publish_curseforge(
                metadata,
                project_id=str(options.publish_project_id),
            )

        release_ready = not unresolved
        return CompletePipelineResult(
            schema_version="mmm/complete-pipeline-result-v2",
            status=(
                "VERIFIED"
                if release_ready
                else "BUILT_WITH_UNRESOLVED_GATES"
            ),
            project_root=str(project_root),
            release_zip=release_zip,
            jar_path=str(jar_path),
            complete_proposal_hash=approved.calculate_hash(),
            source_validation=source_report,
            build_report=build,
            jar_validation=jar_validation,
            module_receipts=tuple(module_receipts),
            world_receipt=world_receipt,
            asset_receipt=asset_receipt,
            audio_receipt=audio_receipt,
            blockbench_receipts=tuple(blockbench_receipts),
            runtime_receipt=runtime_receipt,
            playtest_receipt=playtest_receipt,
            visual_receipt=visual_receipt,
            distribution_receipt=distribution_receipt,
            unresolved_gates=tuple(sorted(set(unresolved))),
            release_ready=release_ready,
        )

    def _prepare_project(
        self,
        approved: CompleteProposal,
        *,
        run_root: Path,
        existing_input: str | Path | None,
    ) -> Path:
        if existing_input is not None:
            report = inspect_existing_project_archive(
                existing_input,
                extract_root=run_root / "existing-source",
            )
            if (
                approved.existing_input_sha256
                and report.archive_sha256
                != approved.existing_input_sha256
            ):
                raise CompleteProductionError(
                    "Existing input bytes changed after complete-plan approval."
                )
            if (
                not report.has_sources
                or not report.has_gradle_project
                or not report.extracted_to
            ):
                raise CompleteProductionError(
                    "Existing mod modification requires a source Gradle ZIP."
                )
            project_root = _locate_existing_fabric_root(
                Path(report.extracted_to).resolve()
            )
            info = inspect_fabric_project(project_root)
            expected = approved.base_proposal.spec
            if (
                info.mod_id != expected.mod_id
                or info.package_name != expected.package_name
            ):
                raise CompleteProductionError(
                    "Approved proposal does not match the existing Fabric project: "
                    f"expected {expected.mod_id}/{expected.package_name}, "
                    f"found {info.mod_id}/{info.package_name}."
                )
            return project_root

        base = approved.base_proposal
        base.approve(base.calculate_hash())
        project_root = (
            run_root / "base/workspaces" / base.spec.mod_id
        )
        FabricProjectGenerator().generate(base.spec, project_root)
        metadata = project_root / ".minecraft_ai"
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / "base-proposal.json").write_text(
            json.dumps(
                base.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return project_root.resolve()

    @staticmethod
    def _write_complete_approval(
        project_root: Path,
        proposal: CompleteProposal,
    ) -> None:
        rendered = (
            json.dumps(
                proposal.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        for path in (
            project_root / ".minecraft_ai/complete-proposal.json",
            project_root
            / "src/main/resources/META-INF/mmm-complete-proposal.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")

    def _new_run_root(self, run_name: str) -> Path:
        if not run_name or any(
            character
            not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in run_name
        ):
            raise CompleteProductionError(
                "run_name must use lowercase letters, numbers, underscore or hyphen."
            )
        root = (self.workspace_root / run_name).resolve()
        try:
            root.relative_to(self.workspace_root)
        except ValueError as exc:
            raise CompleteProductionError(
                "Run path escaped the workspace."
            ) from exc
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        return root

    @staticmethod
    def _merge_world_resources(
        world_output: Path,
        project_root: Path,
        mod_id: str,
    ) -> None:
        source = world_output / "data" / mod_id
        target = project_root / "src/main/resources/data" / mod_id
        if not source.is_dir():
            raise CompleteProductionError(
                "World compiler did not produce the expected namespace."
            )
        shutil.copytree(source, target, dirs_exist_ok=True)
        shutil.copy2(
            world_output / "mmm-world-manifest.json",
            project_root / ".minecraft_ai/mmm-world-manifest.json",
        )

    def _generate_assets(
        self,
        router: ModelRouter,
        proposal: CompleteProposal,
        project_root: Path,
        run_root: Path,
    ) -> dict[str, Any]:
        return generate_assets(
            router,
            proposal,
            project_root,
            run_root,
        )

    @staticmethod
    def _blockbench_review(
        gecko_receipt: dict[str, Any],
        run_root: Path,
    ) -> dict[str, Any]:
        return blockbench_review(gecko_receipt, run_root)

    @staticmethod
    def _run_playtest(
        actions: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        return run_playtest(actions)

    @staticmethod
    def _visual_review(
        router: ModelRouter,
        proposal: CompleteProposal,
        screenshots: tuple[str, ...],
    ) -> dict[str, Any]:
        return visual_review(router, proposal, screenshots)

    @staticmethod
    def _package_source_only(
        run_root: Path,
        project_root: Path,
        proposal: CompleteProposal,
    ) -> str:
        return package_source_only(run_root, project_root, proposal)

    def _policy_with_heap_override(
        self,
        override: int | None,
    ) -> ScalePolicy:
        if override is None:
            return self.policy
        policy = ScalePolicy(
            **{
                **self.policy.__dict__,
                "gradle_min_heap_mb": override,
                "gradle_max_heap_mb": override,
            }
        )
        policy.validate()
        return policy

    @staticmethod
    def _runtime_profile(
        run_root: Path,
        memory_mb: int,
    ) -> Path:
        return runtime_profile(run_root, memory_mb)
