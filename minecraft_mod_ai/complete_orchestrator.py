from __future__ import annotations

import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

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
    _jar_path,
    _locate_existing_fabric_root,
    _module_dict,
    _normalize_modules,
    _system_groups,
)
from .complete_spec import CompleteProposal, CompleteProposalStatus, ProductionModule
from .custom_module_generator import CustomModuleGenerator
from .extended_content_generator import generate_extended_content
from .final_artifact import (
    FinalArtifactError,
    build_requirement_coverage_receipt,
    load_or_empty_reuse_manifest,
    verify_final_mod_artifact,
    verify_runtime_artifact_binding,
    write_downloadable_bundle,
)
from .geckolib_generator import generate_geckolib_entity_assets
from .importer import ExistingProjectImportError, inspect_existing_project_archive
from .java_lsp import JavaLanguageService
from .validation_diagnostic_contract import (
    diagnostic_errors as jdt_diagnostic_errors,
    run_diagnostics as run_jdt_diagnostics,
)
from .local_ai_sidecar_generator import (
    INTEGRATION_TYPE as LOCAL_AI_SIDECAR_INTEGRATION_TYPE,
)
from .local_ai_sidecar_generator import generate_local_ai_sidecar
from .model_router import ModelRouter
from .production_contract import (
    evaluate_quality_contract,
    persist_quality_report,
    quality_unresolved,
)
from .project_edit import ProjectEditError, inspect_fabric_project
from .project_index import ProjectIndex
from .project_index_execution_reuse_contract import (
    execution_scoped,
    mark_post_generation,
    tune_gradle_resources,
)
from .project_index_execution_reuse_contract import (
    project_index as execution_project_index,
)
from .proposal_store import write_sharded_complete_proposal
from .publisher import (
    build_distribution_metadata,
    package_distribution_bundle,
    publish_curseforge,
    publish_modrinth,
)
from .quality_evidence import compile_quality_evidence
from .repair_engine import RepairEngine
from .research_ledger import is_research_shard, write_research_shard
from .runner import GradleRunner
from .runtime_manager import MinecraftRuntimeManager
from .scalable_generator import ScalableFabricProjectGenerator as FabricProjectGenerator
from .scalable_validator import ScalableProjectValidator
from .scale_policy import ScalePolicy
from .spec import SpecValidationError
from .system_pack_generator import generate_system_pack
from .validation_checkpoint_policy import (
    cached_validation_is_reusable,
    validation_checkpoint_input,
)
from .validator import validate_jar
from .work_graph import (
    DurableWorkLedger,
    WorkGraphError,
    WorkGraphPlan,
    WorkNode,
    build_production_work_plan,
    run_named_checkpoint,
)

_REQUIRED_GATE_TO_EVIDENCE = {'registry': 'source', 'resource': 'source', 'recipe': 'source', 'jdt': 'jdt', 'jdt diagnostics': 'jdt', 'gradle': 'gradle', 'gradle clean build': 'gradle', 'gametest': 'gametest', 'gametest spawn and attributes': 'gametest', 'jar validation': 'jar', 'blockbench uv and bone hierarchy review': 'blockbench', 'blockbench uv render review': 'blockbench', 'minecraft server client runtime': 'runtime_client', 'mineflayer playtest': 'playtest', 'runtime interaction tests': 'playtest', 'runtime animation review': 'runtime_visual', 'visual review': 'visual', 'client gui and validated network action test': 'playtest_visual', 'research ledger integrity': 'research_ledger'}


def _fork_custom_work_router(router: Any) -> Any:
    """Fork request-local state while retaining one managed model server."""

    from .custom_generation_research import _fork_router_for_candidate

    return _fork_router_for_candidate(router)


def _semantic_execution_observation(
    module: ProductionModule,
    receipt: dict[str, Any],
    *,
    dependent_ids: Iterable[str],
) -> dict[str, Any] | None:
    """Bind one persisted edit receipt back to its immutable semantic task."""

    config = module.config if isinstance(module.config, dict) else {}
    task = config.get("evidence_task")
    if not isinstance(task, dict):
        return None
    touched = sorted(
        {
            str(value).replace("\\", "/")
            for value in receipt.get("touched_paths", ())
            if isinstance(value, str) and value.strip()
        }
    )
    core: dict[str, Any] = {
        "schema_version": "mmm/semantic-task-observation-v1",
        "task_id": module.module_id,
        "task_sha256": str(task.get("task_sha256") or ""),
        "requirement_refs": list(task.get("requirement_refs") or ()),
        "gap_refs": list(task.get("gap_refs") or ()),
        "applied_action_count": int(receipt.get("operation_count") or 0),
        "touched_paths": touched,
        "touched_paths_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(
                touched,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "patch_receipt": receipt.get("patch_receipt"),
        "source_observation_receipt": receipt.get("source_observation_receipt"),
        "impact_probes": list(task.get("impact_probes") or ()),
        "affected_downstream_task_ids": sorted(set(dependent_ids)),
        "status": "OBSERVED",
    }
    core["observation_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return core

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
    changelog: str = 'Generated and verified by M.M.M'
    gradle_heap_mb: int | None = None
    server_memory_mb: int | None = None
    resume: bool = True

    def validate(self, *, policy: ScalePolicy | None=None) -> None:
        policy = policy or ScalePolicy.from_environment()
        if self.max_repair_attempts is not None and (type(self.max_repair_attempts) is not int or self.max_repair_attempts < 1):
            raise CompleteProductionError('max_repair_attempts must be null or a positive integer.')
        if self.publish_provider not in {None, 'modrinth', 'curseforge'}:
            raise CompleteProductionError('publish_provider must be modrinth or curseforge.')
        if self.publish_provider and (not self.publish_project_id):
            raise CompleteProductionError('publish_project_id is required when publishing.')
        if self.source_only and self.publish_provider:
            raise CompleteProductionError('Source-only execution cannot publish a binary release.')
        if self.gradle_heap_mb is not None and self.gradle_heap_mb < 512:
            raise CompleteProductionError('gradle_heap_mb must be at least 512 when supplied.')
        if self.server_memory_mb is not None and self.server_memory_mb < 1024:
            raise CompleteProductionError('server_memory_mb must be at least 1024 when supplied.')
        if self.gradle_heap_mb is not None and self.gradle_heap_mb > policy.gradle_max_heap_mb:
            raise CompleteProductionError('gradle_heap_mb exceeds MMM_GRADLE_MAX_HEAP_MB host policy.')
        if self.server_memory_mb is not None and self.server_memory_mb > policy.runtime_max_heap_mb:
            raise CompleteProductionError('server_memory_mb exceeds MMM_RUNTIME_MAX_HEAP_MB host policy.')
        if type(self.resume) is not bool:
            raise CompleteProductionError('resume must be boolean.')

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
    asset_receipt: dict[str, Any] | None
    blockbench_receipts: tuple[dict[str, Any], ...]
    runtime_receipt: dict[str, Any] | None
    playtest_receipt: dict[str, Any] | None
    visual_receipt: dict[str, Any] | None
    distribution_receipt: dict[str, Any] | None
    unresolved_gates: tuple[str, ...]
    release_ready: bool
    work_graph_hash: str
    work_ledger_path: str
    run_resumed: bool
    quality_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class CompleteProductionOrchestrator:
    """Approved request -> sharded source -> repair -> runtime -> release."""

    def __init__(self, *, workspace_root: str | Path='mmm-output', profile: str='t4_local', router_factory: Callable[[], ModelRouter] | None=None, policy: ScalePolicy | None=None) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.router_factory = router_factory or (lambda: ModelRouter(profile=profile))
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    @execution_scoped
    def execute(self, proposal: CompleteProposal | dict[str, Any], *, approval_hash: str, run_name: str, options: CompleteExecutionOptions | None=None, existing_input: str | Path | None=None) -> CompletePipelineResult:
        options = options or CompleteExecutionOptions()
        options.validate(policy=self.policy)
        parsed = proposal if isinstance(proposal, CompleteProposal) else CompleteProposal.from_dict(proposal)
        parsed.validate(policy=self.policy)
        approved = parsed.approve(approval_hash)
        if approved.status is not CompleteProposalStatus.APPROVED:
            raise SpecValidationError('Complete proposal approval did not complete.')
        input_is_bound = bool(approved.existing_input_sha256)
        input_is_supplied = existing_input is not None
        if input_is_bound != input_is_supplied:
            if input_is_bound:
                raise CompleteProductionError('This approved complete plan is bound to an existing-project ZIP, so the same ZIP is required.')
            raise CompleteProductionError('An existing-project ZIP may be used only with a complete plan that was approved with that input.')
        base = approved.base_proposal
        spec = base.spec
        ordered, collision_receipts = _normalize_modules(approved.modules, spec)
        work_plan = build_production_work_plan(approved, policy=self.policy, modules=ordered)
        run_root, ledger, run_resumed = self._open_run(run_name, work_plan, resume=options.resume)
        router: ModelRouter | None = None
        module_receipts: list[dict[str, Any]] = []
        blockbench_receipts: list[dict[str, Any]] = []
        asset_receipt: dict[str, Any] | None = None
        jdt_receipt: dict[str, Any] | None = None
        runtime_receipt: dict[str, Any] | None = None
        playtest_receipt: dict[str, Any] | None = None
        visual_receipt: dict[str, Any] | None = None
        distribution_receipt: dict[str, Any] | None = None
        unresolved: list[str] = []
        module_receipts.extend(collision_receipts)
        project_root = run_named_checkpoint(ledger, 'prepare-project', stage='prepare', input_value={'graph_hash': work_plan.graph_hash, 'existing_input_sha256': approved.existing_input_sha256}, action=lambda: self._prepare_project(approved, run_root=run_root, existing_input=existing_input), encode=lambda value: {'project_root': str(value)}, decode=lambda receipt: Path(str(receipt['project_root'])).resolve(), validate_cached=self._valid_project_root)
        self._write_complete_approval(project_root, approved)
        self._succeed_work_node(ledger, 'prepare-project', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'SUCCEEDED', 'project_root': str(project_root)})
        generation = self._execute_generation_work(approved=approved, ordered=ordered, work_plan=work_plan, ledger=ledger, project_root=project_root, run_root=run_root, options=options, router=router)
        mark_post_generation()
        module_receipts.extend(generation['module_receipts'])
        blockbench_receipts.extend(generation['blockbench_receipts'])
        unresolved.extend(generation['unresolved'])
        asset_receipt = generation['asset_receipt']
        router = generation['router']
        index = execution_project_index(ProjectIndex, project_root, policy=self.policy)
        heap_receipt = run_named_checkpoint(ledger, 'tune-resources', stage='prepare:resources', input_value={'graph_hash': work_plan.graph_hash, 'module_count': len(ordered), 'source_file_count': len(index.files), 'gradle_heap_mb': options.gradle_heap_mb}, action=lambda: tune_gradle_resources(project_root, module_count=len(ordered), source_file_count=len(index.files), policy=self._policy_with_heap_override(options.gradle_heap_mb)), encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda _cached: False)
        module_receipts.append({'schema_version': 'mmm/resource-tuning-v1', **heap_receipt})
        execution_project_index(ProjectIndex, project_root, policy=self.policy).write_manifest()
        generated_manifest_hash = self._project_manifest_hash(project_root)
        source_report = run_named_checkpoint(ledger, 'validate-source', stage='validate:source', input_value=validation_checkpoint_input('validate-source', {'graph_hash': work_plan.graph_hash, 'project_manifest': self._project_manifest_hash(project_root)}), action=lambda: ScalableProjectValidator(policy=self.policy).validate(project_root, spec).to_dict(), encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: cached_validation_is_reusable('validate-source', cached))
        if source_report.get('status') != 'PASS':
            raise CompleteProductionError('Generated complete project failed deterministic validation.')
        self._succeed_work_node(ledger, 'validate-source', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'PASS', 'checks_run': source_report.get('checks_run', 0), 'project_manifest': generated_manifest_hash})
        self._persist_work_evidence(project_root, ledger, work_plan)
        if options.run_jdt and (not options.source_only):

            def run_jdt() -> dict[str, Any]:
                return run_jdt_diagnostics(
                    JavaLanguageService,
                    project_root,
                    timeout_seconds=90,
                )
            jdt_receipt = run_named_checkpoint(ledger, 'validate-jdt', stage='validate:jdt', input_value=validation_checkpoint_input('validate-jdt', {'graph_hash': work_plan.graph_hash, 'project_manifest': self._project_manifest_hash(project_root)}), action=run_jdt, encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: cached_validation_is_reusable('validate-jdt', cached))
            module_receipts.append({'schema_version': 'mmm/jdt-gate-v1', **jdt_receipt})
            print('[JDT RECEIPT] ' + json.dumps(jdt_receipt, ensure_ascii=False, sort_keys=True, default=str), flush=True)
            # JDT is an auxiliary source diagnostic.  Its publication gap must
            # remain visible and non-PASS, but must not abort the node before
            # the authoritative clean Gradle verifier can run.
            errors = jdt_diagnostic_errors(jdt_receipt)
            if errors and (not options.auto_repair):
                raise CompleteProductionError('JDT reported errors and automatic repair is disabled.')
        if options.source_only:
            release = run_named_checkpoint(ledger, 'package-source', stage='package:source', input_value={'graph_hash': work_plan.graph_hash, 'project_manifest': self._project_manifest_hash(project_root)}, action=lambda: self._package_source_only(run_root, project_root, approved), encode=lambda value: {'release_zip': value}, decode=lambda cached: str(cached['release_zip']), validate_cached=lambda value: Path(value).is_file())
            unresolved.extend(_external_gates(approved, options))
            quality_report = self._evaluate_quality(approved=approved, run_root=run_root, project_root=project_root, source_validation=source_report, build_report=None, jar_validation=None, module_receipts=module_receipts, asset_receipt=asset_receipt, blockbench_receipts=blockbench_receipts, runtime_receipt=None, playtest_receipt=None, visual_receipt=None)
            if quality_report is not None:
                unresolved.extend(f'quality:{dimension_id}' for dimension_id in quality_unresolved(quality_report))
                self._record_quality_nodes(ledger, quality_report, allow_success=False)
            self._persist_work_evidence(project_root, ledger, work_plan)
            return CompletePipelineResult(schema_version='mmm/complete-pipeline-result-v3', status='SOURCE_READY', project_root=str(project_root), release_zip=release, jar_path=None, complete_proposal_hash=approved.calculate_hash(), source_validation=source_report, build_report=None, jar_validation=None, module_receipts=tuple(module_receipts), asset_receipt=asset_receipt, blockbench_receipts=tuple(blockbench_receipts), runtime_receipt=None, playtest_receipt=None, visual_receipt=None, distribution_receipt=None, unresolved_gates=tuple(sorted(set(unresolved))), release_ready=False, work_graph_hash=work_plan.graph_hash, work_ledger_path=str(ledger.path), run_resumed=run_resumed, quality_report=quality_report)
        cache = run_root / '.cache/gradle'

        def build_with_repair() -> dict[str, Any]:
            nonlocal router
            build_result = GradleRunner(cache).build(project_root, run_gametest=options.run_gametest).to_dict()
            repair_result: dict[str, Any] | None = None
            if build_result.get('status') != 'PASS' and options.auto_repair:
                router = router or self.router_factory()
                repair_result = RepairEngine(router=router, gradle_cache=cache, policy=self.policy).repair(project_root, run_gametest=options.run_gametest, max_attempts=options.max_repair_attempts)
                build_result = GradleRunner(cache).build(project_root, run_gametest=options.run_gametest).to_dict()
            return {'build': build_result, 'repair': repair_result}
        build_bundle = run_named_checkpoint(ledger, 'gradle-build', stage='build', input_value={'graph_hash': work_plan.graph_hash, 'project_manifest': self._project_manifest_hash(project_root), 'run_gametest': options.run_gametest, 'auto_repair': options.auto_repair, 'max_repair_attempts': options.max_repair_attempts}, action=build_with_repair, encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: self._cached_build_exists(cached.get('build')))
        build = build_bundle['build']
        repair = build_bundle.get('repair')
        if isinstance(repair, dict):
            module_receipts.append({'schema_version': 'mmm/repair-receipt-v2', **repair})
        if build.get('status') != 'PASS':
            raise CompleteProductionError('Gradle/GameTest failed after the repair loop.')
        reported_jar = _jar_path(build)
        try:
            artifact_receipt = verify_final_mod_artifact(
                project_root,
                expected_mod_id=spec.mod_id,
                expected_loader=spec.platform.loader,
                expected_minecraft_version=spec.platform.minecraft_version,
                expected_java=spec.platform.java_version,
                expected_gradle=spec.platform.gradle,
            ).to_dict()
        except FinalArtifactError as exc:
            raise CompleteProductionError(
                f'Final generated project has no uniquely verified production JAR: {exc}'
            ) from exc
        jar_path = Path(str(artifact_receipt['artifact_path'])).resolve()
        if reported_jar.resolve() != jar_path:
            raise CompleteProductionError(
                'Gradle build report does not identify the sole verified production JAR.'
            )
        successful_commands = {
            str(item.get('name'))
            for item in build.get('commands', ())
            if isinstance(item, dict)
            and item.get('exit_code') == 0
            and item.get('timed_out', False) is False
        }
        if not successful_commands.intersection({'build', 'clean_build'}):
            raise CompleteProductionError(
                'Final project has no passing full Gradle build command receipt.'
            )
        build = dict(build)
        build['artifact_receipt'] = artifact_receipt
        build_receipt = {
            'schema_version': 'mmm/final-build-receipt-v1',
            'status': 'PASS',
            'toolchain_attested': True,
            'compile_java': 'PASS',
            'tests': 'PASS',
            'gradle_build': 'PASS',
            'gametest': 'PASS' if options.run_gametest else 'NOT_REQUIRED',
            'production_jar': 'PASS',
            'jar_integrity': artifact_receipt['integrity'],
            'mod_metadata': 'PASS',
            'artifact_sha256': artifact_receipt['sha256'],
            'artifact': artifact_receipt['artifact'],
            'toolchain': {
                'loader': artifact_receipt['loader'],
                'minecraft_version': artifact_receipt['minecraft_version'],
                'java': artifact_receipt['java'],
                'gradle': artifact_receipt['gradle'],
            },
            'commands': list(build.get('commands', ())),
        }
        metadata_root = project_root / '.minecraft_ai'
        metadata_root.mkdir(parents=True, exist_ok=True)
        (metadata_root / 'artifact-receipt.json').write_text(
            json.dumps(artifact_receipt, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        (metadata_root / 'build-receipt.json').write_text(
            json.dumps(build_receipt, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        self._succeed_work_node(ledger, 'build-project', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'PASS', 'build': build, 'final_build_receipt': build_receipt})
        jar_validation = run_named_checkpoint(ledger, 'validate-jar', stage='validate:jar', input_value={'graph_hash': work_plan.graph_hash, 'jar_sha256': self._file_hash(jar_path)}, action=lambda: validate_jar(jar_path, spec).to_dict(), encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda _cached: jar_path.is_file())
        if jar_validation.get('status') != 'PASS':
            raise CompleteProductionError('Built JAR failed independent validation.')
        self._succeed_work_node(ledger, 'validate-jar', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'PASS', 'jar_sha256': self._file_hash(jar_path), 'checks_run': jar_validation.get('checks_run', 0)})
        runtime_manager: MinecraftRuntimeManager | None = None
        try:
            if options.run_runtime:
                if not options.server_launcher:
                    raise CompleteProductionError('server_launcher is required for complete runtime verification.')
                if not options.eula_accepted:
                    raise CompleteProductionError('Explicit Minecraft EULA acceptance is required.')
                memory = options.server_memory_mb or self.policy.runtime_heap_mb(module_count=len(ordered), entity_count=sum(1 for module in ordered if module.kind in {'entity', 'boss', 'npc'}), structure_count=sum(1 for module in ordered if module.kind == 'structure'))
                runtime_config = self._runtime_profile(run_root, memory)
                runtime_manager = MinecraftRuntimeManager(run_root, config_path=runtime_config)
                launcher_source = Path(options.server_launcher).expanduser().resolve()
                if not launcher_source.is_file() or launcher_source.is_symlink():
                    raise CompleteProductionError('server_launcher must be a regular file.')
                launcher_copy = run_root / 'integration-inputs/fabric-server-launch.jar'
                launcher_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(launcher_source, launcher_copy)
                prepared = runtime_manager.prepare_instance(
                    'complete-integration',
                    mod_jar=jar_path,
                    server_launcher=launcher_copy,
                    eula_accepted=True,
                    expected_mod_sha256=str(artifact_receipt['sha256']),
                )
                server = runtime_manager.start_server(timeout_seconds=180)
                client = runtime_manager.start_client() if options.run_client else None
                runtime_receipt = {
                    'schema_version': 'mmm/final-runtime-receipt-v1',
                    'status': 'PASS',
                    'artifact_sha256': artifact_receipt['sha256'],
                    'prepared': prepared,
                    'server': server,
                    'client': client,
                    'server_memory_mb': memory,
                }
                verify_runtime_artifact_binding(
                    runtime_receipt, str(artifact_receipt['sha256'])
                )
            else:
                unresolved.append('runtime:not-requested')
            if options.run_mineflayer:
                if not options.run_runtime:
                    raise CompleteProductionError('Mineflayer requires the disposable runtime.')
                playtest_receipt = self._run_playtest(options.playtest_actions)
            else:
                unresolved.append('mineflayer:not-requested')
            if options.run_visual_review:
                if not options.screenshot_paths:
                    raise CompleteProductionError('Visual review requires explicit runtime screenshot paths.')
                router = router or self.router_factory()
                visual_receipt = self._visual_review(router, approved, options.screenshot_paths)
                if visual_receipt.get('status') != 'PASS':
                    raise CompleteProductionError('VisualCritic rejected the runtime screenshots.')
            else:
                unresolved.append('visual-review:not-requested')
        finally:
            if runtime_manager is not None and options.cleanup_runtime:
                cleanup = runtime_manager.cleanup()
                runtime_receipt = {**(runtime_receipt or {}), 'cleanup': cleanup}
        persisted_runtime_receipt = runtime_receipt or {
            'schema_version': 'mmm/final-runtime-receipt-v1',
            'status': 'NOT_REQUIRED',
            'artifact_sha256': artifact_receipt['sha256'],
        }
        (metadata_root / 'runtime-receipt.json').write_text(
            json.dumps(persisted_runtime_receipt, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        runtime_verified = not approved.external_runtime_required or (runtime_receipt is not None and playtest_receipt is not None and (visual_receipt is not None))
        if runtime_verified:
            self._succeed_work_node(ledger, 'runtime-playtest', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'NOT_REQUIRED' if not approved.external_runtime_required else 'PASS', 'runtime': runtime_receipt, 'playtest': playtest_receipt, 'visual': visual_receipt})
        else:
            ledger.fail('runtime-playtest', 'Runtime, interaction, and visual evidence are still required.', input_required=True)
        self._persist_work_evidence(project_root, ledger, work_plan)
        quality_report = self._evaluate_quality(approved=approved, run_root=run_root, project_root=project_root, source_validation=source_report, build_report=build, jar_validation=jar_validation, module_receipts=module_receipts, asset_receipt=asset_receipt, blockbench_receipts=blockbench_receipts, runtime_receipt=runtime_receipt, playtest_receipt=playtest_receipt, visual_receipt=visual_receipt)
        quality_passed = quality_report is None or quality_report.get('overall_status') == 'PASS'
        if quality_report is not None:
            unresolved.extend(f'quality:{dimension_id}' for dimension_id in quality_unresolved(quality_report))
            self._record_quality_nodes(ledger, quality_report, allow_success=True)
        self._persist_work_evidence(project_root, ledger, work_plan)
        contract = approved.game_design.get('_production_contract')
        coverage_receipt = build_requirement_coverage_receipt(
            contract=contract if isinstance(contract, dict) else None,
            proposal_hash=approved.calculate_hash(),
            quality_report=quality_report,
            artifact_sha256=str(artifact_receipt['sha256']),
            unresolved_gates=tuple(sorted(set(unresolved))),
        )
        (metadata_root / 'requirement-coverage.json').write_text(
            json.dumps(coverage_receipt, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        reuse_manifest = load_or_empty_reuse_manifest(project_root, spec.mod_id)
        if not (project_root / 'reuse-manifest.json').is_file() and not (
            project_root / '.minecraft_ai/reuse-manifest.json'
        ).is_file():
            (metadata_root / 'reuse-manifest.json').write_text(
                json.dumps(reuse_manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                encoding='utf-8',
            )
        from .mcp_tools import MMMToolService
        tool_service = MMMToolService(workspace_root=run_root, profile=self.profile)
        release_result = run_named_checkpoint(ledger, 'package-release', stage='package', input_value={'graph_hash': work_plan.graph_hash, 'proposal_hash': base.calculate_hash(), 'jar_sha256': self._file_hash(jar_path)}, action=lambda: tool_service.package_release(str(project_root), base.to_dict(), base.calculate_hash(), output_zip='releases/complete-release.zip', jar_path=str(jar_path)), encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: Path(str(cached.get('release_zip', ''))).is_file())
        release_zip = str(release_result['release_zip'])
        metadata = build_distribution_metadata(jar_path=jar_path, mod_id=spec.mod_id, version=spec.version, name=spec.mod_name, changelog=options.changelog, platform_lock=spec.platform)
        bundle = package_distribution_bundle(metadata, output_zip=run_root / 'releases/distribution-bundle.zip', source_zip=release_zip)
        distribution_receipt = {'metadata': metadata, 'bundle': bundle}
        release_ready = (
            not unresolved
            and quality_passed
            and coverage_receipt.get('status') == 'PASS'
        )
        if options.publish_provider and (not release_ready):
            raise CompleteProductionError('Publishing is blocked because required verification gates remain unresolved.')
        if options.publish_provider == 'modrinth':
            distribution_receipt['publish'] = publish_modrinth(metadata, project_id=str(options.publish_project_id))
        elif options.publish_provider == 'curseforge':
            distribution_receipt['publish'] = publish_curseforge(metadata, project_id=str(options.publish_project_id))
        if release_ready:
            distribution_receipt['downloadable_bundle'] = write_downloadable_bundle(
                run_root / 'releases/final-mod-download',
                artifact_receipt=artifact_receipt,
                requirement_coverage=coverage_receipt,
                reuse_manifest=reuse_manifest,
                build_receipt=build_receipt,
                runtime_receipt=persisted_runtime_receipt,
            )
        if release_ready:
            self._succeed_work_node(ledger, 'package-release', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'PASS', 'release_zip': release_zip})
        else:
            ledger.fail('package-release', 'Release quality evidence is incomplete.', input_required=True)
        self._persist_work_evidence(project_root, ledger, work_plan)
        return CompletePipelineResult(schema_version='mmm/complete-pipeline-result-v3', status='VERIFIED' if release_ready else 'BUILT_WITH_UNRESOLVED_GATES', project_root=str(project_root), release_zip=release_zip, jar_path=str(jar_path), complete_proposal_hash=approved.calculate_hash(), source_validation=source_report, build_report=build, jar_validation=jar_validation, module_receipts=tuple(module_receipts), asset_receipt=asset_receipt, blockbench_receipts=tuple(blockbench_receipts), runtime_receipt=runtime_receipt, playtest_receipt=playtest_receipt, visual_receipt=visual_receipt, distribution_receipt=distribution_receipt, unresolved_gates=tuple(sorted(set(unresolved))), release_ready=release_ready, work_graph_hash=work_plan.graph_hash, work_ledger_path=str(ledger.path), run_resumed=run_resumed, quality_report=quality_report)

    def _evaluate_quality(self, *, approved: CompleteProposal, run_root: Path, project_root: Path, source_validation: dict[str, Any] | None, build_report: dict[str, Any] | None, jar_validation: dict[str, Any] | None, module_receipts: Iterable[dict[str, Any]], asset_receipt: dict[str, Any] | None, blockbench_receipts: Iterable[dict[str, Any]], runtime_receipt: dict[str, Any] | None, playtest_receipt: dict[str, Any] | None, visual_receipt: dict[str, Any] | None) -> dict[str, Any] | None:
        contract = approved.game_design.get('_production_contract')
        if approved.schema_version != 'mmm/complete-proposal-v2':
            return None
        if not isinstance(contract, dict):
            raise CompleteProductionError('Complete proposal v2 is missing its production contract.')
        proposal_hash = approved.calculate_hash()
        evidence = compile_quality_evidence(contract, proposal_hash, game_design=approved.game_design, source_validation=source_validation, build_report=build_report, jar_validation=jar_validation, module_receipts=module_receipts, asset_receipt=asset_receipt, blockbench_receipts=blockbench_receipts, runtime_receipt=runtime_receipt, playtest_receipt=playtest_receipt, visual_receipt=visual_receipt)
        report_path = run_root / '.minecraft_ai/quality-convergence.json'
        previous = self._read_quality_report(report_path)
        current_ids = {dimension_id: str(receipt.get('receipt_id', '')) for dimension_id, receipt in evidence.items()}
        if previous is not None:
            prior_ids = {str(item.get('dimension_id', '')): str(item.get('receipt_id', '')) for item in previous.get('dimensions', []) if isinstance(item, dict) and item.get('status') == 'PASS'}
            if previous.get('proposal_hash') == proposal_hash and previous.get('contract_sha256') == contract.get('contract_sha256') and (prior_ids == current_ids):
                report = previous
            else:
                report = evaluate_quality_contract(contract, evidence, proposal_hash)
        else:
            report = evaluate_quality_contract(contract, evidence, proposal_hash)
        persist_quality_report(report_path, report)
        project_report = project_root / '.minecraft_ai/quality-convergence.json'
        if project_report.resolve() != report_path.resolve():
            persist_quality_report(project_report, report)
        return report

    def _read_quality_report(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if not path.is_file() or path.is_symlink():
            raise CompleteProductionError('Existing quality report must be a regular file.')
        if path.stat().st_size > self.policy.mcp_page_bytes * 8:
            raise CompleteProductionError('Existing quality report exceeds the size policy.')
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompleteProductionError(f'Existing quality report is invalid: {exc}') from exc
        if not isinstance(value, dict):
            raise CompleteProductionError('Existing quality report must contain an object.')
        try:
            quality_unresolved(value)
        except ValueError as exc:
            raise CompleteProductionError(f'Existing quality report failed validation: {exc}') from exc
        return value

    def _record_quality_nodes(self, ledger: DurableWorkLedger, report: dict[str, Any], *, allow_success: bool) -> None:
        for dimension in report.get('dimensions', []):
            if not isinstance(dimension, dict):
                continue
            dimension_id = str(dimension.get('dimension_id', ''))
            node_id = 'validate-quality-' + dimension_id.replace('_', '-')
            if allow_success and dimension.get('status') == 'PASS':
                self._succeed_work_node(ledger, node_id, {'schema_version': 'mmm/quality-work-node-receipt-v1', 'status': 'PASS', 'dimension_id': dimension_id, 'receipt_id': dimension.get('receipt_id', ''), 'receipt_sha256': dimension.get('receipt_sha256', '')})
            else:
                ledger.fail(node_id, str(dimension.get('reason') or 'Quality evidence is missing.'), input_required=True)

    def _execute_generation_work(self, *, approved: CompleteProposal, ordered: list[ProductionModule], work_plan: WorkGraphPlan, ledger: DurableWorkLedger, project_root: Path, run_root: Path, options: CompleteExecutionOptions, router: ModelRouter | None) -> dict[str, Any]:
        """Execute durable generation nodes with capacity-owned, event-driven lanes."""
        import threading
        import time
        from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

        from . import scheduler_parallel_safety_contract as scheduler_safety
        spec = approved.base_proposal.spec
        if spec.platform.minecraft_version:
            os.environ["MMM_MINECRAFT_VERSION"] = str(spec.platform.minecraft_version).strip()
        if spec.platform.loader:
            os.environ["MMM_LOADER"] = str(spec.platform.loader).strip()
        if spec.platform.yarn_mappings:
            os.environ["MMM_YARN_MAPPINGS"] = str(spec.platform.yarn_mappings).strip()
        module_lookup = {module.module_id: module for module in ordered}
        direct_dependents: dict[str, set[str]] = {
            module.module_id: set() for module in ordered
        }
        for module in ordered:
            for dependency in module.depends_on:
                direct_dependents.setdefault(dependency, set()).add(module.module_id)

        def downstream_ids(module_id: str) -> tuple[str, ...]:
            pending = list(direct_dependents.get(module_id, ()))
            affected: set[str] = set()
            while pending:
                candidate = pending.pop()
                if candidate in affected:
                    continue
                affected.add(candidate)
                pending.extend(direct_dependents.get(candidate, ()))
            return tuple(sorted(affected))
        research_modules = tuple(module for module in ordered if is_research_shard(module))
        asset_lookup = {item.asset_id: item for item in approved.assets}
        generation_nodes = tuple(node for node in work_plan.nodes if node.stage.startswith('generate:'))
        node_by_id = {node.node_id: node for node in generation_nodes}
        generation_stages = tuple(sorted({node.stage for node in generation_nodes}))
        extended_kinds = {'item', 'block', 'tool', 'weapon', 'armor', 'food', 'crop', 'machine', 'effect', 'enchantment', 'command', 'recipe', 'advancement', 'loot'}
        module_receipts: list[dict[str, Any]] = []
        blockbench_receipts: list[dict[str, Any]] = []
        unresolved: list[str] = []
        asset_shards: list[dict[str, Any]] = []
        runtime_init_lock = threading.RLock()

        def get_router() -> ModelRouter:
            nonlocal router
            if router is None:
                with runtime_init_lock:
                    if router is None:
                        router = self.router_factory()
            return router
        shared_project_index = execution_project_index(ProjectIndex, project_root, policy=self.policy)
        fallback_custom_generator: CustomModuleGenerator | None = None

        def new_custom_generator() -> CustomModuleGenerator:
            nonlocal fallback_custom_generator
            base_router = get_router()
            worker_router = _fork_custom_work_router(base_router)
            if worker_router is base_router:
                # Legacy routers and test doubles that cannot fork retain the
                # instance-level safety lock. Real ModelRouter workers isolate only
                # workspace/tool state and still share one llama-server/model copy.
                with runtime_init_lock:
                    if fallback_custom_generator is None:
                        fallback_custom_generator = CustomModuleGenerator(base_router, policy=self.policy, fast_mode=getattr(self, '_fast_mode', False), project_index=shared_project_index, checkpoint_root=run_root / '.minecraft_ai' / '.mmm-custom-checkpoints')
                    return fallback_custom_generator
            return CustomModuleGenerator(worker_router, policy=self.policy, fast_mode=getattr(self, '_fast_mode', False), project_index=shared_project_index, checkpoint_root=run_root / '.minecraft_ai' / '.mmm-custom-checkpoints')

        def module_node_action(node: WorkNode, members: list[ProductionModule]) -> dict[str, Any]:
            stage = str(node.payload.get('generation_stage', ''))
            receipts: list[dict[str, Any]] = []
            node_custom_generator: CustomModuleGenerator | None = None

            def generate_custom(module: ProductionModule) -> dict[str, Any]:
                nonlocal node_custom_generator
                if node_custom_generator is None:
                    node_custom_generator = new_custom_generator()
                return node_custom_generator.generate(project_root, module=module, research_modules=research_modules, minecraft_version=spec.platform.minecraft_version, loader=spec.platform.loader, mappings=spec.platform.yarn_mappings)

            if stage == 'content':
                research_shards = [module for module in members if is_research_shard(module)]
                receipts.extend(write_research_shard(project_root, module=module) for module in research_shards)
                deterministic = [module for module in members if module.kind in extended_kinds and module not in research_shards]
                if deterministic:
                    receipts.append(generate_extended_content(project_root=project_root, mod_id=spec.mod_id, package_name=spec.package_name, modules=deterministic, policy=self.policy))
                sidecars = [module for module in members if module.kind == 'integration' and module.config.get('integration_type') == LOCAL_AI_SIDECAR_INTEGRATION_TYPE]
                receipts.extend(generate_local_ai_sidecar(project_root=project_root, mod_id=spec.mod_id, package_name=spec.package_name, module=module, policy=self.policy) for module in sidecars)
                receipts.extend(generate_custom(module) for module in members if module.kind not in extended_kinds and module not in sidecars and (module not in research_shards))
            elif stage == 'system':
                for pack_id, pack_modules in _system_groups(members).items():
                    receipts.append(generate_system_pack(project_root=project_root, pack_id=pack_id, mod_id=spec.mod_id, package_name=spec.package_name, config={'modules': [_module_dict(item) for item in pack_modules]}, policy=self.policy))
            elif stage == 'entity':
                for module in members:
                    config = module.config
                    behavior_default = 'npc' if module.kind == 'npc' else 'hostile_melee'
                    receipts.append(generate_geckolib_entity_assets(project_root=project_root, mod_id=spec.mod_id, package_name=spec.package_name, entity_id=module.module_id, texture_width=int(config.get('texture_width', 64)), texture_height=int(config.get('texture_height', 64)), max_health=float(config.get('max_health', 80.0)), attack_damage=float(config.get('attack_damage', 8.0)), movement_speed=float(config.get('movement_speed', 0.27)), follow_range=float(config.get('follow_range', 40.0)), archetype=str(config.get('archetype', 'biped')), behavior=str(config.get('behavior', behavior_default)), entity_width=float(config.get('entity_width', 0.8)), entity_height=float(config.get('entity_height', 2.0)), spawn_group=str(config['spawn_group']) if config.get('spawn_group') else None, custom_bones=config.get('custom_bones') if isinstance(config.get('custom_bones'), list) else None, policy=self.policy))
            elif stage == 'custom':
                receipts.extend(generate_custom(module) for module in members)
            else:
                raise CompleteProductionError(f'Unsupported generation work stage: {stage}')
            semantic_observations = [
                observation
                for module, receipt in zip(members, receipts, strict=False)
                if isinstance(receipt, dict)
                and (
                    observation := _semantic_execution_observation(
                        module,
                        receipt,
                        dependent_ids=downstream_ids(module.module_id),
                    )
                )
                is not None
            ]
            return {'schema_version': 'mmm/generation-work-node-v1', 'status': 'SUCCEEDED', 'node_id': node.node_id, 'stage': stage, 'module_ids': [module.module_id for module in members], 'receipts': receipts, 'semantic_observations': semantic_observations}

        def process_node(node: WorkNode) -> None:
            if not node.stage.startswith('generate:'):
                return
            kind = str(node.payload.get('kind', ''))
            if kind == 'module-shard':
                member_ids = [str(item.get('module_id')) for item in node.payload.get('members', []) if isinstance(item, dict)]
                if not member_ids or any(item not in module_lookup for item in member_ids):
                    raise CompleteProductionError(f'Work node {node.node_id} has invalid module members.')
                members = [module_lookup[item] for item in member_ids]
                receipt = self._run_work_node(ledger, node, action=lambda node=node, members=members: module_node_action(node, members), validate_cached=lambda value: self._receipt_outputs_exist(value, project_root=project_root), shared_index=shared_project_index)
                children = [item for item in receipt.get('receipts', []) if isinstance(item, dict)]
                module_receipts.extend(children)
                if node.payload.get('generation_stage') == 'entity':
                    entity_receipts = {str(item.get('entity_id')): item for item in children if item.get('entity_id')}
                    for module in members:
                        entity_receipt = entity_receipts.get(module.module_id)
                        if entity_receipt is None:
                            raise CompleteProductionError(f'Entity generation node omitted its receipt: {module.module_id}')
                        if options.run_blockbench and (not options.source_only):
                            blockbench_receipts.append(run_named_checkpoint(ledger, f'blockbench-review-{module.module_id}', stage='validate:blockbench', input_value={'graph_hash': work_plan.graph_hash, 'entity_receipt': entity_receipt}, action=lambda receipt=entity_receipt: self._blockbench_review(receipt, run_root), encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: Path(str(cached.get('preview', ''))).is_file()))
                        elif options.run_blockbench:
                            unresolved.append(f'blockbench:{module.module_id}:not-run-in-source-only-mode')
            elif kind == 'asset-shard':
                ids = [str(item.get('asset_id')) for item in node.payload.get('members', []) if isinstance(item, dict)]
                if not ids or any(item not in asset_lookup for item in ids):
                    raise CompleteProductionError(f'Work node {node.node_id} has invalid assets.')
                shard_proposal = replace(approved, assets=tuple(asset_lookup[item] for item in ids), approval_hash='')
                asset_shards.append(self._run_work_node(ledger, node, action=lambda proposal=shard_proposal: self._generate_assets(get_router(), proposal, project_root, run_root), validate_cached=lambda cached: all(Path(str(item.get('target', ''))).is_file() for item in cached.get('assets', [])), shared_index=shared_project_index))
            else:
                raise CompleteProductionError(f'Unsupported work node payload kind: {kind}')
        capacities = scheduler_safety._capacities()
        cpu_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['cpu_io'])), thread_name_prefix='cpu_io')
        llm_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['llm'])), thread_name_prefix='llm')
        image_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['image_gpu'])), thread_name_prefix='image_gpu')
        commit_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['commit'])), thread_name_prefix='commit')
        node_futures: dict[str, Future[Any]] = {}
        idle_wait = threading.Event()
        lease_seconds = 900
        heartbeat_seconds = 60.0

        def dispatch_node(node: WorkNode) -> Future[Any]:
            resource_class = node.resource_class or str(node.payload.get('resource_class', 'cpu_io'))
            if resource_class == 'llm':
                return llm_pool.submit(process_node, node)
            if resource_class == 'image_gpu':
                return image_pool.submit(process_node, node)
            if resource_class == 'commit':
                return commit_pool.submit(process_node, node)
            return cpu_pool.submit(process_node, node)
        try:
            while True:
                ledger.raise_if_cancelled()
                done_ids = [node_id for node_id, future in node_futures.items() if future.done()]
                for node_id in done_ids:
                    future = node_futures.pop(node_id)
                    try:
                        future.result()
                    except BaseException as exc:
                        print(f"\n[ORCHESTRATOR ERROR] Node {node_id} failed with {type(exc).__name__}:\n{exc}\n", flush=True)
                        raise CompleteProductionError(f'Pipeline generation node failed: {node_id}: {type(exc).__name__}: {exc}') from exc
                while True:
                    claimed = ledger.claim_ready(worker_id='mmm-orchestrator', stages=generation_stages, lease_seconds=lease_seconds)
                    if claimed is None:
                        break
                    node_id = str(claimed['node_id'])
                    node = node_by_id.get(node_id)
                    if node is None:
                        raise CompleteProductionError(f'Ledger claimed an unknown generation node: {node_id}')
                    if node_id in node_futures:
                        raise CompleteProductionError(f'Generation node was claimed twice: {node_id}')
                    node_futures[node_id] = dispatch_node(node)
                if node_futures:
                    wait(tuple(node_futures.values()), timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
                    continue
                task_rows = {node.node_id: ledger.task(node.node_id) for node in generation_nodes}
                states = {node_id: str(task['state']) for node_id, task in task_rows.items()}
                failed_ids = [node_id for node_id, state in states.items() if state in {'failed', 'cancelled', 'input_required'}]
                if failed_ids:
                    raise CompleteProductionError(f'Pipeline generation failed on nodes: {failed_ids}')
                if all(state in {'succeeded', 'completed'} for state in states.values()):
                    break
                running = [task for task in task_rows.values() if str(task['state']) == 'running']
                if running:
                    deadlines = [float(task['lease_until']) for task in running if task.get('lease_until') is not None]
                    delay = 1.0
                    if deadlines:
                        delay = min(heartbeat_seconds, max(0.05, min(deadlines) - time.time() + 0.01))
                    idle_wait.wait(delay)
                    continue
                pending_ids = [node_id for node_id, state in states.items() if state not in {'succeeded', 'completed'}]
                if pending_ids:
                    raise CompleteProductionError(f'WorkGraph DAG deadlock: pending nodes remain but no ready nodes are available: {pending_ids}')
                break
        finally:
            cpu_pool.shutdown(wait=True, cancel_futures=True)
            llm_pool.shutdown(wait=True, cancel_futures=True)
            image_pool.shutdown(wait=True, cancel_futures=True)
            commit_pool.shutdown(wait=True, cancel_futures=True)
        asset_receipt = {'schema_version': 'mmm/complete-assets-sharded-v1', 'status': 'GENERATED', 'shard_count': len(asset_shards), 'asset_count': sum(len(item.get('assets', [])) for item in asset_shards), 'shards': asset_shards} if asset_shards else None
        return {'module_receipts': module_receipts, 'blockbench_receipts': blockbench_receipts, 'asset_receipt': asset_receipt, 'unresolved': unresolved, 'router': router}

    @staticmethod
    def _run_work_node(ledger: DurableWorkLedger, node: WorkNode, *, action: Callable[[], dict[str, Any]], validate_cached: Callable[[dict[str, Any]], bool], shared_index: ProjectIndex | None=None) -> dict[str, Any]:
        cached = ledger.cached_receipt(node.node_id, input_hash=node.input_hash)
        if cached is not None and validate_cached(cached):
            return cached
        if cached is not None:
            ledger.invalidate(node.node_id)
        current = ledger.task(node.node_id)
        if current['state'] in {'failed', 'input_required', 'cancelled'}:
            ledger.retry(node.node_id)
            current = ledger.task(node.node_id)
        ledger.raise_if_cancelled()
        if current['state'] != 'running':
            ledger.begin(node.node_id, worker_id='complete-orchestrator')
        try:
            receipt = action()
            if not isinstance(receipt, dict):
                raise CompleteProductionError(f'Work node {node.node_id} returned a non-object receipt.')
            ledger.raise_if_cancelled()
            ledger.succeed(node.node_id, receipt)
            if shared_index is not None:
                touched = receipt.get('touched_paths') or receipt.get('written_files') or []
                if touched:
                    try:
                        shared_index.update_files(touched)
                        shared_index.write_manifest()
                    except Exception:
                        pass
            return receipt
        except BaseException as exc:
            try:
                if ledger.task(node.node_id)['state'] == 'running':
                    ledger.fail(node.node_id, f'{type(exc).__name__}: {exc}')
            except WorkGraphError:
                pass
            raise

    def _prepare_project(self, approved: CompleteProposal, *, run_root: Path, existing_input: str | Path | None) -> Path:
        if existing_input is not None:
            try:
                report = inspect_existing_project_archive(existing_input, extract_root=run_root / 'existing-source', expected_archive_sha256=approved.existing_input_sha256)
            except ExistingProjectImportError as exc:
                raise CompleteProductionError(str(exc)) from exc
            if not report.has_sources or not report.has_gradle_project or (not report.extracted_to):
                raise CompleteProductionError('Existing mod modification requires a source Gradle ZIP.')
            project_root = self._locate_imported_project(report, run_root=run_root)
            info = inspect_fabric_project(project_root)
            expected = approved.base_proposal.spec
            if info.mod_id != expected.mod_id or info.package_name != expected.package_name:
                raise CompleteProductionError(f'Approved proposal does not match the existing Fabric project: expected {expected.mod_id}/{expected.package_name}, found {info.mod_id}/{info.package_name}.')
            return project_root
        base = approved.base_proposal
        base.approve(base.calculate_hash())
        project_root = run_root / 'base/workspaces' / base.spec.mod_id
        if project_root.exists():
            if self._project_matches_spec(project_root, base.spec):
                self._write_base_proposal(project_root, base)
                return project_root.resolve()
            self._preserve_partial_project(project_root)
        staging = project_root.with_name(f'.{project_root.name}.staging')
        if staging.exists():
            self._preserve_partial_project(staging)
        FabricProjectGenerator(policy=self.policy).generate(base.spec, staging)
        self._write_base_proposal(staging, base)
        if project_root.exists():
            if self._project_matches_spec(project_root, base.spec):
                self._preserve_partial_project(staging)
                return project_root.resolve()
            self._preserve_partial_project(project_root)
        staging.replace(project_root)
        return project_root.resolve()

    def _locate_imported_project(self, report: Any, *, run_root: Path) -> Path:
        extracted = Path(str(report.extracted_to)).resolve()
        try:
            return _locate_existing_fabric_root(extracted)
        except CompleteProductionError as outer_error:
            nested_members = sorted({path.split('!/', 1)[0] for path in report.source_files if '!/' in path} & {path.split('!/', 1)[0] for path in report.gradle_files if '!/' in path})
            if len(nested_members) != 1:
                raise outer_error
            nested_archive = (extracted / Path(*nested_members[0].split('/'))).resolve()
            try:
                nested_archive.relative_to(extracted)
            except ValueError as exc:
                raise CompleteProductionError('Nested source archive escaped the validated import.') from exc
            if not nested_archive.is_file() or nested_archive.is_symlink():
                raise CompleteProductionError('Nested source archive is missing after validated extraction.')
            try:
                nested_report = inspect_existing_project_archive(nested_archive, extract_root=run_root / 'existing-source-nested')
            except ExistingProjectImportError as exc:
                raise CompleteProductionError(f'Nested source archive failed validation: {exc}') from exc
            if not nested_report.has_sources or not nested_report.has_gradle_project or (not nested_report.extracted_to):
                raise CompleteProductionError('Nested release source is not an editable Gradle project.')
            return _locate_existing_fabric_root(Path(nested_report.extracted_to).resolve())

    @staticmethod
    def _project_matches_spec(project_root: Path, spec: Any) -> bool:
        if not project_root.is_dir() or project_root.is_symlink():
            return False
        try:
            info = inspect_fabric_project(project_root)
        except (OSError, ValueError, json.JSONDecodeError, ProjectEditError):
            return False
        has_build = any((project_root / name).is_file() and (not (project_root / name).is_symlink()) for name in ('build.gradle', 'build.gradle.kts'))
        return has_build and info.main_java.is_file() and (not info.main_java.is_symlink()) and (info.mod_id == spec.mod_id) and (info.package_name == spec.package_name)

    @staticmethod
    def _write_base_proposal(project_root: Path, base: Any) -> None:
        metadata = project_root / '.minecraft_ai'
        if metadata.exists() and (not metadata.is_dir() or metadata.is_symlink()):
            raise CompleteProductionError('Generated project metadata path is not a safe directory.')
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / 'base-proposal.json').write_text(json.dumps(base.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    @staticmethod
    def _preserve_partial_project(path: Path) -> Path:
        parent = path.parent.resolve()
        resolved = path.resolve(strict=False)
        if resolved.parent != parent:
            raise CompleteProductionError('Partial project path escaped its run-owned workspace.')
        revision = 1
        while True:
            candidate = parent / f'{path.name}.incomplete-{revision}'
            if not candidate.exists():
                path.rename(candidate)
                return candidate
            revision += 1

    @staticmethod
    def _succeed_work_node(ledger: DurableWorkLedger, node_id: str, receipt: dict[str, Any]) -> None:
        ledger.raise_if_cancelled()
        current = ledger.task(node_id)
        state = str(current['state'])
        if state == 'succeeded':
            return
        if state in {'failed', 'input_required', 'cancelled'}:
            ledger.retry(node_id)
        ledger.begin(node_id, worker_id='complete-orchestrator')
        ledger.succeed(node_id, receipt)

    @staticmethod
    def _persist_work_evidence(project_root: Path, ledger: DurableWorkLedger, plan: WorkGraphPlan) -> None:
        metadata = project_root / '.minecraft_ai'
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / 'work-graph.json').write_text(json.dumps(plan.to_dict(include_payloads=False), ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        ledger.export_receipts(metadata / 'production-receipts.jsonl')

    def _write_complete_approval(self, project_root: Path, proposal: CompleteProposal) -> None:
        for path in (project_root / '.minecraft_ai/complete-proposal.json', project_root / 'src/main/resources/META-INF/mmm-complete-proposal.json'):
            write_sharded_complete_proposal(proposal, path, shard_size=max(1, self.policy.java_shard_size), policy=self.policy)

    def _open_run(self, run_name: str, plan: WorkGraphPlan, *, resume: bool) -> tuple[Path, DurableWorkLedger, bool]:
        if not run_name or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for character in run_name):
            raise CompleteProductionError('run_name must use lowercase letters, numbers, underscore or hyphen.')
        revision = 1
        while True:
            suffix = '' if revision == 1 else f'-revision-{revision}'
            root = (self.workspace_root / f'{run_name}{suffix}').resolve()
            try:
                root.relative_to(self.workspace_root)
            except ValueError as exc:
                raise CompleteProductionError('Run path escaped the workspace.') from exc
            ledger_path = root / '.minecraft_ai/work-ledger.sqlite3'
            if root.exists():
                if revision == 1 and resume and ledger_path.is_file() and (not ledger_path.is_symlink()):
                    try:
                        ledger = DurableWorkLedger(ledger_path, proposal_hash=plan.proposal_hash, graph_hash=plan.graph_hash)
                        ledger.sync_plan(plan)
                    except WorkGraphError as exc:
                        if 'different approved proposal' not in str(exc):
                            raise CompleteProductionError(f'Existing run ledger is invalid: {exc}') from exc
                    else:
                        self._write_work_graph_summary(root, plan)
                        return (root, ledger, True)
                revision += 1
                continue
            try:
                root.mkdir(parents=True)
            except FileExistsError:
                revision += 1
                continue
            ledger = DurableWorkLedger(ledger_path, proposal_hash=plan.proposal_hash, graph_hash=plan.graph_hash)
            ledger.sync_plan(plan)
            self._write_work_graph_summary(root, plan)
            return (root, ledger, False)

    @staticmethod
    def _write_work_graph_summary(run_root: Path, plan: WorkGraphPlan) -> None:
        target = run_root / '.minecraft_ai/work-graph.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(plan.to_dict(include_payloads=False), ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    @staticmethod
    def _valid_project_root(path: Path) -> bool:
        if not path.is_dir() or path.is_symlink():
            return False
        if not any((path / name).is_file() and (not (path / name).is_symlink()) for name in ('build.gradle', 'build.gradle.kts')):
            return False
        try:
            info = inspect_fabric_project(path)
        except (OSError, ValueError, json.JSONDecodeError, ProjectEditError):
            return False
        return info.main_java.is_file() and (not info.main_java.is_symlink())

    @staticmethod
    def _receipt_outputs_exist(receipt: dict[str, Any], *, project_root: Path) -> bool:
        if receipt.get('status') == 'SKIPPED':
            return True
        raw_paths: list[str] = []
        research_outputs: list[dict[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                if value.get('schema_version') == 'mmm/research-ledger-write-receipt-v1':
                    research_outputs.append(value)
                for key, nested in value.items():
                    if key in {'files', 'generated_files'} and isinstance(nested, list):
                        raw_paths.extend(str(item) for item in nested if isinstance(item, str))
                    elif isinstance(nested, (dict, list)):
                        collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)
        collect(receipt)
        for research in research_outputs:
            raw = research.get('target_path')
            expected = research.get('sha256')
            if not isinstance(raw, str) or not isinstance(expected, str):
                return False
            path = (project_root / raw).resolve()
            try:
                path.relative_to(project_root.resolve())
            except ValueError:
                return False
            if not path.is_file() or path.is_symlink():
                return False
            if CompleteProductionOrchestrator._file_hash(path) != expected:
                return False
        if not raw_paths:
            return CompleteProductionOrchestrator._valid_project_root(project_root)
        for raw in raw_paths:
            path = Path(raw)
            path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
            if not path.is_file() or path.is_symlink():
                return False
        return True

    def _project_manifest_hash(self, project_root: Path) -> str:
        return str(execution_project_index(ProjectIndex, project_root, policy=self.policy).manifest_receipt()['sha256'])

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return 'sha256:' + digest.hexdigest()

    @staticmethod
    def _cached_build_exists(build: Any) -> bool:
        if not isinstance(build, dict) or build.get('status') != 'PASS':
            return False
        raw = build.get('jar_path')
        if not isinstance(raw, str):
            return False
        path = Path(raw).expanduser().resolve()
        return path.is_file() and (not path.is_symlink())

    @staticmethod
    def _required_gate_failures(proposal: CompleteProposal, *, generated_receipts: Iterable[Any], project_root: Path | None=None, source_validation: dict[str, Any] | None, jdt_receipt: dict[str, Any] | None, build_report: dict[str, Any] | None, jar_validation: dict[str, Any] | None, blockbench_receipts: Iterable[dict[str, Any]], runtime_receipt: dict[str, Any] | None, playtest_receipt: dict[str, Any] | None, visual_receipt: dict[str, Any] | None) -> list[str]:
        """Resolve every declared gate against an explicit evidence receipt.

        Unknown gate names are deliberately unresolved. Merely generating a
        ``required_gates`` string is never treated as proof that the gate ran.
        """
        receipt_values = tuple(generated_receipts)
        requirements: set[tuple[str, str]] = {(module.module_id, gate.strip()) for module in proposal.modules for gate in module.required_gates if gate.strip()}
        research_ledger_receipts: list[dict[str, Any]] = []

        def collect(value: Any, owner: str='generated') -> None:
            if isinstance(value, dict):
                if value.get('schema_version') == 'mmm/research-ledger-write-receipt-v1':
                    research_ledger_receipts.append(value)
                local_owner = next((str(value[key]) for key in ('module_id', 'entity_id', 'pack_id', 'sound_id') if isinstance(value.get(key), str) and value[key]), owner)
                gates = value.get('required_gates')
                if isinstance(gates, (list, tuple)):
                    requirements.update((local_owner, gate.strip()) for gate in gates if isinstance(gate, str) and gate.strip())
                for key, nested in value.items():
                    if key != 'required_gates':
                        collect(nested, local_owner)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    collect(nested, owner)
        for receipt in receipt_values:
            collect(receipt)
        expected_research = {module.module_id: (str(module.config.get('receipt', {}).get('shard_sha256', '')), str(module.config.get('receipt', {}).get('facts_sha256', ''))) for module in proposal.modules if module.kind == 'integration' and module.config.get('integration_type') == 'mmm_research_shard'}

        def research_file_matches(receipt: dict[str, Any]) -> bool:
            if project_root is None:
                return False
            raw = receipt.get('target_path')
            expected = receipt.get('sha256')
            if not isinstance(raw, str) or not isinstance(expected, str):
                return False
            path = (project_root / raw).resolve()
            try:
                path.relative_to(project_root.resolve())
            except ValueError:
                return False
            return path.is_file() and (not path.is_symlink()) and (CompleteProductionOrchestrator._file_hash(path) == expected)
        passed_research = {str(receipt.get('module_id')): (str(receipt.get('shard_sha256', '')), str(receipt.get('corpus_sha256', ''))) for receipt in research_ledger_receipts if receipt.get('status') in {'WRITTEN', 'VERIFIED_EXISTING'} and research_file_matches(receipt)}
        gradle_passed = isinstance(build_report, dict) and build_report.get('status') == 'PASS' and CompleteProductionOrchestrator._command_receipt_passed(build_report, 'clean_build')
        jdt_passed = isinstance(jdt_receipt, dict) and (
            (jdt_receipt.get('status') != 'UNAVAILABLE' and int(jdt_receipt.get('error_count', -1)) == 0 and int(jdt_receipt.get('files_opened', 0)) > 0)
            or (jdt_receipt.get('status') == 'UNAVAILABLE' and gradle_passed)
        )
        evidence = {'source': isinstance(source_validation, dict) and source_validation.get('status') == 'PASS', 'jdt': jdt_passed, 'gradle': gradle_passed, 'gametest': gradle_passed and CompleteProductionOrchestrator._gametest_receipt_passed(build_report, proposal.base_proposal.spec), 'jar': isinstance(jar_validation, dict) and jar_validation.get('status') == 'PASS', 'runtime_client': isinstance(runtime_receipt, dict) and isinstance(runtime_receipt.get('server'), dict) and (runtime_receipt['server'].get('server_running') is True) and isinstance(runtime_receipt.get('client'), dict) and (runtime_receipt['client'].get('client_running') is True), 'playtest': isinstance(playtest_receipt, dict) and playtest_receipt.get('status') == 'PASS' and (int(playtest_receipt.get('interaction_count', 0)) > 0) and (int(playtest_receipt.get('assertion_count', 0)) > 0), 'visual': isinstance(visual_receipt, dict) and visual_receipt.get('status') == 'PASS', 'research_ledger': bool(expected_research) and all((passed_research.get(module_id) == hashes for module_id, hashes in expected_research.items()))}
        evidence['runtime_visual'] = evidence['runtime_client'] and evidence['visual']
        evidence['playtest_visual'] = evidence['playtest'] and evidence['visual']
        blockbench = tuple(blockbench_receipts)
        entity_ids = {module.module_id for module in proposal.modules if module.kind in {'entity', 'boss', 'npc'}}

        def blockbench_passed(owner: str) -> bool:
            expected = {owner} if owner in entity_ids else entity_ids
            if not expected:
                return False
            passed = {str(receipt.get('entity')) for receipt in blockbench if isinstance(receipt.get('uv'), dict) and receipt['uv'].get('status') in {'PASS', 'OK'} and isinstance(receipt.get('preview'), str) and Path(receipt['preview']).is_file() and (not Path(receipt['preview']).is_symlink())}
            return expected <= passed
        failures: list[str] = []
        for owner, gate in sorted(requirements):
            normalized = _normalize_required_gate(gate)
            evidence_key = _REQUIRED_GATE_TO_EVIDENCE.get(normalized)
            if evidence_key is None:
                failures.append(f'required-gate:{owner}:{gate}:unsupported')
                continue
            passed = blockbench_passed(owner) if evidence_key == 'blockbench' else bool(evidence.get(evidence_key, False))
            if not passed:
                failures.append(f'required-gate:{owner}:{gate}:missing-{evidence_key}')
        return failures

    @staticmethod
    def _command_receipt_passed(build_report: dict[str, Any] | None, name: str) -> bool:
        if not isinstance(build_report, dict):
            return False
        return any(isinstance(command, dict) and command.get('name') == name and (command.get('exit_code') == 0) and (command.get('timed_out') is not True) for command in build_report.get('commands', []))

    @staticmethod
    def _gametest_receipt_passed(build_report: dict[str, Any] | None, spec: Any) -> bool:
        if not CompleteProductionOrchestrator._command_receipt_passed(build_report, 'gametest') or not isinstance(build_report, dict) or (not isinstance(build_report.get('gametest_report'), str)):
            return False
        raw_report_path = Path(build_report['gametest_report']).expanduser()
        if raw_report_path.is_symlink():
            return False
        report_path = raw_report_path.resolve()
        if not report_path.is_file():
            return False
        try:
            root = ET.parse(report_path).getroot()
        except (ET.ParseError, OSError):
            return False
        testcases = list(root.iter('testcase'))
        if not testcases:
            return False
        for suite in root.iter('testsuite'):
            for aggregate in ('failures', 'errors', 'skipped'):
                value = suite.attrib.get(aggregate)
                if value is None:
                    continue
                try:
                    if int(value) != 0:
                        return False
                except ValueError:
                    return False
        if any(testcase.find('failure') is not None or testcase.find('error') is not None or testcase.find('skipped') is not None for testcase in testcases):
            return False
        main_class = ''.join(part.capitalize() for part in spec.mod_id.split('_')) + 'Mod'
        expected = f'{main_class}GameTests.generatedRegistriesAreLive'.lower()
        return any(testcase.attrib.get('name', '').lower() == expected for testcase in testcases)

    def _generate_assets(self, router: ModelRouter, proposal: CompleteProposal, project_root: Path, run_root: Path) -> dict[str, Any]:
        return generate_assets(router, proposal, project_root, run_root)

    @staticmethod
    def _blockbench_review(gecko_receipt: dict[str, Any], run_root: Path) -> dict[str, Any]:
        return blockbench_review(gecko_receipt, run_root)

    @staticmethod
    def _run_playtest(actions: Iterable[dict[str, Any]]) -> dict[str, Any]:
        return run_playtest(actions)

    @staticmethod
    def _visual_review(router: ModelRouter, proposal: CompleteProposal, screenshots: tuple[str, ...]) -> dict[str, Any]:
        return visual_review(router, proposal, screenshots)

    @staticmethod
    def _package_source_only(run_root: Path, project_root: Path, proposal: CompleteProposal) -> str:
        return package_source_only(run_root, project_root, proposal)

    def _policy_with_heap_override(self, override: int | None) -> ScalePolicy:
        if override is None:
            return self.policy
        policy = ScalePolicy(**{**self.policy.__dict__, 'gradle_min_heap_mb': override, 'gradle_max_heap_mb': override})
        policy.validate()
        return policy

    @staticmethod
    def _runtime_profile(run_root: Path, memory_mb: int) -> Path:
        return runtime_profile(run_root, memory_mb)

def _normalize_required_gate(value: str) -> str:
    return ' '.join(''.join(character.casefold() if character.isalnum() else ' ' for character in value).split())

