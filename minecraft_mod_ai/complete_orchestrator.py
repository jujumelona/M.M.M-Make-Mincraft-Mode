from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .artifact_graph_executor import execute_artifact_graph
from .artifact_job import ArtifactJob
from .artifact_materializer import ensure_artifact_scaffolding
from .complete_build_repair import run_build_repair_checkpoint
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
from .custom_module_generator import (
    CustomModuleGenerator,
    finalize_persisted_generation_checkpoint,
)
from .execution_feedback_replan_contract import (
    execution_feedback_scoped,
    feedback_run_context,
)
from .execution_feedback_replan_contract import (
    semantic_execution_observation as _semantic_execution_observation,
)
from .extended_content_generator import generate_extended_content
from .final_artifact import (
    FinalArtifactError,
    build_debug_fixture_coverage_receipt,
    build_requirement_coverage_receipt,
    load_or_empty_reuse_manifest,
    verify_debug_fixture_source,
    verify_final_mod_artifact,
    verify_runtime_artifact_binding,
    write_build_artifact_bundle,
    write_downloadable_bundle,
)
from .geckolib_generator import generate_geckolib_entity_assets
from .importer import ExistingProjectImportError, inspect_existing_project_archive
from .java_lsp import JavaLanguageService
from .local_ai_sidecar_generator import (
    INTEGRATION_TYPE as LOCAL_AI_SIDECAR_INTEGRATION_TYPE,
)
from .local_ai_sidecar_generator import generate_local_ai_sidecar
from .model_concurrency import run_with_model_execution_deadline
from .model_router import ModelRouter
from .platform_catalog import adapter_for_lock_values, adapter_from_project
from .prepared_project_resume_integrity import (
    prepared_project_cache_valid,
    prepared_project_matches_spec,
)
from .production_contract import (
    evaluate_quality_contract,
    persist_quality_report,
    quality_unresolved,
)
from .project_edit import inspect_fabric_project
from .project_index import ProjectIndex
from .project_index_execution_reuse_contract import (
    execution_scoped,
    mark_post_generation,
    tune_gradle_resources,
)
from .project_index_execution_reuse_contract import (
    project_index as execution_project_index,
)
from .project_index_execution_reuse_contract import (
    update_from_receipt as update_execution_project_index_from_receipt,
)
from .proposal_store import write_sharded_complete_proposal
from .publisher import (
    build_distribution_metadata,
    package_distribution_bundle,
    publish_curseforge,
    publish_modrinth,
)
from .quality_evidence import compile_quality_evidence
from .research_ledger import is_research_shard, write_research_shard
from .root_cause_trace import emit_root_cause
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
from .validation_diagnostic_contract import (
    diagnostic_errors as jdt_diagnostic_errors,
)
from .validation_diagnostic_contract import (
    run_diagnostics as run_jdt_diagnostics,
)
from .validation_diagnostic_contract import unwrap_diagnostic_receipt
from .validator import validate_jar
from .work_graph import (
    DurableWorkLedger,
    WorkGraphError,
    WorkGraphPlan,
    WorkNode,
    build_production_work_plan,
    run_named_checkpoint,
)


def _jdt_verification_timeout_seconds() -> int:
    """Return bounded verifier time for Gradle-backed JDT workspace bootstrap."""

    raw = os.environ.get("MMM_JDT_VERIFICATION_TIMEOUT_SECONDS", "180").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 180
    return max(30, min(value, 600))


def _jdt_verification_attempts() -> int:
    raw = os.environ.get("MMM_JDT_VERIFICATION_ATTEMPTS", "2").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 2
    return max(1, min(value, 3))


def _retryable_jdt_bootstrap_failure(receipt: dict[str, Any] | None) -> bool:
    normalized, _path = unwrap_diagnostic_receipt(receipt)
    if not normalized:
        return False
    if str(normalized.get("status") or "").strip().upper() != "UNAVAILABLE":
        return False
    error = str(normalized.get("error") or "").casefold()
    return "serviceready" in error and "not observed" in error


def _run_release_jdt_verification(
    project_root: str | Path,
    *,
    timeout_seconds: int | None = None,
    attempts: int | None = None,
) -> dict[str, Any]:
    """Retry only transient JDT ServiceReady bootstrap misses; remain fail-closed."""

    per_attempt = (
        int(timeout_seconds)
        if timeout_seconds is not None
        else _jdt_verification_timeout_seconds()
    )
    total_attempts = (
        int(attempts) if attempts is not None else _jdt_verification_attempts()
    )
    total_attempts = max(1, min(total_attempts, 3))
    receipt: dict[str, Any] = {}
    for attempt in range(1, total_attempts + 1):
        receipt = run_jdt_diagnostics(
            JavaLanguageService,
            project_root,
            timeout_seconds=per_attempt,
        )
        if _jdt_release_evidence_passed(receipt):
            result = dict(receipt)
            result["verification_attempts"] = attempt
            return result
        if _blocking_jdt_errors(receipt) or not _retryable_jdt_bootstrap_failure(receipt):
            result = dict(receipt)
            result["verification_attempts"] = attempt
            return result
        emit_root_cause(
            "jdt_release_retry",
            stage="verify",
            operation="java_diagnostics",
            gate="jdt_service_ready",
            result="RETRY",
            reason="JDT ServiceReady was not observed; retrying cold bootstrap",
            details={
                "attempt": attempt,
                "max_attempts": total_attempts,
                "timeout_seconds": per_attempt,
            },
        )
    result = dict(receipt)
    result["verification_attempts"] = total_attempts
    return result


_REQUIRED_GATE_TO_EVIDENCE = {
    'registry': 'source',
    'resource': 'source',
    'recipe': 'source',
    'source static validation': 'source',
    'generated resource validation': 'source',
    'jdt': 'jdt',
    'jdt diagnostics': 'jdt',
    'gradle': 'gradle',
    'gradle clean build': 'gradle',
    'target compile': 'gradle',
    'gametest': 'gametest',
    'gametest spawn and attributes': 'gametest',
    'worldgen runtime validation': 'gametest',
    'jar': 'jar',
    'jar validation': 'jar',
    'runtime': 'runtime_client',
    'minecraft server client runtime': 'runtime_client',
    'network protocol validation': 'playtest',
    'mineflayer playtest': 'playtest',
    'runtime interaction tests': 'playtest',
    'runtime animation review': 'runtime_visual',
    'blockbench uv and bone hierarchy review': 'blockbench',
    'blockbench uv render review': 'blockbench',
    'visual review': 'visual',
    'client gui and validated network action test': 'playtest_visual',
    'research ledger integrity': 'research_ledger',
}


def _fork_custom_work_router(router: Any) -> Any:
    """Fork request-local state while retaining one managed model server."""

    from .custom_generation_research import _fork_router_for_candidate

    return _fork_router_for_candidate(router)


def _receipt_owned_module_ids(receipt: dict[str, Any]) -> tuple[str, ...]:
    """Return only module ownership explicitly declared by a generation receipt."""
    owned: set[str] = set()
    for key in ("module_ids", "modules"):
        values = receipt.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                owned.add(value.strip())
            elif isinstance(value, dict):
                module_id = value.get("module_id")
                if isinstance(module_id, str) and module_id.strip():
                    owned.add(module_id.strip())
    for key in ("module_id", "entity_id"):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            owned.add(value.strip())
    return tuple(sorted(owned))


def _semantic_execution_observations(
    members: Iterable[ProductionModule],
    receipts: Iterable[dict[str, Any]],
    *,
    downstream_ids: Callable[[str], Iterable[str]],
    evidence_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Attribute receipts by explicit ownership and fail closed on evidence gaps."""
    member_by_id = {module.module_id: module for module in members}
    member_ids = set(member_by_id)
    tracked_ids = {
        module_id
        for module_id, module in member_by_id.items()
        if isinstance(module.config, dict)
        and isinstance(module.config.get("evidence_task"), dict)
    }
    observations: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        owner_ids = _receipt_owned_module_ids(receipt)
        if not owner_ids:
            if len(member_by_id) == 1:
                owner_ids = tuple(member_by_id)
            elif tracked_ids:
                raise CompleteProductionError(
                    "SEMANTIC_RECEIPT_OWNERSHIP_MISSING: "
                    "multi-module generation receipt did not declare module ownership"
                )
            else:
                continue
        foreign_ids = sorted(set(owner_ids) - member_ids)
        if foreign_ids:
            raise CompleteProductionError(
                "SEMANTIC_RECEIPT_OWNER_OUTSIDE_NODE: " + ", ".join(foreign_ids)
            )
        for module_id in owner_ids:
            module = member_by_id[module_id]
            observation = _semantic_execution_observation(
                module,
                receipt,
                dependent_ids=downstream_ids(module_id),
            )
            if observation is not None and evidence_context is not None:
                from .evidence_first_pipeline_contract import (
                    enrich_execution_observation,
                )

                observation = enrich_execution_observation(
                    observation,
                    evidence_context,
                )
            if observation is not None:
                observations.append(observation)
                observed_ids.add(module_id)
    missing_ids = sorted(tracked_ids - observed_ids)
    if missing_ids:
        raise CompleteProductionError(
            "SEMANTIC_RECEIPT_COVERAGE_MISSING: " + ", ".join(missing_ids)
        )
    return observations

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
    build_bundle_zip: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unchanged_postbuild_validation(
    source_report: dict[str, Any],
    jdt_receipt: dict[str, Any] | None,
    validate_jdt: Callable[[], dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    needs_postbuild_jdt = jdt_receipt is None
    if jdt_receipt is not None:
        normalized, _path = unwrap_diagnostic_receipt(jdt_receipt)
        needs_postbuild_jdt = (
            str(normalized.get("status") or "").strip().upper()
            == "DEFERRED_TO_POST_BUILD"
        )
    if validate_jdt is not None and needs_postbuild_jdt:
        return source_report, validate_jdt(), True
    return source_report, jdt_receipt, False


_JDT_INFRASTRUCTURE_CODES = frozenset({
    "JDT_DIAGNOSTICS_UNAVAILABLE",
    "JDT_WORKSPACE_NOT_READY",
})


def _runtime_visual_download_artifacts(
    visual_receipt: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(visual_receipt, dict):
        return {}
    screenshots = visual_receipt.get("runtime_screenshots")
    if not isinstance(screenshots, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(screenshots, start=1):
        if not isinstance(item, dict):
            continue
        raw = item.get("evidence_path")
        digest = item.get("sha256")
        if not isinstance(raw, str) or not isinstance(digest, str):
            continue
        path = Path(raw).expanduser().resolve()
        suffix = path.suffix.lower()
        result[f"runtime-screenshot-{index:03d}{suffix}"] = {
            "path": str(path),
            "sha256": digest,
        }
    return result


def _stable_payload_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _generation_receipt_sort_key(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        return ("", "", _stable_payload_sha256(value))
    owner = next(
        (
            str(value.get(key) or "")
            for key in ("module_id", "entity_id", "pack_id", "sound_id")
            if value.get(key)
        ),
        "",
    )
    return (
        owner,
        str(value.get("schema_version") or ""),
        _stable_payload_sha256(value),
    )


def _replace_stale_file_target(
    target: Path,
    action: Callable[[], Any],
) -> Any:
    path = target.expanduser().resolve()
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise CompleteProductionError(
                f"Stale package target is not a regular file: {path}"
            )
        path.unlink()
    return action()


def _attach_verified_release_artifact(
    release_result: dict[str, Any],
    descriptor: dict[str, Any] | None,
    *,
    archive_name: str,
    allowed_root: Path,
    manifest_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    if descriptor is None and not manifest_provenance:
        return dict(release_result)
    if (
        not isinstance(release_result, dict)
        or release_result.get('status') != 'PACKAGED'
        or not isinstance(release_result.get('release_zip'), str)
        or not isinstance(release_result.get('sha256'), str)
    ):
        raise CompleteProductionError('Release package receipt is invalid before attachment.')
    if descriptor is not None and (
        not archive_name or Path(archive_name).name != archive_name
    ):
        raise CompleteProductionError('Release attachment name is unsafe.')

    root = allowed_root.expanduser().resolve()
    release_path = Path(str(release_result['release_zip'])).expanduser().resolve()
    source = (
        Path(str(descriptor.get('path') or '')).expanduser().resolve()
        if descriptor is not None
        else None
    )
    candidates = [(release_path, 'release ZIP')]
    if source is not None:
        candidates.append((source, 'release attachment'))
    for candidate, label in candidates:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise CompleteProductionError(
                f'{label} escaped the run workspace: {candidate}'
            ) from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise CompleteProductionError(f'{label} is missing or unsafe: {candidate}')

    if CompleteProductionOrchestrator._file_hash(release_path) != release_result['sha256']:
        raise CompleteProductionError('Release ZIP changed before verified attachment.')
    expected = str(descriptor.get('sha256') or '') if descriptor is not None else ''
    if source is not None and (
        not expected or CompleteProductionOrchestrator._file_hash(source) != expected
    ):
        raise CompleteProductionError('Release attachment digest mismatch.')

    temp = release_path.with_name('.' + release_path.name + '.rewrite.tmp')
    if temp.exists():
        if temp.is_symlink() or not temp.is_file():
            raise CompleteProductionError('Release rewrite temporary target is unsafe.')
        temp.unlink()
    member_name = 'additional/' + archive_name if descriptor is not None else ''
    try:
        with zipfile.ZipFile(release_path, 'r') as original:
            infos = original.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise CompleteProductionError('Release ZIP contains duplicate member names.')
            if 'release-manifest.json' not in names:
                raise CompleteProductionError('Release ZIP has no release manifest.')
            if descriptor is not None and member_name in names:
                raise CompleteProductionError(
                    f'Release ZIP already contains attachment member: {member_name}'
                )
            try:
                manifest = json.loads(
                    original.read('release-manifest.json').decode('utf-8')
                )
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
                raise CompleteProductionError(
                    'Release manifest is unreadable before attachment.'
                ) from exc
            if not isinstance(manifest, dict):
                raise CompleteProductionError('Release manifest is not a JSON object.')
            if manifest_provenance:
                for key, value in manifest_provenance.items():
                    if not isinstance(key, str) or not key or not isinstance(value, str):
                        raise CompleteProductionError(
                            'Release manifest provenance must contain non-empty string keys and string values.'
                        )
                    manifest[key] = value

            if descriptor is not None:
                additional = manifest.get('additional_artifacts')
                if additional is None:
                    additional_map: dict[str, str] = {}
                elif isinstance(additional, dict):
                    additional_map = {
                        str(key): str(value) for key, value in additional.items()
                    }
                else:
                    raise CompleteProductionError(
                        'Release manifest additional_artifacts is invalid.'
                    )
                additional_map[archive_name] = expected
                manifest['additional_artifacts'] = dict(sorted(additional_map.items()))

            with zipfile.ZipFile(temp, 'w') as rewritten:
                manifest_info: zipfile.ZipInfo | None = None
                for info in infos:
                    if info.filename == 'release-manifest.json':
                        manifest_info = info
                        continue
                    rewritten.writestr(info, original.read(info.filename))
                if source is not None:
                    attachment_info = zipfile.ZipInfo(
                        member_name,
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    attachment_info.compress_type = zipfile.ZIP_DEFLATED
                    attachment_info.external_attr = (0o644 & 0xFFFF) << 16
                    rewritten.writestr(attachment_info, source.read_bytes())
                if manifest_info is None:
                    raise CompleteProductionError('Release manifest metadata disappeared.')
                rewritten.writestr(
                    manifest_info,
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(',', ':'),
                    ).encode('utf-8'),
                )
        os.replace(temp, release_path)
    finally:
        if temp.exists():
            temp.unlink()

    updated = dict(release_result)
    updated['sha256'] = CompleteProductionOrchestrator._file_hash(release_path)
    if descriptor is not None:
        updated['additional_artifacts'] = dict(
            sorted(
                {
                    **(
                        release_result.get('additional_artifacts')
                        if isinstance(release_result.get('additional_artifacts'), dict)
                        else {}
                    ),
                    archive_name: expected,
                }.items()
            )
        )
    if manifest_provenance:
        updated['manifest_provenance'] = dict(manifest_provenance)
    return updated


def _replace_stale_directory_target(
    target: Path,
    action: Callable[[], Any],
) -> Any:
    path = target.expanduser().resolve()
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise CompleteProductionError(
                f"Stale package target is not a regular directory: {path}"
            )
        shutil.rmtree(path)
    return action()


def _blocking_jdt_errors(
    receipt: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return only source diagnostics that remain actionable after build verification."""

    return [
        item
        for item in jdt_diagnostic_errors(receipt)
        if str(item.get("code") or "").strip().upper() not in _JDT_INFRASTRUCTURE_CODES
    ]


def _jdt_release_evidence_passed(receipt: dict[str, Any] | None) -> bool:
    """Require one real, clean JDT receipt for release authority."""

    # Gate evaluation probes optional evidence too. Absence is not a malformed
    # tool response; callers still enforce any explicitly requested JDT gate.
    if receipt is None:
        return False
    normalized, _path = unwrap_diagnostic_receipt(receipt)
    if not normalized:
        return False
    if jdt_diagnostic_errors(receipt):
        return False
    status = str(normalized.get("status") or "").strip().upper()
    if status == "DEFERRED_TO_POST_BUILD":
        return False
    try:
        error_count = int(normalized.get("error_count", -1))
        files_opened = int(normalized.get("files_opened", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    return error_count == 0 and files_opened > 0


def _requested_verification_failures(
    *,
    run_jdt: bool,
    jdt_receipt: dict[str, Any] | None,
) -> list[str]:
    """Keep explicitly requested verifier work release-blocking when it is unavailable."""

    if run_jdt and not _jdt_release_evidence_passed(jdt_receipt):
        return ["execution-gate:jdt:missing-jdt"]
    return []


def _validate_required_gate_contract(proposal: Any) -> None:
    unsupported: list[str] = []
    for module in getattr(proposal, "modules", ()):
        module_id = str(getattr(module, "module_id", "") or "")
        for gate in getattr(module, "required_gates", ()):
            rendered = str(gate).strip()
            if not rendered:
                continue
            if _normalize_required_gate(rendered) not in _REQUIRED_GATE_TO_EVIDENCE:
                unsupported.append(f"{module_id}:{rendered}")
    if unsupported:
        raise CompleteProductionError(
            "Approved proposal contains unsupported required gates: "
            + ", ".join(sorted(unsupported))
        )


def _validate_external_execution_preflight(
    proposal: Any,
    options: Any,
) -> None:
    if bool(getattr(options, "source_only", False)):
        return

    required = bool(getattr(proposal, "external_runtime_required", False))
    if required:
        disabled = [
            name
            for name, enabled in (
                ("runtime", getattr(options, "run_runtime", False)),
                ("client", getattr(options, "run_client", False)),
                ("mineflayer", getattr(options, "run_mineflayer", False)),
                ("visual-review", getattr(options, "run_visual_review", False)),
            )
            if not enabled
        ]
        if disabled:
            raise CompleteProductionError(
                "Approved proposal requires external runtime verification, but these "
                "checks are disabled: " + ", ".join(disabled)
            )

    if bool(getattr(options, "run_client", False)) and not bool(
        getattr(options, "run_runtime", False)
    ):
        raise CompleteProductionError(
            "Client verification requires runtime verification."
        )

    has_entity_review = any(
        getattr(module, "kind", "") in {"entity", "boss", "npc"}
        for module in getattr(proposal, "modules", ())
    )
    if (
        has_entity_review
        and not bool(getattr(options, "source_only", False))
        and not bool(getattr(options, "run_blockbench", False))
    ):
        raise CompleteProductionError(
            "Entity production requires Blockbench UV/render verification."
        )

    if bool(getattr(options, "run_runtime", False)):
        if not bool(getattr(options, "eula_accepted", False)):
            raise CompleteProductionError(
                "Runtime verification was requested without explicit Minecraft EULA acceptance."
            )
        raw_launcher = getattr(options, "server_launcher", None)
        if not isinstance(raw_launcher, str) or not raw_launcher.strip():
            raise CompleteProductionError(
                "Runtime verification requires server_launcher before generation starts."
            )
        launcher = Path(raw_launcher).expanduser().resolve()
        if not launcher.is_file() or launcher.is_symlink():
            raise CompleteProductionError(
                "server_launcher must be an existing regular file before generation starts."
            )

    if bool(getattr(options, "run_mineflayer", False)):
        if not bool(getattr(options, "run_runtime", False)):
            raise CompleteProductionError(
                "Mineflayer verification requires runtime verification."
            )
        actions = getattr(options, "playtest_actions", ())
        if not isinstance(actions, (list, tuple)) or not actions:
            raise CompleteProductionError(
                "Mineflayer verification requires explicit playtest_actions before generation starts."
            )
        expected_tests = tuple(
            str(value) for value in getattr(proposal, "acceptance_tests", ())
        )
        if expected_tests:
            expected_set = set(expected_tests)
            covered: set[str] = set()
            unknown: set[str] = set()
            for action in actions:
                if not isinstance(action, dict):
                    continue
                if str(action.get("action") or "") != "wait_for":
                    continue
                raw_test = action.get("acceptance_test")
                if raw_test is None:
                    continue
                test = str(raw_test)
                if test in expected_set:
                    covered.add(test)
                else:
                    unknown.add(test)
            missing = [test for test in expected_tests if test not in covered]
            if missing or unknown:
                details: list[str] = []
                if missing:
                    details.append("missing=" + ", ".join(missing))
                if unknown:
                    details.append("unknown=" + ", ".join(sorted(unknown)))
                raise CompleteProductionError(
                    "Mineflayer playtest actions do not match the approved acceptance tests: "
                    + "; ".join(details)
                )

    if bool(getattr(options, "run_visual_review", False)) and not bool(
        getattr(options, "run_runtime", False)
    ):
        raise CompleteProductionError(
            "Visual verification requires the disposable runtime."
        )


def _collect_runtime_screenshot_receipts(
    runtime_manager: MinecraftRuntimeManager,
    explicit_paths: Iterable[str],
    *,
    evidence_root: Path | None = None,
) -> list[dict[str, Any]]:
    status = runtime_manager.status()
    instance_raw = status.get("instance_root")
    if (
        not isinstance(instance_raw, str)
        or status.get("server_running") is not True
        or status.get("client_running") is not True
    ):
        raise CompleteProductionError(
            "Runtime visual evidence requires a live server and client."
        )
    client_root = (Path(instance_raw).expanduser().resolve() / "client").resolve()
    explicit = tuple(str(value) for value in explicit_paths if str(value).strip())
    if explicit:
        candidates = [Path(value).expanduser().resolve() for value in explicit]
    else:
        screenshots_root = client_root / "screenshots"
        candidates = (
            [
                path.resolve()
                for path in sorted(screenshots_root.rglob("*"))
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if screenshots_root.is_dir()
            else []
        )
    if not candidates:
        raise CompleteProductionError(
            "No screenshots were produced by the current disposable runtime client."
        )
    receipts: list[dict[str, Any]] = []
    for path in candidates:
        try:
            path.relative_to(client_root)
        except ValueError as exc:
            raise CompleteProductionError(
                "Visual evidence must come from the current disposable client directory."
            ) from exc
        receipt = runtime_manager.register_screenshot(path)
        if (
            receipt.get("server_running") is not True
            or receipt.get("client_running") is not True
            or not isinstance(receipt.get("sha256"), str)
        ):
            raise CompleteProductionError(
                "Runtime screenshot receipt is not bound to a live client session."
            )
        expected_sha = str(receipt["sha256"])
        digest = expected_sha.removeprefix("sha256:")
        preserved_root = (
            evidence_root.expanduser().resolve()
            if evidence_root is not None
            else (
                Path(runtime_manager.workspace_root).expanduser().resolve()
                / "integration-evidence"
                / "runtime-screenshots"
            )
        )
        preserved_root.mkdir(parents=True, exist_ok=True)
        evidence_path = preserved_root / (digest + path.suffix.lower())
        if evidence_path.exists():
            if (
                evidence_path.is_symlink()
                or not evidence_path.is_file()
                or CompleteProductionOrchestrator._file_hash(evidence_path)
                != expected_sha
            ):
                raise CompleteProductionError(
                    "Existing runtime screenshot evidence does not match its digest."
                )
        else:
            shutil.copy2(path, evidence_path)
            if CompleteProductionOrchestrator._file_hash(evidence_path) != expected_sha:
                evidence_path.unlink(missing_ok=True)
                raise CompleteProductionError(
                    "Runtime screenshot changed while preserving visual evidence."
                )
        receipts.append(
            {
                **receipt,
                "runtime_source_path": str(path),
                "path": str(evidence_path),
                "evidence_path": str(evidence_path),
            }
        )
    return receipts


def _visual_runtime_evidence_passed(
    visual_receipt: dict[str, Any] | None,
    runtime_receipt: dict[str, Any] | None,
) -> bool:
    if (
        not isinstance(visual_receipt, dict)
        or visual_receipt.get("status") != "PASS"
        or not isinstance(runtime_receipt, dict)
    ):
        return False
    artifact_sha = runtime_receipt.get("artifact_sha256")
    if (
        not isinstance(artifact_sha, str)
        or visual_receipt.get("artifact_sha256") != artifact_sha
    ):
        return False
    screenshots = visual_receipt.get("runtime_screenshots")
    return (
        isinstance(screenshots, list)
        and bool(screenshots)
        and all(
            isinstance(item, dict)
            and item.get("server_running") is True
            and item.get("client_running") is True
            and isinstance(item.get("sha256"), str)
            and bool(item.get("sha256"))
            and isinstance(item.get("evidence_path"), str)
            and Path(str(item["evidence_path"])).is_file()
            and not Path(str(item["evidence_path"])).is_symlink()
            and CompleteProductionOrchestrator._file_hash(
                Path(str(item["evidence_path"]))
            )
            == item.get("sha256")
            for item in screenshots
        )
    )


def _persisted_runtime_evidence(
    runtime_receipt: dict[str, Any] | None,
    *,
    required: bool,
    artifact_sha256: str,
) -> dict[str, Any]:
    if isinstance(runtime_receipt, dict):
        return dict(runtime_receipt)
    return {
        "schema_version": "mmm/final-runtime-receipt-v1",
        "status": "REQUIRED_NOT_RUN" if required else "NOT_REQUIRED",
        "artifact_sha256": artifact_sha256,
    }


def _refresh_runtime_receipt_status(
    receipt: dict[str, Any],
    live_status: dict[str, Any],
    *,
    require_client: bool,
) -> dict[str, Any]:
    server = dict(receipt.get("server") or {})
    server["server_running"] = live_status.get("server_running") is True
    if "server_log_lines" in live_status:
        server["server_log_lines"] = live_status.get("server_log_lines")

    client_value = receipt.get("client")
    client = dict(client_value) if isinstance(client_value, dict) else None
    if client is not None:
        client["client_running"] = live_status.get("client_running") is True
        if "client_log_lines" in live_status:
            client["client_log_lines"] = live_status.get("client_log_lines")

    terminal_ok = server.get("server_running") is True and (
        not require_client
        or (client is not None and client.get("client_running") is True)
    )
    return {
        **receipt,
        "status": "PASS" if terminal_ok else "FAIL",
        "server": server,
        "client": client,
        "final_status": dict(live_status),
    }


def _playtest_evidence_passed(
    playtest_receipt: dict[str, Any] | None,
    expected_acceptance_tests: Iterable[str] = (),
) -> bool:
    if (
        not isinstance(playtest_receipt, dict)
        or playtest_receipt.get("status") != "PASS"
        or int(playtest_receipt.get("interaction_count", 0)) <= 0
        or int(playtest_receipt.get("assertion_count", 0)) <= 0
    ):
        return False
    expected = tuple(str(value) for value in expected_acceptance_tests)
    if not expected:
        return True
    return (
        playtest_receipt.get("acceptance_tests") == list(expected)
        and playtest_receipt.get("covered_acceptance_tests") == list(expected)
        and isinstance(playtest_receipt.get("acceptance_test_results"), list)
        and {
            str(item.get("test"))
            for item in playtest_receipt["acceptance_test_results"]
            if isinstance(item, dict) and item.get("status") == "PASS"
        }
        >= set(expected)
    )


def _runtime_verification_passed(
    *,
    required: bool,
    runtime_receipt: dict[str, Any] | None,
    playtest_receipt: dict[str, Any] | None,
    visual_receipt: dict[str, Any] | None,
    expected_acceptance_tests: Iterable[str] = (),
) -> bool:
    if not required:
        return True
    if not isinstance(runtime_receipt, dict) or runtime_receipt.get("status") != "PASS":
        return False
    server = runtime_receipt.get("server")
    client = runtime_receipt.get("client")
    if not (
        isinstance(server, dict)
        and server.get("server_running") is True
        and isinstance(client, dict)
        and client.get("client_running") is True
    ):
        return False
    if (
        not isinstance(playtest_receipt, dict)
        or playtest_receipt.get("artifact_sha256")
        != runtime_receipt.get("artifact_sha256")
    ):
        return False
    prepared = runtime_receipt.get("prepared")
    if (
        isinstance(prepared, dict)
        and isinstance(prepared.get("instance_root"), str)
        and playtest_receipt.get("runtime_instance_root") != prepared.get("instance_root")
    ):
        return False
    if not _playtest_evidence_passed(
        playtest_receipt,
        expected_acceptance_tests,
    ):
        return False
    return _visual_runtime_evidence_passed(visual_receipt, runtime_receipt)


def _final_validation_failure(
    *,
    source_report: dict[str, Any],
    jdt_receipt: dict[str, Any] | None,
    run_jdt: bool,
) -> str | None:
    if source_report.get("status") != "PASS":
        return "Final project failed deterministic source validation after build/repair."
    if run_jdt and jdt_receipt is not None and _blocking_jdt_errors(jdt_receipt):
        return "Final JDT validation still reports source errors after build/repair."
    return None


def _refresh_validation_after_build(
    *,
    prebuild_manifest: str,
    final_manifest: str,
    source_report: dict[str, Any],
    jdt_receipt: dict[str, Any] | None,
    validate_source: Callable[[], dict[str, Any]],
    validate_jdt: Callable[[], dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Run JDT after compile; refresh source validation only when the tree changed."""
    if final_manifest == prebuild_manifest:
        return _unchanged_postbuild_validation(
            source_report, jdt_receipt, validate_jdt
        )

    from .deadline_executor import iter_completed_with_deadlines

    validation_jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("source", validate_source),
    ]
    if validate_jdt is not None:
        validation_jobs.append(("jdt", validate_jdt))

    validation_results: dict[str, dict[str, Any]] = {}
    for job, result in iter_completed_with_deadlines(
        validation_jobs,
        lambda item: item[1](),
        max_workers=len(validation_jobs),
        stage="complete_post_build_validation",
        sort_key=lambda item: item[0],
    ):
        validation_results[job[0]] = result

    refreshed_source = validation_results["source"]
    if refreshed_source.get("status") != "PASS":
        raise CompleteProductionError(
            "Final repaired project failed deterministic validation."
        )
    refreshed_jdt = validation_results.get("jdt")
    return refreshed_source, refreshed_jdt, True

class CompleteProductionOrchestrator:
    """Approved request -> sharded source -> repair -> runtime -> release."""

    def __init__(self, *, workspace_root: str | Path='mmm-output', profile: str='t4_local', router_factory: Callable[[], ModelRouter] | None=None, policy: ScalePolicy | None=None) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.router_factory = router_factory or (lambda: ModelRouter(profile=profile))
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    @execution_feedback_scoped
    @execution_scoped
    def execute(self, proposal: CompleteProposal | dict[str, Any], *, approval_hash: str, run_name: str, options: CompleteExecutionOptions | None=None, existing_input: str | Path | None=None) -> CompletePipelineResult:
        options = options or CompleteExecutionOptions()
        options.validate(policy=self.policy)
        parsed = proposal if isinstance(proposal, CompleteProposal) else CompleteProposal.from_dict(proposal)
        parsed.validate(policy=self.policy)
        approved = parsed.approve(approval_hash)
        if approved.status is not CompleteProposalStatus.APPROVED:
            raise SpecValidationError('Complete proposal approval did not complete.')
        _validate_external_execution_preflight(approved, options)
        _validate_required_gate_contract(approved)
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
        debug_source_acceptance: dict[str, Any] | None = None
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
        validation_manifest = self._project_manifest_hash(project_root)

        def validate_source() -> dict[str, Any]:
            return run_named_checkpoint(ledger, 'validate-source', stage='validate:source', input_value=validation_checkpoint_input('validate-source', {'graph_hash': work_plan.graph_hash, 'project_manifest': validation_manifest}), action=lambda: ScalableProjectValidator(policy=self.policy).validate(project_root, spec).to_dict(), encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: cached_validation_is_reusable('validate-source', cached))

        def validate_jdt() -> dict[str, Any]:
            def run_jdt() -> dict[str, Any]:
                return _run_release_jdt_verification(project_root)
            return run_named_checkpoint(ledger, 'validate-jdt', stage='validate:jdt', input_value=validation_checkpoint_input('validate-jdt', {'graph_hash': work_plan.graph_hash, 'project_manifest': validation_manifest}), action=run_jdt, encode=lambda value: value, decode=lambda cached: cached, validate_cached=lambda cached: cached_validation_is_reusable('validate-jdt', cached))

        jdt_receipt = None
        if options.run_jdt and options.source_only:
            from .deadline_executor import iter_completed_with_deadlines

            validation_jobs = (("source", validate_source), ("jdt", validate_jdt))
            validation_results: dict[str, dict[str, Any]] = {}
            for job, result in iter_completed_with_deadlines(
                validation_jobs,
                lambda item: item[1](),
                max_workers=2,
                stage="complete_validation",
                sort_key=lambda item: item[0],
            ):
                validation_results[job[0]] = result
            source_report = validation_results["source"]
            jdt_receipt = validation_results["jdt"]
        else:
            source_report = validate_source()

        if source_report.get('status') != 'PASS':
            raise CompleteProductionError('Generated complete project failed deterministic validation.')
        if (
            approved.schema_version == 'mmm/complete-proposal-v1'
            and approved.game_design.get('mode') == 'debug_fixture'
        ):
            fixture = approved.game_design.get('fixture')
            fixture_module_id = (
                str(fixture.get('module_id') or '').strip()
                if isinstance(fixture, dict)
                else ''
            )
            fixture_module = next(
                (
                    module
                    for module in approved.modules
                    if module.module_id == fixture_module_id
                ),
                None,
            )
            source_contract = (
                fixture_module.config.get('observable_source_contract')
                if fixture_module is not None and isinstance(fixture_module.config, dict)
                else None
            )
            normalized_source_contract = (
                source_contract if isinstance(source_contract, dict) else None
            )
            host_facts_json = approved.base_proposal.spec.platform.host_facts_json
            debug_source_acceptance = run_named_checkpoint(
                ledger,
                'validate-debug-source',
                stage='validate:debug-source',
                input_value={
                    'graph_hash': work_plan.graph_hash,
                    'project_manifest': validation_manifest,
                    'source_contract_sha256': _stable_payload_sha256(
                        normalized_source_contract or {}
                    ),
                    'host_facts_sha256': 'sha256:' + hashlib.sha256(
                        host_facts_json.encode('utf-8')
                    ).hexdigest(),
                },
                action=lambda: verify_debug_fixture_source(
                    project_root,
                    source_contract=normalized_source_contract,
                    host_facts_json=host_facts_json,
                ),
                encode=lambda value: value,
                decode=lambda cached: cached,
                # Always re-evaluate after generation replay. The input hash is still
                # persisted for audit, but a stale passing source contract must never
                # survive a source mutation.
                validate_cached=lambda _cached: False,
            )
            module_receipts.append(
                {
                    'schema_version': 'mmm/debug-source-gate-v1',
                    **debug_source_acceptance,
                }
            )
            if debug_source_acceptance.get('status') != 'PASS':
                findings = debug_source_acceptance.get('findings')
                rendered = (
                    '; '.join(str(value) for value in findings)
                    if isinstance(findings, list) and findings
                    else 'observable source contract was not proven'
                )
                raise CompleteProductionError(
                    'Debug fixture observable source acceptance failed: ' + rendered
                )
        self._succeed_work_node(ledger, 'validate-source', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'PASS', 'checks_run': source_report.get('checks_run', 0), 'project_manifest': generated_manifest_hash})
        self._persist_work_evidence(project_root, ledger, work_plan)
        if jdt_receipt is not None:
            module_receipts.append({'schema_version': 'mmm/jdt-gate-v1', **jdt_receipt})
            print('[JDT RECEIPT] ' + json.dumps(jdt_receipt, ensure_ascii=False, sort_keys=True, default=str), flush=True)
            # Infrastructure-unavailable JDT remains auxiliary, but real source diagnostics
            # cannot be packaged in source-only mode and must be repaired before a full build exits.
            errors = _blocking_jdt_errors(jdt_receipt)
            if errors and (options.source_only or not options.auto_repair):
                raise CompleteProductionError(
                    'JDT reported source errors that cannot be left unresolved.'
                )
        if options.source_only:
            source_package = run_named_checkpoint(
                ledger,
                'package-source',
                stage='package:source',
                input_value={
                    'graph_hash': work_plan.graph_hash,
                    'project_manifest': self._project_manifest_hash(project_root),
                },
                action=lambda: self._source_package_receipt(
                    self._package_source_only(run_root, project_root, approved)
                ),
                encode=lambda value: value,
                decode=lambda cached: cached,
                validate_cached=lambda cached: self._cached_package_exists(
                    cached, path_key='release_zip'
                ),
            )
            release = str(source_package['release_zip'])
            unresolved.extend(_external_gates(approved, options))
            unresolved.extend(
                _requested_verification_failures(
                    run_jdt=options.run_jdt,
                    jdt_receipt=jdt_receipt,
                )
            )
            quality_report = self._evaluate_quality(approved=approved, run_root=run_root, project_root=project_root, source_validation=source_report, build_report=None, jar_validation=None, module_receipts=module_receipts, asset_receipt=asset_receipt, blockbench_receipts=blockbench_receipts, runtime_receipt=None, playtest_receipt=None, visual_receipt=None)
            if quality_report is not None:
                unresolved.extend(f'quality:{dimension_id}' for dimension_id in quality_unresolved(quality_report))
                self._record_quality_nodes(ledger, quality_report, allow_success=False)
            unresolved.extend(
                self._required_gate_failures(
                    approved,
                    generated_receipts=module_receipts,
                    project_root=project_root,
                    source_validation=source_report,
                    jdt_receipt=jdt_receipt,
                    build_report=None,
                    jar_validation=None,
                    blockbench_receipts=blockbench_receipts,
                    runtime_receipt=None,
                    playtest_receipt=None,
                    visual_receipt=None,
                )
            )
            self._persist_work_evidence(project_root, ledger, work_plan)
            return CompletePipelineResult(schema_version='mmm/complete-pipeline-result-v3', status='SOURCE_READY', project_root=str(project_root), release_zip=release, jar_path=None, complete_proposal_hash=approved.calculate_hash(), source_validation=source_report, build_report=None, jar_validation=None, module_receipts=tuple(module_receipts), asset_receipt=asset_receipt, blockbench_receipts=tuple(blockbench_receipts), runtime_receipt=None, playtest_receipt=None, visual_receipt=None, distribution_receipt=None, unresolved_gates=tuple(sorted(set(unresolved))), release_ready=False, work_graph_hash=work_plan.graph_hash, work_ledger_path=str(ledger.path), run_resumed=run_resumed, quality_report=quality_report)
        build_bundle, router = run_build_repair_checkpoint(
            ledger=ledger, graph_hash=work_plan.graph_hash,
            validation_manifest=validation_manifest, project_root=project_root,
            run_root=run_root, options=options, router=router,
            router_factory=self.router_factory, policy=self.policy,
            validate_cached=lambda cached: self._cached_build_exists(
                cached,
                require_gametest=options.run_gametest,
                spec=spec,
            ),
        )
        build = build_bundle['build']
        repair = build_bundle.get('repair')
        if isinstance(repair, dict):
            module_receipts.append({'schema_version': 'mmm/repair-receipt-v2', **repair})
            if repair.get('patch_receipts'):
                update_execution_project_index_from_receipt(project_root, repair)
        if build.get('status') != 'PASS':
            raise CompleteProductionError('Gradle/GameTest failed after the repair loop.')

        final_manifest = str(
            execution_project_index(ProjectIndex, project_root, policy=self.policy)
            .manifest_receipt()['sha256']
        )

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
        if not self._full_gradle_build_receipt_passed(build):
            raise CompleteProductionError(
                'Final project has no passing full Gradle build command receipt.'
            )
        build = dict(build)
        build['artifact_receipt'] = artifact_receipt
        gametest_attestation = _gametest_attestation_status(
            build,
            spec,
            requested=options.run_gametest,
        )
        if options.run_gametest and gametest_attestation != 'PASS':
            raise CompleteProductionError(
                'GameTest was requested but no passing structured GameTest evidence was produced.'
            )
        build_receipt = {
            'schema_version': 'mmm/final-build-receipt-v1',
            'status': 'PASS',
            'toolchain_attested': True,
            'compile_java': 'PASS',
            'tests': 'PASS',
            'gradle_build': 'PASS',
            'gametest': gametest_attestation,
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

        def validate_final_source() -> dict[str, Any]:
            return run_named_checkpoint(
                ledger,
                'validate-source-final',
                stage='validate:source-final',
                input_value=validation_checkpoint_input(
                    'validate-source-final',
                    {'graph_hash': work_plan.graph_hash, 'project_manifest': final_manifest},
                ),
                action=lambda: ScalableProjectValidator(policy=self.policy).validate(project_root, spec).to_dict(),
                encode=lambda value: value,
                decode=lambda cached: cached,
                validate_cached=lambda cached: cached_validation_is_reusable('validate-source-final', cached),
            )

        def validate_final_jdt() -> dict[str, Any]:
            return run_named_checkpoint(
                ledger,
                'validate-jdt-final',
                stage='validate:jdt-final',
                input_value=validation_checkpoint_input(
                    'validate-jdt-final',
                    {'graph_hash': work_plan.graph_hash, 'project_manifest': final_manifest},
                ),
                action=lambda: _run_release_jdt_verification(project_root),
                encode=lambda value: value,
                decode=lambda cached: cached,
                validate_cached=lambda cached: cached_validation_is_reusable('validate-jdt-final', cached),
            )

        source_report, final_jdt_receipt, validation_refreshed = _refresh_validation_after_build(
            prebuild_manifest=validation_manifest,
            final_manifest=final_manifest,
            source_report=source_report,
            jdt_receipt=jdt_receipt,
            validate_source=validate_final_source,
            validate_jdt=(validate_final_jdt if options.run_jdt else None),
        )
        if validation_refreshed:
            jdt_receipt = final_jdt_receipt
            if jdt_receipt is not None:
                module_receipts.append(
                    {'schema_version': 'mmm/jdt-gate-v1', 'phase': 'final', **jdt_receipt}
                )
        final_validation_failure = _final_validation_failure(
            source_report=source_report,
            jdt_receipt=jdt_receipt,
            run_jdt=options.run_jdt,
        )
        if final_validation_failure is not None:
            raise CompleteProductionError(final_validation_failure)
        self._succeed_work_node(
            ledger,
            'validate-source-final',
            {
                'schema_version': 'mmm/work-node-receipt-v1',
                'status': 'PASS',
                'checks_run': source_report.get('checks_run', 0),
                'project_manifest': final_manifest,
                'refreshed_after_build': validation_refreshed,
            },
        )
        jar_validation = run_named_checkpoint(
            ledger,
            'validate-jar',
            stage='validate:jar',
            input_value=validation_checkpoint_input(
                'validate-jar',
                {
                    'graph_hash': work_plan.graph_hash,
                    'jar_sha256': self._file_hash(jar_path),
                },
            ),
            action=lambda: validate_jar(jar_path, spec).to_dict(),
            encode=lambda value: value,
            decode=lambda cached: cached,
            validate_cached=lambda cached: (
                jar_path.is_file()
                and cached_validation_is_reusable('validate-jar', cached)
            ),
        )
        if jar_validation.get('status') != 'PASS':
            raw_findings = jar_validation.get('findings')
            findings = (
                [item for item in raw_findings if isinstance(item, dict)]
                if isinstance(raw_findings, list)
                else []
            )
            compact_findings = [
                {
                    'code': str(item.get('code') or ''),
                    'path': str(item.get('path') or ''),
                    'message': str(item.get('message') or ''),
                    'severity': str(item.get('severity') or ''),
                }
                for item in findings[:8]
            ]
            emit_root_cause(
                'jar_validation_failed',
                stage='verify',
                operation='validate_jar',
                gate='independent_jar_validation',
                result='FAIL',
                reason='Built JAR failed independent validation.',
                details={
                    'status': jar_validation.get('status'),
                    'checks_run': jar_validation.get('checks_run'),
                    'finding_count': len(findings),
                    'findings': compact_findings,
                },
            )
            summary = ' | '.join(
                f"{item['code']}:{item['path']}:{item['message']}"
                for item in compact_findings
            )
            raise CompleteProductionError(
                'Built JAR failed independent validation.'
                + (f' {summary}' if summary else '')
            )
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
                if approved.external_runtime_required:
                    unresolved.append('runtime:not-requested')
            if options.run_mineflayer:
                if not options.run_runtime:
                    raise CompleteProductionError('Mineflayer requires the disposable runtime.')
                playtest_receipt = self._run_playtest(
                    options.playtest_actions,
                    approved.acceptance_tests,
                )
                current_runtime_status = (
                    runtime_manager.status() if runtime_manager is not None else {}
                )
                playtest_receipt = {
                    **playtest_receipt,
                    'artifact_sha256': str(artifact_receipt['sha256']),
                    'runtime_instance_root': current_runtime_status.get('instance_root'),
                }
            else:
                if approved.external_runtime_required:
                    unresolved.append('mineflayer:not-requested')
            if options.run_visual_review:
                if runtime_manager is None:
                    raise CompleteProductionError(
                        'Visual review requires the disposable runtime.'
                    )
                screenshot_receipts = _collect_runtime_screenshot_receipts(
                    runtime_manager,
                    options.screenshot_paths,
                    evidence_root=metadata_root / 'runtime-screenshots',
                )
                router = router or self.router_factory()
                visual_receipt = self._visual_review(
                    router,
                    approved,
                    tuple(str(item['path']) for item in screenshot_receipts),
                )
                visual_receipt = {
                    **visual_receipt,
                    'artifact_sha256': str(artifact_receipt['sha256']),
                    'runtime_screenshots': screenshot_receipts,
                }
                if visual_receipt.get('status') != 'PASS':
                    raise CompleteProductionError('VisualCritic rejected the runtime screenshots.')
            else:
                if approved.external_runtime_required:
                    unresolved.append('visual-review:not-requested')
            if runtime_manager is not None and runtime_receipt is not None:
                runtime_receipt = _refresh_runtime_receipt_status(
                    runtime_receipt,
                    runtime_manager.status(),
                    require_client=options.run_client,
                )
        finally:
            if runtime_manager is not None and options.cleanup_runtime:
                cleanup = runtime_manager.cleanup()
                runtime_receipt = {**(runtime_receipt or {}), 'cleanup': cleanup}
        runtime_verified = _runtime_verification_passed(
            required=approved.external_runtime_required,
            runtime_receipt=runtime_receipt,
            playtest_receipt=playtest_receipt,
            visual_receipt=visual_receipt,
            expected_acceptance_tests=approved.acceptance_tests,
        )
        persisted_runtime_receipt = {
            **_persisted_runtime_evidence(
                runtime_receipt,
                required=approved.external_runtime_required,
                artifact_sha256=str(artifact_receipt['sha256']),
            ),
            'verification_status': (
                'PASS'
                if runtime_verified
                else ('NOT_REQUIRED' if not approved.external_runtime_required else 'FAIL')
            ),
            'playtest': playtest_receipt,
            'visual': visual_receipt,
        }
        (metadata_root / 'runtime-receipt.json').write_text(
            json.dumps(persisted_runtime_receipt, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        if runtime_verified:
            self._succeed_work_node(ledger, 'runtime-playtest', {'schema_version': 'mmm/work-node-receipt-v1', 'status': 'NOT_REQUIRED' if not approved.external_runtime_required else 'PASS', 'runtime': runtime_receipt, 'playtest': playtest_receipt, 'visual': visual_receipt})
        else:
            unresolved.append('runtime:verification-incomplete')
            ledger.fail('runtime-playtest', 'Runtime, interaction, and visual evidence are still required.', input_required=True)
        self._persist_work_evidence(project_root, ledger, work_plan)
        quality_report = self._evaluate_quality(approved=approved, run_root=run_root, project_root=project_root, source_validation=source_report, build_report=build, jar_validation=jar_validation, module_receipts=module_receipts, asset_receipt=asset_receipt, blockbench_receipts=blockbench_receipts, runtime_receipt=runtime_receipt, playtest_receipt=playtest_receipt, visual_receipt=visual_receipt)
        quality_passed = quality_report is None or quality_report.get('overall_status') == 'PASS'
        if quality_report is not None:
            unresolved.extend(f'quality:{dimension_id}' for dimension_id in quality_unresolved(quality_report))
            self._record_quality_nodes(ledger, quality_report, allow_success=True)
        unresolved.extend(
            _requested_verification_failures(
                run_jdt=options.run_jdt,
                jdt_receipt=jdt_receipt,
            )
        )
        unresolved.extend(
            self._required_gate_failures(
                approved,
                generated_receipts=module_receipts,
                project_root=project_root,
                source_validation=source_report,
                jdt_receipt=jdt_receipt,
                build_report=build,
                jar_validation=jar_validation,
                blockbench_receipts=blockbench_receipts,
                runtime_receipt=runtime_receipt,
                playtest_receipt=playtest_receipt,
                visual_receipt=visual_receipt,
            )
        )
        unresolved.extend(
            self._mandatory_blockbench_failures(
                approved,
                blockbench_receipts,
            )
        )
        unresolved.extend(
            self._mandatory_gametest_failures(
                build,
                spec,
            )
        )
        self._persist_work_evidence(project_root, ledger, work_plan)
        contract = approved.game_design.get('_production_contract')
        normalized_unresolved = tuple(sorted(set(unresolved)))
        if (
            approved.schema_version == 'mmm/complete-proposal-v1'
            and approved.game_design.get('mode') == 'debug_fixture'
        ):
            coverage_receipt = build_debug_fixture_coverage_receipt(
                proposal_hash=approved.calculate_hash(),
                acceptance_tests=approved.acceptance_tests,
                artifact_sha256=str(artifact_receipt['sha256']),
                source_validation=source_report,
                build_report=build,
                jar_validation=jar_validation,
                gametest_passed=self._gametest_receipt_passed(build, spec),
                observable_acceptance=debug_source_acceptance,
                unresolved_gates=normalized_unresolved,
            )
        else:
            coverage_receipt = build_requirement_coverage_receipt(
                contract=contract if isinstance(contract, dict) else None,
                proposal_hash=approved.calculate_hash(),
                quality_report=quality_report,
                artifact_sha256=str(artifact_receipt['sha256']),
                unresolved_gates=normalized_unresolved,
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
        release_ready = (
            not normalized_unresolved
            and quality_passed
            and coverage_receipt.get('status') == 'PASS'
        )
        emit_root_cause(
            'release_gate_evaluation',
            stage='verify',
            operation='evaluate_release_readiness',
            gate='release_readiness',
            result='PASS' if release_ready else 'FAIL',
            reason=(
                ''
                if release_ready
                else 'Required release evidence remains unresolved.'
            ),
            details={
                'unresolved_gates': list(normalized_unresolved),
                'quality_passed': quality_passed,
                'coverage_status': coverage_receipt.get('status'),
                'gametest_receipt_passed': self._gametest_receipt_passed(build, spec),
                'gametest_mode': build.get('gametest_mode') if isinstance(build, dict) else None,
                'gametest_task': build.get('gametest_task') if isinstance(build, dict) else None,
                'gametest_report': build.get('gametest_report') if isinstance(build, dict) else None,
            },
        )
        build_bundle_input = {
            'artifact_sha256': str(artifact_receipt.get('sha256') or ''),
            'build_receipt_sha256': _stable_payload_sha256(build_receipt),
            'source_validation_sha256': _stable_payload_sha256(source_report),
            'jdt_receipt_sha256': _stable_payload_sha256(jdt_receipt),
            'jar_validation_sha256': _stable_payload_sha256(jar_validation),
            'coverage_sha256': str(coverage_receipt.get('coverage_sha256') or ''),
            'unresolved_sha256': _stable_payload_sha256(list(normalized_unresolved)),
            'release_ready': release_ready,
        }
        build_bundle_sha256 = _stable_payload_sha256(build_bundle_input)
        build_bundle_output = (
            run_root
            / 'releases'
            / (
                'build-artifact-'
                + build_bundle_sha256.split(':', 1)[1][:16]
                + '.zip'
            )
        )
        build_bundle_receipt = run_named_checkpoint(
            ledger,
            'package-build-artifact',
            stage='package:build-artifact',
            input_value=build_bundle_input,
            action=lambda: _replace_stale_file_target(
                build_bundle_output,
                lambda: write_build_artifact_bundle(
                    build_bundle_output,
                    artifact_receipt=artifact_receipt,
                    build_receipt=build_receipt,
                    unresolved_gates=normalized_unresolved,
                    release_ready=release_ready,
                    proposal_hash=approved.calculate_hash(),
                    receipts={
                        'source-validation.json': source_report,
                        'jdt-receipt.json': jdt_receipt,
                        'jar-validation.json': jar_validation,
                        'requirement-coverage.json': coverage_receipt,
                        'quality-report.json': quality_report,
                    },
                ),
            ),
            encode=lambda value: value,
            decode=lambda cached: cached,
            validate_cached=lambda cached: self._cached_package_exists(
                cached, path_key='build_bundle_zip'
            ),
        )
        build_bundle_zip = str(build_bundle_receipt['build_bundle_zip'])
        self._succeed_work_node(
            ledger,
            'package-build-artifact',
            {
                'schema_version': 'mmm/work-node-receipt-v1',
                'status': 'PASS',
                'build_bundle_zip': build_bundle_zip,
                'release_ready': release_ready,
                'unresolved_gates': list(normalized_unresolved),
            },
        )
        # The release ZIP snapshots project-owned work evidence before packaging.
        # Persist the completed build-artifact node now so the embedded ledger has
        # exactly one expected pending node: package-release itself.
        self._persist_work_evidence(project_root, ledger, work_plan)

        if options.publish_provider and (not release_ready):
            raise CompleteProductionError(
                'Publishing is blocked because required verification gates remain unresolved.'
            )

        release_zip: str | None = None
        distribution_receipt: dict[str, Any] | None = None
        if release_ready:
            from .mcp_tools import MMMToolService
            tool_service = MMMToolService(workspace_root=run_root, profile=self.profile)
            resource_pack_bundle = (
                asset_receipt.get('resource_pack_bundle')
                if isinstance(asset_receipt, dict)
                and isinstance(asset_receipt.get('resource_pack_bundle'), dict)
                else None
            )
            release_package_input = {
                'graph_hash': work_plan.graph_hash,
                'proposal_hash': approved.calculate_hash(),
                'base_proposal_hash': base.calculate_hash(),
                'jar_sha256': self._file_hash(jar_path),
                'coverage_sha256': str(coverage_receipt.get('coverage_sha256') or ''),
                'runtime_receipt_sha256': _stable_payload_sha256(persisted_runtime_receipt),
                'build_receipt_sha256': _stable_payload_sha256(build_receipt),
                'reuse_manifest_sha256': _stable_payload_sha256(reuse_manifest),
                'quality_report_sha256': _stable_payload_sha256(quality_report),
                'resource_pack_sha256': str(
                    resource_pack_bundle.get('sha256') if resource_pack_bundle else ''
                ),
                'visual_evidence_sha256': _stable_payload_sha256(
                    _runtime_visual_download_artifacts(visual_receipt)
                ),
            }
            release_package_sha256 = _stable_payload_sha256(release_package_input)
            release_output = (
                'releases/complete-release-'
                + release_package_sha256.split(':', 1)[1][:16]
                + '.zip'
            )
            release_result = run_named_checkpoint(
                ledger,
                'package-release',
                stage='package',
                input_value=release_package_input,
                action=lambda: _replace_stale_file_target(
                    run_root / release_output,
                    lambda: _attach_verified_release_artifact(
                        tool_service.package_release(
                            str(project_root),
                            base.to_dict(),
                            base.calculate_hash(),
                            output_zip=release_output,
                            jar_path=str(jar_path),
                        ),
                        resource_pack_bundle,
                        archive_name='generated-resource-pack.zip',
                        allowed_root=run_root,
                        manifest_provenance={
                            'proposal_hash': approved.calculate_hash(),
                            'base_proposal_hash': base.calculate_hash(),
                            'proposal_scope': 'complete',
                        },
                    ),
                ),
                encode=lambda value: value,
                decode=lambda cached: cached,
                validate_cached=lambda cached: self._cached_package_exists(
                    cached, path_key='release_zip'
                ),
            )
            release_zip = str(release_result['release_zip'])
            metadata = build_distribution_metadata(
                jar_path=jar_path,
                mod_id=spec.mod_id,
                version=spec.version,
                name=spec.mod_name,
                changelog=options.changelog,
                platform_lock=spec.platform,
            )
            distribution_input = {
                'metadata_sha256': _stable_payload_sha256(metadata),
                'source_zip_sha256': str(release_result.get('sha256') or ''),
            }
            distribution_sha256 = _stable_payload_sha256(distribution_input)
            distribution_output = (
                run_root
                / 'releases'
                / (
                    'distribution-bundle-'
                    + distribution_sha256.split(':', 1)[1][:16]
                    + '.zip'
                )
            )
            bundle = run_named_checkpoint(
                ledger,
                'package-distribution',
                stage='package:distribution',
                input_value=distribution_input,
                action=lambda: _replace_stale_file_target(
                    distribution_output,
                    lambda: package_distribution_bundle(
                        metadata,
                        output_zip=distribution_output,
                        source_zip=release_zip,
                    ),
                ),
                encode=lambda value: value,
                decode=lambda cached: cached,
                validate_cached=lambda cached: self._cached_package_exists(
                    cached, path_key='path'
                ),
            )
            distribution_receipt = {'metadata': metadata, 'bundle': bundle}

            if options.publish_provider in {'modrinth', 'curseforge'}:
                provider = str(options.publish_provider)
                publish_input = {
                    'provider': provider,
                    'project_id': str(options.publish_project_id),
                    'metadata_sha256': _stable_payload_sha256(metadata),
                    'jar_sha256': str(metadata.get('jar_sha256') or ''),
                }
                distribution_receipt['publish'] = run_named_checkpoint(
                    ledger,
                    'publish-' + provider,
                    stage='publish:' + provider,
                    input_value=publish_input,
                    action=(
                        (lambda: publish_modrinth(
                            metadata,
                            project_id=str(options.publish_project_id),
                        ))
                        if provider == 'modrinth'
                        else (lambda: publish_curseforge(
                            metadata,
                            project_id=str(options.publish_project_id),
                        ))
                    ),
                    encode=lambda value: value,
                    decode=lambda cached: cached,
                    validate_cached=lambda cached: (
                        isinstance(cached, dict)
                        and cached.get('status') == 'PUBLISHED'
                        and cached.get('provider') == provider
                        and cached.get('jar_sha256') == metadata.get('jar_sha256')
                    ),
                )

            downloadable_input = {
                'artifact_sha256': str(artifact_receipt.get('sha256') or ''),
                'coverage_sha256': str(coverage_receipt.get('coverage_sha256') or ''),
                'reuse_manifest_sha256': _stable_payload_sha256(reuse_manifest),
                'build_receipt_sha256': _stable_payload_sha256(build_receipt),
                'runtime_receipt_sha256': _stable_payload_sha256(persisted_runtime_receipt),
                'resource_pack_sha256': str(
                    resource_pack_bundle.get('sha256') if resource_pack_bundle else ''
                ),
            }
            downloadable_sha256 = _stable_payload_sha256(downloadable_input)
            downloadable_target = (
                run_root
                / 'releases'
                / (
                    'final-mod-download-'
                    + downloadable_sha256.split(':', 1)[1][:16]
                )
            )
            distribution_receipt['downloadable_bundle'] = run_named_checkpoint(
                ledger,
                'package-downloadable',
                stage='package:downloadable',
                input_value=downloadable_input,
                action=lambda: _replace_stale_directory_target(
                    downloadable_target,
                    lambda: write_downloadable_bundle(
                        downloadable_target,
                        artifact_receipt=artifact_receipt,
                        requirement_coverage=coverage_receipt,
                        reuse_manifest=reuse_manifest,
                        build_receipt=build_receipt,
                        runtime_receipt=persisted_runtime_receipt,
                        additional_artifacts={
                            **(
                                {
                                    'generated-resource-pack.zip': resource_pack_bundle
                                }
                                if resource_pack_bundle is not None
                                else {}
                            ),
                            **_runtime_visual_download_artifacts(visual_receipt),
                        } or None,
                    ),
                ),
                encode=lambda value: value,
                decode=lambda cached: cached,
                validate_cached=self._cached_download_bundle_exists,
            )
            self._succeed_work_node(
                ledger,
                'package-release',
                {
                    'schema_version': 'mmm/work-node-receipt-v1',
                    'status': 'PASS',
                    'release_zip': release_zip,
                },
            )
        else:
            ledger.fail(
                'package-release',
                'Release quality evidence is incomplete.',
                input_required=True,
            )
        self._persist_work_evidence(project_root, ledger, work_plan)
        return CompletePipelineResult(schema_version='mmm/complete-pipeline-result-v3', status='VERIFIED' if release_ready else 'BUILT_WITH_UNRESOLVED_GATES', project_root=str(project_root), release_zip=release_zip, jar_path=str(jar_path), complete_proposal_hash=approved.calculate_hash(), source_validation=source_report, build_report=build, jar_validation=jar_validation, module_receipts=tuple(module_receipts), asset_receipt=asset_receipt, blockbench_receipts=tuple(blockbench_receipts), runtime_receipt=runtime_receipt, playtest_receipt=playtest_receipt, visual_receipt=visual_receipt, distribution_receipt=distribution_receipt, unresolved_gates=tuple(sorted(set(unresolved))), release_ready=release_ready, work_graph_hash=work_plan.graph_hash, work_ledger_path=str(ledger.path), run_resumed=run_resumed, quality_report=quality_report, build_bundle_zip=build_bundle_zip)

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
        from concurrent.futures import FIRST_COMPLETED, Future, wait

        from . import scheduler_parallel_safety_contract as scheduler_safety
        from .evidence_first_pipeline_contract import build_execution_context
        from .production_generation_preflight import (
            ProductionGenerationPreflightError,
            validate_production_generation_project,
        )

        active_registry = getattr(router, "registry", None) if router is not None else None
        active_profile = (
            str(getattr(router, "profile", "") or "")
            if router is not None
            else str(getattr(self, "profile", "") or "")
        )
        ledger._mmm_shared_local_gpu_lane = scheduler_safety._profile_uses_shared_local_gpu(
            active_profile,
            active_registry,
        )
        evidence_context = build_execution_context(
            approved,
            project_root,
            policy=self.policy,
        )

        if bool(getattr(options, "resume", False)):
            for node in work_plan.nodes:
                node_id = str(node.node_id)
                if not str(node.stage).startswith("generate:"):
                    continue
                if str(ledger.task(node_id)["state"]) == "failed":
                    ledger.retry(node_id)

        spec = approved.base_proposal.spec
        try:
            validate_production_generation_project(
                project_root,
                ordered,
                mod_id=spec.mod_id,
                package_name=spec.package_name,
                policy=self.policy,
            )
        except ProductionGenerationPreflightError as exc:
            raise CompleteProductionError(
                f"Production generation preflight failed before dispatch: {exc}"
            ) from exc
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
        extended_kinds = {'item', 'block', 'tool', 'weapon', 'armor', 'food', 'crop', 'machine', 'effect', 'enchantment', 'command', 'recipe', 'tag', 'advancement', 'loot'}
        module_receipts: list[dict[str, Any]] = []
        blockbench_receipts: list[dict[str, Any]] = []
        unresolved: list[str] = []
        asset_shards: list[dict[str, Any]] = []
        review_futures: list[tuple[str, Future[dict[str, Any]], float]] = []
        review_futures_lock = threading.Lock()
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
        generation_checkpoint_owners: dict[str, CustomModuleGenerator] = {}
        generation_checkpoint_owners_lock = threading.RLock()

        def _custom_generation_receipts(value: Any) -> tuple[dict[str, Any], ...]:
            found: list[dict[str, Any]] = []

            def visit(item: Any) -> None:
                if isinstance(item, dict):
                    if (
                        item.get('schema_version') == 'mmm/custom-module-result-v3'
                        and isinstance(item.get('generation_checkpoint'), dict)
                    ):
                        found.append(item)
                    for nested in item.values():
                        if nested is not item:
                            visit(nested)
                elif isinstance(item, (list, tuple)):
                    for nested in item:
                        visit(nested)

            visit(value)
            return tuple(found)

        def _register_checkpoint_owner(
            result: dict[str, Any],
            generator: CustomModuleGenerator,
        ) -> None:
            checkpoint = result.get('generation_checkpoint')
            token = checkpoint.get('cleanup_token') if isinstance(checkpoint, dict) else None
            if isinstance(token, str) and token:
                with generation_checkpoint_owners_lock:
                    generation_checkpoint_owners[token] = generator

        def _finalize_committed_generation_receipts(receipt: dict[str, Any]) -> None:
            for result in _custom_generation_receipts(receipt):
                checkpoint = result.get('generation_checkpoint')
                token = checkpoint.get('cleanup_token') if isinstance(checkpoint, dict) else None
                owner = None
                if isinstance(token, str):
                    with generation_checkpoint_owners_lock:
                        owner = generation_checkpoint_owners.pop(token, None)
                finalized = (
                    owner.finalize_committed_generation_checkpoint(
                        result,
                        project_root=project_root,
                    )
                    if owner is not None
                    else finalize_persisted_generation_checkpoint(
                        result,
                        project_root=project_root,
                        checkpoint_root=run_root / '.minecraft_ai' / '.mmm-custom-checkpoints',
                    )
                )
                if not finalized:
                    raise CompleteProductionError(
                        'Committed custom-generation checkpoint could not be finalized safely.'
                    )

        def _release_uncommitted_generation_receipts(receipt: dict[str, Any]) -> None:
            for result in _custom_generation_receipts(receipt):
                checkpoint = result.get('generation_checkpoint')
                token = checkpoint.get('cleanup_token') if isinstance(checkpoint, dict) else None
                owner = None
                if isinstance(token, str):
                    with generation_checkpoint_owners_lock:
                        owner = generation_checkpoint_owners.pop(token, None)
                if owner is not None:
                    owner.release_generation_checkpoint(result)

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

        def module_node_action(
            node: WorkNode,
            members: list[ProductionModule],
            uncommitted_custom_results: list[dict[str, Any]],
        ) -> dict[str, Any]:
            stage = str(node.payload.get('generation_stage', ''))
            receipts: list[dict[str, Any]] = []
            node_custom_generator: CustomModuleGenerator | None = None

            def generate_custom(module: ProductionModule) -> dict[str, Any]:
                nonlocal node_custom_generator
                if node_custom_generator is None:
                    node_custom_generator = new_custom_generator()

                execution_feedback = None
                feedback_context = getattr(
                    self, "_mmm_execution_feedback_context", None
                )
                if isinstance(feedback_context, dict):
                    invalidation = feedback_context.get("invalidation_receipt")
                    owner_ids = (
                        invalidation.get("owner_ids")
                        if isinstance(invalidation, dict)
                        else ()
                    )
                    if (
                        isinstance(owner_ids, (list, tuple))
                        and module.module_id in owner_ids
                    ):
                        raw_feedback = feedback_context.get("feedback")
                        if isinstance(raw_feedback, dict):
                            execution_feedback = raw_feedback

                result = node_custom_generator.generate(
                    project_root,
                    module=module,
                    research_modules=research_modules,
                    minecraft_version=spec.platform.minecraft_version,
                    loader=spec.platform.loader,
                    mappings=spec.platform.yarn_mappings,
                    execution_feedback=execution_feedback,
                )
                _register_checkpoint_owner(result, node_custom_generator)
                uncommitted_custom_results.append(result)
                return result

            if stage == 'content':
                research_shards = [module for module in members if is_research_shard(module)]
                receipts.extend(write_research_shard(project_root, module=module) for module in research_shards)

                raw_jobs = approved.game_design.get("_artifact_jobs") or []
                jobs_by_owner: dict[str, list[Any]] = {}
                for rj in raw_jobs:
                    owner = (
                        rj.get("owner_module")
                        if isinstance(rj, dict)
                        else getattr(rj, "owner_module", "")
                    )
                    if owner:
                        jobs_by_owner.setdefault(owner, []).append(rj)

                artifact_handled_members: list[ProductionModule] = []
                artifact_jobs_to_run: list[ArtifactJob] = []
                seen_job_ids: set[str] = set()
                for module in members:
                    if module.module_id in jobs_by_owner:
                        artifact_handled_members.append(module)
                        for j in jobs_by_owner[module.module_id]:
                            job_obj = ArtifactJob.from_dict(j) if isinstance(j, dict) else j
                            if job_obj.job_id not in seen_job_ids:
                                artifact_jobs_to_run.append(job_obj)
                                seen_job_ids.add(job_obj.job_id)

                # Include only the exact transitive prerequisites of these owners.
                all_jobs = [ArtifactJob.from_dict(j) if isinstance(j, dict) else j for j in raw_jobs]
                producers = {}
                for candidate in all_jobs:
                    for port in candidate.produces:
                        if port in producers and producers[port].job_id != candidate.job_id:
                            raise CompleteProductionError(f"ARTIFACT_DUPLICATE_PRODUCER: {port}")
                        producers[port] = candidate
                cursor = 0
                while cursor < len(artifact_jobs_to_run):
                    current_job = artifact_jobs_to_run[cursor]
                    cursor += 1
                    for dependency in current_job.requires:
                        prerequisite = producers.get(dependency)
                        if prerequisite is not None and prerequisite.job_id not in seen_job_ids:
                            artifact_jobs_to_run.append(prerequisite)
                            seen_job_ids.add(prerequisite.job_id)

                if artifact_jobs_to_run:
                    ensure_artifact_scaffolding(
                        project_root,
                        mod_id=spec.mod_id,
                        package_name=spec.package_name,
                        main_class=getattr(spec, "main_class", "") or "",
                    )
                    graph_receipt = execute_artifact_graph(
                        artifact_jobs_to_run,
                        context={"project_root": project_root, "base_dir": project_root,
                                 "resolved_version_context": spec.platform.version_context.to_dict()},
                        base_dir=project_root,
                    )
                    touched_paths: list[str] = [
                        str(r["materialization"]["path"])
                        for r in graph_receipt.get("receipts", [])
                        if isinstance(r, dict)
                        and r.get("materialization")
                        and r["materialization"].get("path")
                    ]

                    receipts.append(
                        {
                            "schema_version": "mmm/artifact-graph-execution-receipt-v1",
                            "status": "SUCCEEDED",
                            "module_ids": [m.module_id for m in artifact_handled_members],
                            "files": touched_paths,
                            "touched_paths": touched_paths,
                            "completed_jobs": graph_receipt.get("completed_jobs", []),
                            "receipts": graph_receipt.get("receipts", []),
                            "ports": graph_receipt.get("ports", {}),
                        }
                    )

                deterministic = [
                    module
                    for module in members
                    if module.kind in extended_kinds
                    and module not in research_shards
                    and module not in artifact_handled_members
                ]
                if deterministic:
                    receipts.append(generate_extended_content(project_root=project_root, mod_id=spec.mod_id, package_name=spec.package_name, modules=deterministic, policy=self.policy))
                sidecars = [module for module in members if module.kind == 'integration' and module.config.get('integration_type') == LOCAL_AI_SIDECAR_INTEGRATION_TYPE]
                receipts.extend(generate_local_ai_sidecar(project_root=project_root, mod_id=spec.mod_id, package_name=spec.package_name, module=module, policy=self.policy) for module in sidecars)
                receipts.extend(generate_custom(module) for module in members if (module.kind not in extended_kinds or module.config.get("requires_custom_generation")) and module not in sidecars and (module not in research_shards) and (module not in artifact_handled_members))
            elif stage == 'system':
                for pack_id, pack_modules in _system_groups(members).items():
                    receipts.append(generate_system_pack(project_root=project_root, pack_id=pack_id, mod_id=spec.mod_id, package_name=spec.package_name, config={'modules': [_module_dict(item) for item in pack_modules]}, policy=self.policy))
            elif stage == 'entity':
                for module in members:
                    config = dict(module.config)
                    required_entity_config = (
                        "max_health",
                        "attack_damage",
                        "movement_speed",
                        "follow_range",
                        "archetype",
                        "behavior",
                        "entity_width",
                        "entity_height",
                        "spawn_group",
                        "main_color",
                    )
                    missing_entity_config = [
                        key for key in required_entity_config if config.get(key) in (None, "")
                    ]
                    if missing_entity_config:
                        raise CompleteProductionError(
                            f"ENTITY_DESIGN_UNRESOLVED: {module.module_id} missing {missing_entity_config}"
                        )
                    receipts.append(
                        generate_geckolib_entity_assets(
                            project_root=project_root,
                            mod_id=spec.mod_id,
                            package_name=spec.package_name,
                            entity_id=module.module_id,
                            texture_width=int(config.get("texture_width", 64)),
                            texture_height=int(config.get("texture_height", 64)),
                            max_health=float(config["max_health"]),
                            attack_damage=float(config["attack_damage"]),
                            movement_speed=float(config["movement_speed"]),
                            follow_range=float(config["follow_range"]),
                            archetype=str(config["archetype"]),
                            behavior=str(config["behavior"]),
                            entity_width=float(config["entity_width"]),
                            entity_height=float(config["entity_height"]),
                            spawn_group=str(config["spawn_group"]),
                            texture_color=str(config["main_color"]),
                            custom_bones=config.get("custom_bones")
                            if isinstance(config.get("custom_bones"), list)
                            else None,
                            policy=self.policy,
                        )
                    )
                    if config.get("requires_custom_generation"):
                        receipts.append(generate_custom(module))
            elif stage == 'custom':
                receipts.extend(generate_custom(module) for module in members)
            else:
                raise CompleteProductionError(f'Unsupported generation work stage: {stage}')
            if spec.platform.host_facts_json:
                resolved_context = spec.platform.version_context
                for receipt in receipts:
                    if isinstance(receipt, dict):
                        if "context_id" in receipt:
                            resolved_context.assert_context(receipt["context_id"])
                        receipt["context_id"] = resolved_context.context_id
            semantic_observations = _semantic_execution_observations(
                members,
                receipts,
                downstream_ids=downstream_ids,
                evidence_context=evidence_context,
            )
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
                uncommitted_custom_results: list[dict[str, Any]] = []
                receipt = self._run_work_node(
                    ledger,
                    node,
                    action=lambda node=node, members=members: module_node_action(
                        node,
                        members,
                        uncommitted_custom_results,
                    ),
                    validate_cached=lambda value: self._receipt_outputs_exist(
                        value,
                        project_root=project_root,
                    ),
                    shared_index=shared_project_index,
                    on_commit=_finalize_committed_generation_receipts,
                    on_abort=lambda _receipt: _release_uncommitted_generation_receipts(
                        {'receipts': uncommitted_custom_results}
                    ),
                )
                children = [item for item in receipt.get('receipts', []) if isinstance(item, dict)]
                module_receipts.extend(children)
                if node.payload.get('generation_stage') == 'entity':
                    entity_receipts = {str(item.get('entity_id')): item for item in children if item.get('entity_id')}
                    for module in members:
                        entity_receipt = entity_receipts.get(module.module_id)
                        if entity_receipt is None:
                            raise CompleteProductionError(f'Entity generation node omitted its receipt: {module.module_id}')
                        if options.run_blockbench and (not options.source_only):
                            review_deadline = time.monotonic() + lease_seconds
                            def review_action(receipt=entity_receipt, module_id=module.module_id):
                                return run_named_checkpoint(
                                    ledger,
                                    f'blockbench-review-{module_id}',
                                    stage='validate:blockbench',
                                    input_value={
                                        'graph_hash': work_plan.graph_hash,
                                        'entity_receipt': receipt,
                                        'geometry_sha256': self._blockbench_geometry_sha256(receipt),
                                    },
                                    action=lambda: self._blockbench_review(receipt, run_root),
                                    encode=lambda value: value,
                                    decode=lambda cached: cached,
                                    validate_cached=self._cached_blockbench_review,
                                )
                            review_future = review_pool.submit(
                                run_with_model_execution_deadline,
                                review_deadline,
                                review_action,
                            )
                            with review_futures_lock:
                                review_futures.append((module.module_id, review_future, review_deadline))
                        elif options.run_blockbench:
                            unresolved.append(f'blockbench:{module.module_id}:not-run-in-source-only-mode')
            elif kind == 'asset-shard':
                ids = [str(item.get('asset_id')) for item in node.payload.get('members', []) if isinstance(item, dict)]
                if not ids or any(item not in asset_lookup for item in ids):
                    raise CompleteProductionError(f'Work node {node.node_id} has invalid assets.')
                shard_proposal = replace(approved, assets=tuple(asset_lookup[item] for item in ids), approval_hash='')
                asset_shards.append(self._run_work_node(ledger, node, action=lambda proposal=shard_proposal: self._generate_assets(get_router(), proposal, project_root, run_root), validate_cached=self._cached_asset_shard, shared_index=shared_project_index))
            else:
                raise CompleteProductionError(f'Unsupported work node payload kind: {kind}')
        capacities = scheduler_safety._capacities()
        cpu_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['cpu_io'])), thread_name_prefix='cpu_io')
        llm_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['llm'])), thread_name_prefix='llm')
        image_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['image_gpu'])), thread_name_prefix='image_gpu')
        commit_pool = ThreadPoolExecutor(max_workers=max(1, int(capacities['commit'])), thread_name_prefix='commit')
        review_workers_raw = os.environ.get('MMM_BLOCKBENCH_REVIEW_WORKERS', '').strip()
        try:
            review_workers = int(review_workers_raw) if review_workers_raw else max(1, min(4, int(capacities['cpu_io'])))
        except ValueError:
            review_workers = max(1, min(4, int(capacities['cpu_io'])))
        review_pool = ThreadPoolExecutor(max_workers=max(1, review_workers), thread_name_prefix='blockbench_review')
        node_futures: dict[str, Future[Any]] = {}
        idle_wait = threading.Event()
        lease_seconds = 900
        heartbeat_seconds = 60.0

        def dispatch_node(node: WorkNode) -> Future[Any]:
            resource_class = node.resource_class or str(node.payload.get('resource_class', 'cpu_io'))
            deadline = time.monotonic() + lease_seconds
            args = (run_with_model_execution_deadline, deadline, process_node, node)
            if resource_class == 'llm':
                return llm_pool.submit(*args)
            if resource_class == 'image_gpu':
                return image_pool.submit(*args)
            if resource_class == 'commit':
                return commit_pool.submit(*args)
            return cpu_pool.submit(*args)
        try:
            while True:
                ledger.raise_if_cancelled()
                done_ids = [node_id for node_id, future in node_futures.items() if future.done()]
                for node_id in done_ids:
                    future = node_futures.pop(node_id)
                    try:
                        future.result(timeout=0)
                    except BaseException as exc:
                        print(
                            f"\n[ORCHESTRATOR ERROR] Node {node_id} failed with "
                            f"{type(exc).__name__}:\n{exc}\n"
                            f"[ORCHESTRATOR TRACEBACK]\n{traceback.format_exc()}"
                            f"[ORCHESTRATOR MODULE] {__file__}\n",
                            flush=True,
                        )
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
                    wait(
                        tuple(node_futures.values()),
                        timeout=heartbeat_seconds,
                        return_when=FIRST_COMPLETED,
                    )
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
            review_results: list[tuple[str, dict[str, Any]]] = []
            for module_id, future, deadline in tuple(review_futures):
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0.0 and not future.done():
                    future.cancel()
                    raise CompleteProductionError(
                        f'Blockbench review deadline exceeded: {module_id}'
                    )
                try:
                    receipt = future.result(timeout=0 if future.done() else remaining)
                except TimeoutError as exc:
                    future.cancel()
                    raise CompleteProductionError(
                        f'Blockbench review deadline exceeded: {module_id}'
                    ) from exc
                review_results.append((module_id, receipt))
            blockbench_receipts.extend(receipt for _, receipt in sorted(review_results, key=lambda item: item[0]))
        finally:
            for future in node_futures.values():
                future.cancel()
            for _, future, _ in review_futures:
                future.cancel()
            # Generation workers mutate the shared staged workspace.  Never return
            # while a running worker from this attempt can still write into it.
            cpu_pool.shutdown(wait=True, cancel_futures=True)
            llm_pool.shutdown(wait=True, cancel_futures=True)
            image_pool.shutdown(wait=True, cancel_futures=True)
            commit_pool.shutdown(wait=True, cancel_futures=True)
            review_pool.shutdown(wait=True, cancel_futures=True)
        module_receipts.sort(key=_generation_receipt_sort_key)
        asset_shards.sort(key=_generation_receipt_sort_key)
        unresolved.sort()
        resource_pack_bundle = self._finalize_resource_pack_bundle(
            asset_shards,
            run_root=run_root,
        )
        asset_receipt = (
            {
                'schema_version': 'mmm/complete-assets-sharded-v1',
                'status': 'GENERATED',
                'shard_count': len(asset_shards),
                'asset_count': sum(len(item.get('assets', [])) for item in asset_shards),
                'shards': asset_shards,
                'resource_pack_bundle': resource_pack_bundle,
            }
            if asset_shards
            else None
        )
        return {'module_receipts': module_receipts, 'blockbench_receipts': blockbench_receipts, 'asset_receipt': asset_receipt, 'unresolved': unresolved, 'router': router}

    @staticmethod
    def _finalize_resource_pack_bundle(
        asset_shards: Iterable[dict[str, Any]],
        *,
        run_root: Path,
    ) -> dict[str, Any] | None:
        roots: set[Path] = set()
        for shard in asset_shards:
            validation = shard.get('container_validation')
            if not isinstance(validation, dict):
                continue
            pack = validation.get('standalone_resource_pack')
            if not isinstance(pack, dict) or pack.get('status') != 'PASS':
                continue
            raw = pack.get('root')
            if isinstance(raw, str) and raw:
                roots.add(Path(raw).expanduser().resolve())
        if not roots:
            return None
        if len(roots) != 1:
            raise CompleteProductionError(
                'Asset shards disagree on the standalone resource-pack root.'
            )
        root = next(iter(roots))
        allowed_root = run_root.expanduser().resolve()
        try:
            root.relative_to(allowed_root)
        except ValueError as exc:
            raise CompleteProductionError(
                'Standalone resource-pack root escaped the run workspace.'
            ) from exc
        if not root.is_dir() or root.is_symlink():
            raise CompleteProductionError(
                'Standalone resource-pack root is missing or unsafe.'
            )

        target = allowed_root / 'resource-packs' / 'generated-resource-pack-final.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise CompleteProductionError(
                    'Standalone resource-pack bundle target is unsafe.'
                )
            target.unlink()

        files = [
            path
            for path in sorted(root.rglob('*'))
            if path.is_file() and not path.is_symlink()
        ]
        if not files:
            raise CompleteProductionError(
                'Standalone resource-pack root contains no files.'
            )
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o644 & 0xFFFF) << 16
                archive.writestr(info, path.read_bytes())
        return {
            'status': 'PASS',
            'path': str(target),
            'sha256': CompleteProductionOrchestrator._file_hash(target),
            'file_count': len(files),
        }

    @staticmethod
    def _run_work_node(
        ledger: DurableWorkLedger,
        node: WorkNode,
        *,
        action: Callable[[], dict[str, Any]],
        validate_cached: Callable[[dict[str, Any]], bool],
        shared_index: ProjectIndex | None = None,
        on_commit: Callable[[dict[str, Any]], None] | None = None,
        on_abort: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        from .project_write_lock import project_write_lock
        from .scheduler_claim_fencing_contract import (
            _commit_success,
            _fenced_fail,
            _snapshot_claim,
        )

        cached = ledger.cached_receipt(node.node_id, input_hash=node.input_hash)
        emit_root_cause(
            'orchestrator_node_decision',
            stage=node.stage,
            operation=node.node_id,
            gate='cache_and_ledger_state',
            result='START',
            details={
                'node': node.to_dict() if hasattr(node, 'to_dict') else str(node),
                'cached_receipt': cached,
            },
        )
        if cached is not None and validate_cached(cached):
            emit_root_cause(
                'orchestrator_node_cache_hit',
                stage=node.stage,
                operation=node.node_id,
                gate='cached_receipt_validation',
                result='PASS',
                details={'receipt': cached},
            )
            if on_commit is not None:
                on_commit(cached)
            return cached
        if cached is not None:
            emit_root_cause(
                'orchestrator_node_cache_invalidated',
                stage=node.stage,
                operation=node.node_id,
                gate='cached_receipt_validation',
                result='FAIL',
                reason='cached outputs failed existence/integrity validation',
                details={'receipt': cached},
            )
            ledger.invalidate(node.node_id)

        current = ledger.task(node.node_id)
        if current['state'] in {'failed', 'input_required', 'cancelled'}:
            emit_root_cause(
                'orchestrator_node_retry',
                stage=node.stage,
                operation=node.node_id,
                gate='retry_policy',
                result='START',
                reason=str(current.get('error') or current['state']),
                details={'before': current},
            )
            ledger.retry(node.node_id)
            current = ledger.task(node.node_id)

        ledger.raise_if_cancelled()
        if current['state'] != 'running':
            ledger.begin(node.node_id, worker_id='complete-orchestrator')
        running = ledger.task(node.node_id)
        claim_attempt = running.get('attempt')
        claim_owner = running.get('lease_owner')
        claim_fenced = (
            type(claim_attempt) is int
            and claim_attempt >= 1
            and isinstance(claim_owner, str)
            and bool(claim_owner)
        )
        if claim_fenced:
            claim_attempt, claim_owner = _snapshot_claim(ledger, node.node_id)

        receipt: dict[str, Any] | None = None
        committed = False

        def execute_and_commit() -> dict[str, Any]:
            nonlocal receipt, committed
            emit_root_cause(
                'orchestrator_node_action_start',
                stage=node.stage,
                operation=node.node_id,
                gate='work_node_action',
                result='START',
                details={'ledger_state': current, 'payload': node.payload},
            )
            receipt = action()
            if not isinstance(receipt, dict):
                raise CompleteProductionError(
                    f'Work node {node.node_id} returned a non-object receipt.'
                )
            ledger.raise_if_cancelled()
            if claim_fenced:
                _commit_success(
                    ledger,
                    node.node_id,
                    receipt,
                    attempt=claim_attempt,
                    owner=claim_owner,
                    shared_index=shared_index,
                    index_error_type=CompleteProductionError,
                )
            else:
                ledger.succeed(node.node_id, receipt)
                if shared_index is not None:
                    from .scheduler_parallel_safety_contract import (
                        _receipt_touched_paths,
                    )

                    touched = _receipt_touched_paths(receipt)
                    if touched:
                        shared_index.update_files(touched)
                        shared_index.write_manifest()
            committed = True
            if on_commit is not None:
                on_commit(receipt)
            emit_root_cause(
                'orchestrator_node_action_result',
                stage=node.stage,
                operation=node.node_id,
                gate='work_node_action',
                result='PASS',
                details={'receipt': receipt},
            )
            return receipt

        project_root = (
            getattr(shared_index, 'root', None)
            if getattr(node, 'resource_class', '') == 'commit' and shared_index is not None
            else None
        )
        try:
            if project_root is not None:
                with project_write_lock(project_root):
                    return execute_and_commit()
            return execute_and_commit()
        except BaseException as exc:
            if not committed and on_abort is not None:
                try:
                    on_abort(receipt or {})
                except BaseException as abort_exc:  # noqa: BLE001 - retain the original failure after cleanup
                    emit_root_cause(
                        'orchestrator_node_abort_cleanup_failure',
                        stage=node.stage,
                        operation=node.node_id,
                        gate='generation_checkpoint_release',
                        result='FAIL',
                        reason=f'{type(abort_exc).__name__}: {abort_exc}',
                        exc=abort_exc,
                    )
            if claim_fenced:
                _fenced_fail(
                    ledger,
                    node.node_id,
                    attempt=claim_attempt,
                    owner=claim_owner,
                    error=exc,
                )
            else:
                try:
                    if ledger.task(node.node_id)['state'] == 'running':
                        ledger.fail(node.node_id, f'{type(exc).__name__}: {exc}')
                except WorkGraphError:
                    pass
            emit_root_cause(
                'orchestrator_node_action_failure',
                stage=node.stage,
                operation=node.node_id,
                gate='work_node_action',
                result='FAIL',
                reason=f'{type(exc).__name__}: {exc}',
                details={'ledger_state': current, 'payload': node.payload},
                exc=exc,
            )
            raise

    def _inspect_existing_project_input(
        self,
        approved: CompleteProposal,
        *,
        run_root: Path,
        existing_input: str | Path,
    ) -> tuple[Any, Path]:
        try:
            report = inspect_existing_project_archive(
                existing_input,
                extract_root=run_root / 'existing-source',
                expected_archive_sha256=approved.existing_input_sha256,
            )
        except ExistingProjectImportError as exc:
            raise CompleteProductionError(str(exc)) from exc
        if not report.has_sources or not report.has_gradle_project or (not report.extracted_to):
            raise CompleteProductionError('Existing mod modification requires a source Gradle ZIP.')
        project_root = self._locate_imported_project(report, run_root=run_root)
        info = inspect_fabric_project(project_root)
        expected = approved.base_proposal.spec
        if info.mod_id != expected.mod_id or info.package_name != expected.package_name:
            raise CompleteProductionError(
                f'Approved proposal does not match the existing Fabric project: '
                f'expected {expected.mod_id}/{expected.package_name}, '
                f'found {info.mod_id}/{info.package_name}.'
            )
        return report, project_root

    @staticmethod
    def _bind_debug_fixture_runtime(
        approved: CompleteProposal,
        project_root: Path,
    ) -> Path:
        """Bind the model-owned DebugToken source to host-owned runtime execution."""

        root = project_root.expanduser().resolve()
        schema_version = str(getattr(approved, 'schema_version', '') or '')
        game_design = getattr(approved, 'game_design', {})
        if not isinstance(game_design, dict):
            game_design = {}
        if not (
            schema_version == 'mmm/complete-proposal-v1'
            and game_design.get('mode') == 'debug_fixture'
        ):
            return root

        fixture = game_design.get('fixture')
        fixture_module_id = (
            str(fixture.get('module_id') or '').strip()
            if isinstance(fixture, dict)
            else ''
        )
        fixture_module = next(
            (
                module
                for module in approved.modules
                if module.module_id == fixture_module_id
            ),
            None,
        )
        source_contract = (
            fixture_module.config.get('observable_source_contract')
            if fixture_module is not None and isinstance(fixture_module.config, dict)
            else None
        )
        if not isinstance(source_contract, dict):
            raise CompleteProductionError(
                'Debug fixture is missing its observable source contract.'
            )
        binding_field = str(source_contract.get('binding_field') or '').strip()
        if not binding_field or re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', binding_field) is None:
            raise CompleteProductionError(
                'Debug fixture binding_field is missing or invalid.'
            )

        package_name = approved.base_proposal.spec.package_name
        package_path = package_name.replace('.', '/')
        main_class = ''.join(
            part.capitalize()
            for part in approved.base_proposal.spec.mod_id.split('_')
        ) + 'Mod'
        main_relative = f'src/main/java/{package_path}/{main_class}.java'
        gametest_relative = (
            f'src/main/java/{package_path}/{main_class}GameTests.java'
        )

        bootstrap_receipt = root / '.minecraft_ai/fabric-template-receipt.json'
        if bootstrap_receipt.is_file() and not bootstrap_receipt.is_symlink():
            try:
                bootstrap = json.loads(
                    bootstrap_receipt.read_text(encoding='utf-8')
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CompleteProductionError(
                    'Debug fixture Fabric template receipt is invalid.'
                ) from exc
            runtime_contract = bootstrap.get('runtime_contract')
            gametest_contract = bootstrap.get('gametest_contract')
            if isinstance(runtime_contract, dict):
                candidate = runtime_contract.get('main_source')
                if isinstance(candidate, str) and candidate.strip():
                    main_relative = candidate.strip()
            if isinstance(gametest_contract, dict):
                candidate = gametest_contract.get('source')
                if isinstance(candidate, str) and candidate.strip():
                    gametest_relative = candidate.strip()

        def owned_source(relative: str, label: str) -> Path:
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise CompleteProductionError(
                    f'Debug fixture {label} escaped the project root.'
                ) from exc
            if not candidate.is_file() or candidate.is_symlink():
                raise CompleteProductionError(
                    f'Debug fixture {label} is missing: {relative}'
                )
            return candidate

        main_source = owned_source(main_relative, 'main entrypoint source')
        gametest_source = owned_source(gametest_relative, 'GameTest source')
        debug_class_name = package_name + '.DebugToken'

        main_text = main_source.read_text(encoding='utf-8')
        main_marker = 'MMM_DEBUG_FIXTURE_RUNTIME_BINDING'
        if main_marker not in main_text:
            match = re.search(
                r'public\s+void\s+onInitialize\s*\(\s*\)\s*\{',
                main_text,
            )
            if match is None:
                raise CompleteProductionError(
                    'Debug fixture host main entrypoint has no onInitialize method.'
                )
            injection = (
                match.group(0)
                + '\n        // '
                + main_marker
                + '\n        try {\n'
                + '            Class<?> debugTokenClass = Class.forName('
                + f'"{debug_class_name}", true, {main_class}.class.getClassLoader());'
                + '\n            if (debugTokenClass.getField("'
                + binding_field
                + '").get(null) == null) {\n'
                + '                throw new IllegalStateException('
                + '"debug_token registration binding is null");\n'
                + '            }\n'
                + '        } catch (ReflectiveOperationException exc) {\n'
                + '            throw new IllegalStateException('
                + '"debug_token runtime binding failed", exc);\n'
                + '        }'
            )
            main_text = (
                main_text[: match.start()]
                + injection
                + main_text[match.end() :]
            )
            main_source.write_text(main_text, encoding='utf-8', newline='\n')

        gametest_text = gametest_source.read_text(encoding='utf-8')
        gametest_marker = 'MMM_DEBUG_FIXTURE_REGISTRY_ASSERTION'
        if gametest_marker not in gametest_text:
            terminal = next(
                (
                    value
                    for value in ('context.succeed();', 'context.complete();')
                    if value in gametest_text
                ),
                None,
            )
            if terminal is None:
                raise CompleteProductionError(
                    'Debug fixture host GameTest has no terminal success call.'
                )
            gametest_class = main_class + 'GameTests'
            assertion = (
                '        // '
                + gametest_marker
                + '\n        try {\n'
                + '            Class<?> debugTokenClass = Class.forName('
                + f'"{debug_class_name}", true, {gametest_class}.class.getClassLoader());'
                + '\n            if (debugTokenClass.getField("'
                + binding_field
                + '").get(null) == null) {\n'
                + '                throw new AssertionError('
                + '"debug_token runtime registry binding is null");\n'
                + '            }\n'
                + '        } catch (ReflectiveOperationException exc) {\n'
                + '            throw new AssertionError('
                + '"debug_token runtime registry binding is unavailable", exc);\n'
                + '        }\n'
            )
            gametest_text = gametest_text.replace(
                '        ' + terminal,
                assertion + '        ' + terminal,
                1,
            )
            gametest_source.write_text(
                gametest_text,
                encoding='utf-8',
                newline='\n',
            )

        return root

    def _prepare_project(self, approved: CompleteProposal, *, run_root: Path, existing_input: str | Path | None) -> Path:
        if existing_input is not None:
            _report, project_root = self._inspect_existing_project_input(
                approved,
                run_root=run_root,
                existing_input=existing_input,
            )
            return self._bind_debug_fixture_runtime(approved, project_root)
        base = approved.base_proposal
        from .platform_catalog import adapter_for_lock_values
        from .platform_live_execution_contract import (
            _uses_official_scaffold,
            prepare_official_fabric_project,
        )

        adapter = adapter_for_lock_values(base.spec.platform)
        if _uses_official_scaffold(adapter):
            project_root = prepare_official_fabric_project(
                self,
                approved,
                run_root=run_root,
                adapter=adapter,
                error_type=CompleteProductionError,
            )
            return self._bind_debug_fixture_runtime(approved, project_root)

        base.approve(base.calculate_hash())
        project_root = run_root / 'base/workspaces' / base.spec.mod_id
        if project_root.exists():
            if self._project_matches_spec(project_root, base.spec):
                self._write_base_proposal(project_root, base)
                return self._bind_debug_fixture_runtime(approved, project_root)
            self._preserve_partial_project(project_root)
        staging = project_root.with_name(f'.{project_root.name}.staging')
        if staging.exists():
            self._preserve_partial_project(staging)
        FabricProjectGenerator(policy=self.policy).generate(base.spec, staging)
        self._write_base_proposal(staging, base)
        if project_root.exists():
            if self._project_matches_spec(project_root, base.spec):
                self._preserve_partial_project(staging)
                return self._bind_debug_fixture_runtime(approved, project_root)
            self._preserve_partial_project(project_root)
        staging.replace(project_root)
        return self._bind_debug_fixture_runtime(approved, project_root)

    def _locate_imported_project(self, report: Any, *, run_root: Path) -> Path:
        extracted = Path(str(report.extracted_to)).resolve()
        try:
            return _locate_existing_fabric_root(extracted)
        except CompleteProductionError:
            nested_members = sorted({path.split('!/', 1)[0] for path in report.source_files if '!/' in path} & {path.split('!/', 1)[0] for path in report.gradle_files if '!/' in path})
            if len(nested_members) != 1:
                raise
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
        if not prepared_project_matches_spec(project_root, spec):
            return False
        try:
            return (
                adapter_from_project(project_root).adapter_id
                == adapter_for_lock_values(spec.platform).adapter_id
            )
        except (OSError, TypeError, ValueError):
            return False

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

    @feedback_run_context
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
        return prepared_project_cache_valid(path)

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
                    if key in {'files', 'generated_files', 'touched_paths', 'written_files'} and isinstance(nested, (list, tuple)):
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
    def _source_package_receipt(path: str) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        if not target.is_file() or target.is_symlink():
            raise CompleteProductionError(
                'Source package did not produce a regular ZIP file.'
            )
        return {
            'status': 'PACKAGED',
            'release_zip': str(target),
            'sha256': CompleteProductionOrchestrator._file_hash(target),
        }

    @staticmethod
    def _cached_asset_shard(receipt: Any) -> bool:
        if (
            not isinstance(receipt, dict)
            or receipt.get('status') != 'TEXTURE_PRODUCTION_PASS'
        ):
            return False
        for collection, path_key in (
            (receipt.get('assets'), 'target'),
            (receipt.get('documents'), 'resolved_path'),
        ):
            if not isinstance(collection, list):
                return False
            for item in collection:
                if not isinstance(item, dict):
                    return False
                raw = item.get(path_key)
                expected = item.get('sha256')
                if not isinstance(raw, str) or not isinstance(expected, str):
                    return False
                path = Path(raw).expanduser().resolve()
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or CompleteProductionOrchestrator._file_hash(path) != expected
                ):
                    return False
        graph = receipt.get('resource_graph_validation')
        contract = receipt.get('resource_contract_validation')
        return (
            isinstance(graph, dict)
            and graph.get('status') == 'PASS'
            and isinstance(contract, dict)
            and contract.get('status') == 'PASS'
        )

    @staticmethod
    def _blockbench_geometry_sha256(receipt: Any) -> str:
        if not isinstance(receipt, dict):
            raise CompleteProductionError('Entity receipt is invalid for Blockbench review.')
        raw = next(
            (
                str(path)
                for path in receipt.get('files', ())
                if isinstance(path, str) and path.endswith('.geo.json')
            ),
            '',
        )
        if not raw:
            raise CompleteProductionError(
                'Entity receipt contains no geometry for Blockbench review.'
            )
        path = Path(raw).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise CompleteProductionError(
                'Entity geometry is missing or unsafe before Blockbench review.'
            )
        return CompleteProductionOrchestrator._file_hash(path)

    @staticmethod
    def _cached_blockbench_review(receipt: Any) -> bool:
        if not isinstance(receipt, dict):
            return False
        uv = receipt.get('uv')
        raw = receipt.get('preview')
        expected = receipt.get('preview_sha256')
        if (
            not isinstance(uv, dict)
            or uv.get('status') not in {'PASS', 'OK'}
            or not isinstance(raw, str)
            or not isinstance(expected, str)
        ):
            return False
        path = Path(raw).expanduser().resolve()
        return (
            path.is_file()
            and not path.is_symlink()
            and CompleteProductionOrchestrator._file_hash(path) == expected
        )

    @staticmethod
    def _cached_download_bundle_exists(receipt: Any) -> bool:
        if not isinstance(receipt, dict) or receipt.get('status') != 'PASS':
            return False
        raw = receipt.get('path')
        members = receipt.get('members')
        manifest_sha256 = receipt.get('manifest_sha256')
        if (
            not isinstance(raw, str)
            or not isinstance(members, list)
            or not isinstance(manifest_sha256, str)
        ):
            return False
        root = Path(raw).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            return False
        for item in members:
            if not isinstance(item, dict):
                return False
            name = str(item.get('path') or '')
            expected = str(item.get('sha256') or '')
            if not name or Path(name).name != name or not expected:
                return False
            target = root / name
            if not target.is_file() or target.is_symlink():
                return False
            if CompleteProductionOrchestrator._file_hash(target) != expected:
                return False
        manifest_path = root / 'bundle-receipt.json'
        if not manifest_path.is_file() or manifest_path.is_symlink():
            return False
        try:
            persisted = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return (
            isinstance(persisted, dict)
            and persisted.get('manifest_sha256') == manifest_sha256
            and persisted.get('artifact_sha256') == receipt.get('artifact_sha256')
            and persisted.get('members') == members
        )

    @staticmethod
    def _cached_package_exists(receipt: Any, *, path_key: str) -> bool:
        if not isinstance(receipt, dict) or receipt.get('status') != 'PACKAGED':
            return False
        raw = receipt.get(path_key)
        expected = receipt.get('sha256')
        if not isinstance(raw, str) or not isinstance(expected, str):
            return False
        path = Path(raw).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            return False
        return CompleteProductionOrchestrator._file_hash(path) == expected

    @staticmethod
    def _cached_build_exists(
        build: Any,
        *,
        require_gametest: bool = False,
        spec: Any = None,
    ) -> bool:
        if not isinstance(build, dict) or build.get('status') != 'PASS':
            return False
        if not CompleteProductionOrchestrator._full_gradle_build_receipt_passed(build):
            return False
        raw = build.get('jar_path')
        if not isinstance(raw, str):
            return False
        path = Path(raw).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            return False
        if require_gametest:
            if spec is None:
                return False
            if not CompleteProductionOrchestrator._gametest_receipt_passed(build, spec):
                return False
        return True

    @staticmethod
    def _mandatory_gametest_failures(
        build_report: dict[str, Any] | None,
        spec: Any,
    ) -> list[str]:
        return (
            []
            if CompleteProductionOrchestrator._gametest_receipt_passed(
                build_report,
                spec,
            )
            else ['gametest:missing-host-required-evidence']
        )

    @staticmethod
    def _mandatory_blockbench_failures(
        proposal: CompleteProposal,
        blockbench_receipts: Iterable[dict[str, Any]],
    ) -> list[str]:
        expected = {
            module.module_id
            for module in proposal.modules
            if module.kind in {'entity', 'boss', 'npc'}
        }
        if not expected:
            return []
        passed = {
            str(receipt.get('entity'))
            for receipt in blockbench_receipts
            if CompleteProductionOrchestrator._cached_blockbench_review(receipt)
        }
        return [
            f'blockbench:{module_id}:missing-host-required-review'
            for module_id in sorted(expected - passed)
        ]

    @staticmethod
    def _required_gate_failures(proposal: CompleteProposal, *, generated_receipts: Iterable[Any], project_root: Path | None=None, source_validation: dict[str, Any] | None, jdt_receipt: dict[str, Any] | None, build_report: dict[str, Any] | None, jar_validation: dict[str, Any] | None, blockbench_receipts: Iterable[dict[str, Any]], runtime_receipt: dict[str, Any] | None, playtest_receipt: dict[str, Any] | None, visual_receipt: dict[str, Any] | None) -> list[str]:
        """Resolve every declared gate against an explicit evidence receipt.

        Unknown approved gate names are deliberately unresolved. Generated receipts
        are evidence only and cannot introduce release obligations beyond the approved plan.
        """
        receipt_values = tuple(generated_receipts)
        requirements: set[tuple[str, str]] = {(module.module_id, gate.strip()) for module in proposal.modules for gate in module.required_gates if gate.strip()}
        research_ledger_receipts: list[dict[str, Any]] = []

        def collect(value: Any, owner: str='generated') -> None:
            if isinstance(value, dict):
                if value.get('schema_version') == 'mmm/research-ledger-write-receipt-v1':
                    research_ledger_receipts.append(value)
                local_owner = next((str(value[key]) for key in ('module_id', 'entity_id', 'pack_id', 'sound_id') if isinstance(value.get(key), str) and value[key]), owner)
                # Release-gate authority is owned only by the approved proposal.
                # Receipts may describe checks they performed, but they cannot add
                # obligations that were never approved by the plan.
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
        gradle_passed = CompleteProductionOrchestrator._full_gradle_build_receipt_passed(
            build_report
        )
        jdt_passed = _jdt_release_evidence_passed(jdt_receipt)
        evidence = {'source': isinstance(source_validation, dict) and source_validation.get('status') == 'PASS', 'jdt': jdt_passed, 'gradle': gradle_passed, 'gametest': gradle_passed and CompleteProductionOrchestrator._gametest_receipt_passed(build_report, proposal.base_proposal.spec), 'jar': isinstance(jar_validation, dict) and jar_validation.get('status') == 'PASS', 'runtime_client': isinstance(runtime_receipt, dict) and isinstance(runtime_receipt.get('server'), dict) and (runtime_receipt['server'].get('server_running') is True) and isinstance(runtime_receipt.get('client'), dict) and (runtime_receipt['client'].get('client_running') is True), 'playtest': _playtest_evidence_passed(playtest_receipt, getattr(proposal, 'acceptance_tests', ())), 'visual': _visual_runtime_evidence_passed(visual_receipt, runtime_receipt), 'research_ledger': bool(expected_research) and all((passed_research.get(module_id) == hashes for module_id, hashes in expected_research.items()))}
        evidence['runtime_visual'] = evidence['runtime_client'] and evidence['visual']
        evidence['playtest_visual'] = evidence['playtest'] and evidence['visual']
        blockbench = tuple(blockbench_receipts)
        entity_ids = {module.module_id for module in proposal.modules if module.kind in {'entity', 'boss', 'npc'}}

        def blockbench_passed(owner: str) -> bool:
            expected = {owner} if owner in entity_ids else entity_ids
            if not expected:
                return False
            passed = {
                str(receipt.get('entity'))
                for receipt in blockbench
                if CompleteProductionOrchestrator._cached_blockbench_review(receipt)
            }
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
    def _full_gradle_build_receipt_passed(
        build_report: dict[str, Any] | None,
    ) -> bool:
        """Require a successful receipt for the Gradle `build` task itself.

        Optimized validation may label the command `incremental_build` because it
        skips `clean`, but it is still release-grade only when its recorded argv
        proves that the full Gradle `build` task actually ran.
        """
        if not isinstance(build_report, dict) or build_report.get('status') != 'PASS':
            return False
        for command in build_report.get('commands', ()):
            if (
                not isinstance(command, dict)
                or command.get('exit_code') != 0
                or command.get('timed_out') is True
            ):
                continue
            name = str(command.get('name') or '')
            if name in {'build', 'clean_build'}:
                return True
            if name != 'incremental_build':
                continue
            argv = command.get('command')
            if not isinstance(argv, (list, tuple)):
                continue
            if any(str(argument) == 'build' for argument in argv):
                return True
        return False

    @staticmethod
    def _gametest_receipt_passed(build_report: dict[str, Any] | None, spec: Any) -> bool:
        if not isinstance(build_report, dict) or not isinstance(build_report.get('gametest_report'), str):
            return False
        mode = str(build_report.get('gametest_mode') or '').strip()
        if mode == 'integrated_build':
            execution_passed = (
                CompleteProductionOrchestrator._full_gradle_build_receipt_passed(
                    build_report
                )
            )
        else:
            # Backward compatible with older receipts that had no gametest_mode.
            execution_passed = CompleteProductionOrchestrator._command_receipt_passed(
                build_report, 'gametest'
            )
        if not execution_passed:
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
        expected_class = f'{main_class}GameTests'.lower()
        expected_method = 'generatedRegistriesAreLive'.lower()
        expected_combined = f'{expected_class}.{expected_method}'
        expected_method_compact = ''.join(
            character for character in expected_method if character.isalnum()
        )
        mod_id_compact = ''.join(
            character
            for character in str(spec.mod_id).casefold()
            if character.isalnum()
        )

        for testcase in testcases:
            name = str(testcase.attrib.get('name') or '').strip().lower()
            classname = str(testcase.attrib.get('classname') or '').strip().lower()
            if name == expected_combined:
                return True
            if (
                name == expected_method
                and (
                    not classname
                    or classname.rsplit('.', 1)[-1] == expected_class
                )
            ):
                return True

            # Fabric's GameTest JUnit writer may serialize the registered test id
            # (for example mod_id:method_name) instead of the Java class/method pair.
            # Bind that form to both our host-owned method id and this exact mod id.
            name_compact = ''.join(
                character for character in name if character.isalnum()
            )
            if (
                name_compact.endswith(expected_method_compact)
                and mod_id_compact
                and name_compact.startswith(mod_id_compact)
            ):
                return True
        return False

    def _generate_assets(self, router: ModelRouter, proposal: CompleteProposal, project_root: Path, run_root: Path) -> dict[str, Any]:
        return generate_assets(router, proposal, project_root, run_root)

    @staticmethod
    def _blockbench_review(gecko_receipt: dict[str, Any], run_root: Path) -> dict[str, Any]:
        return blockbench_review(gecko_receipt, run_root)

    @staticmethod
    def _run_playtest(
        actions: Iterable[dict[str, Any]],
        acceptance_tests: Iterable[str] = (),
    ) -> dict[str, Any]:
        return run_playtest(actions, acceptance_tests)

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

def _gametest_attestation_status(
    build_report: dict[str, Any] | None,
    spec: Any,
    *,
    requested: bool,
) -> str:
    if not requested:
        return 'NOT_REQUIRED'
    return (
        'PASS'
        if CompleteProductionOrchestrator._gametest_receipt_passed(build_report, spec)
        else 'NO_EVIDENCE'
    )


def _normalize_required_gate(value: str) -> str:
    return ' '.join(''.join(character.casefold() if character.isalnum() else ' ' for character in value).split())
