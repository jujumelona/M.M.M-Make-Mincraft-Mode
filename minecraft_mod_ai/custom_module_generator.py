from __future__ import annotations

from .custom_module_architecture_support import (
    apply_authored_request as _apply_authored_request,
    implementation_phase as _implementation_phase,
    output_exhaustion_continuation_messages as _architecture_continuation_messages,
    task_local_module_contract as _architecture_task_contract,
)
from .filesystem_copy import reflink_or_copy
from .model_response_templates import parse_response_text, response_template_prompt
from .research_validation_fingerprint_performance import content_digest


import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

from .coder_execution_contract import project_task_for_coder
from .complete_spec import ProductionModule
from .host_grounding import (
    build_coder_grounding,
    custom_module_path_allowed,
    custom_module_path_protected,
)
from .llama_finish_reason_contract import OUTPUT_EXHAUSTED, completion_boundary_kind
from .model_context_budget import request_message_budget
from .model_router import ModelRouter
from .platform_catalog import adapter_for_target, adapter_from_project
from .project_index import ProjectIndex
from .research_ledger import select_module_research_context
from .scale_policy import ScalePolicy
from .small_model_atomic_coder_execution import (
    atomic_coder_call,
    bounded_initial_observations,
    bounded_reuse_context,
)
from .small_model_task_capsule_contract import (
    task_capsule_generation_scope,
    task_local_module_contract_owner,
)
from .small_model_write_scope_enforcement import (
    exact_task_operation_validator,
    generation_authority_scoped,
)
from .source_patch import SourcePatchError, TransactionalSourcePatcher
from .target_contract import TargetContractError, validate_target_coordinates


class CustomModuleGenerationError(RuntimeError):
    pass


_APPROVED_REUSE_CONTEXT_SCHEMA = "mmm/approved-reuse-context-v1"
_APPROVED_REUSE_CONTEXT_BYTES = 12 * 1024
_CODE_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{2,}")
_REUSE_BOILERPLATE = frozenset(
    {
        "abstract",
        "boolean",
        "class",
        "default",
        "extends",
        "final",
        "import",
        "implements",
        "interface",
        "package",
        "private",
        "protected",
        "public",
        "record",
        "return",
        "static",
        "string",
        "super",
        "this",
        "throws",
        "void",
    }
)


def _owned_reuse_plan(module: ProductionModule) -> Mapping[str, Any] | None:
    config = module.config if isinstance(module.config, dict) else {}
    value = config.get("_owned_reuse_plan")
    return value if isinstance(value, Mapping) else None


def _source_donor_decisions(
    plan: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(plan, Mapping):
        return ()
    raw = plan.get("capabilities")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(
        item
        for item in raw
        if isinstance(item, Mapping)
        and str(item.get("mode") or "").strip().casefold()
        in {"source_transplant", "adapt"}
        and isinstance(item.get("donor"), Mapping)
    )


@bounded_reuse_context
def _materialize_owned_reuse_context(
    project_root: Path,
    module: ProductionModule,
    *,
    byte_budget: int = _APPROVED_REUSE_CONTEXT_BYTES,
) -> dict[str, Any] | None:
    """Materialize selected donors and return bounded code-bearing coder context.

    Retrieval/selection stays host-owned and evidence-first. The small coder receives
    only the already-selected source slices, not the global reuse plan or another donor
    search problem. ``read_reuse_source`` remains available for bounded pagination.
    """

    plan = _owned_reuse_plan(module)
    decisions = _source_donor_decisions(plan)
    if not decisions:
        return None

    from .production_tools import ProductionToolService
    from .source_transplant import SourceTransplantError, materialize_source_slices

    try:
        materialization = materialize_source_slices(project_root, plan)
    except (OSError, ValueError, SourceTransplantError) as exc:
        raise CustomModuleGenerationError(
            f"Approved reuse donor materialization failed: {type(exc).__name__}: {exc}"
        ) from exc

    donors = materialization.get("donors")
    if (
        not isinstance(donors, list)
        or materialization.get("count") != len(donors)
        or len(donors) != len(decisions)
    ):
        raise CustomModuleGenerationError(
            "Approved reuse plan did not materialize every selected source donor."
        )

    effective_budget = max(1024, int(byte_budget))
    remaining = effective_budget
    snippets: list[dict[str, Any]] = []
    service = ProductionToolService(workspace_root=project_root)
    try:
        for donor in donors:
            files = donor.get("files") if isinstance(donor, Mapping) else None
            if not isinstance(files, list):
                raise CustomModuleGenerationError(
                    "Materialized reuse donor receipt has no authorized files."
                )
            for file_receipt in files:
                if not isinstance(file_receipt, Mapping) or remaining <= 0:
                    continue
                path = str(file_receipt.get("path") or "")
                if not path:
                    continue
                chunk_limit = min(8 * 1024, remaining)
                try:
                    source = service.read_reuse_source(
                        ".",
                        path,
                        limit_bytes=chunk_limit,
                    )
                except (OSError, ValueError) as exc:
                    raise CustomModuleGenerationError(
                        "Approved reuse source could not be read: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                content = str(source.get("content") or "")
                used = len(content.encode("utf-8"))
                remaining = max(0, remaining - used)
                snippets.append(
                    {
                        "repository": source.get("repository"),
                        "commit_sha": source.get("commit_sha"),
                        "license_id": source.get("license_id"),
                        "capability": source.get("capability"),
                        "path": source.get("path"),
                        "sha256": source.get("sha256"),
                        "offset_bytes": source.get("offset_bytes"),
                        "next_offset_bytes": source.get("next_offset_bytes"),
                        "eof": source.get("eof"),
                        "symbols": list(file_receipt.get("symbols") or ()),
                        "content": content,
                    }
                )
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
    finally:
        service.close()

    if not snippets:
        raise CustomModuleGenerationError(
            "Approved reuse donors materialized without any readable source context."
        )
    return {
        "schema_version": _APPROVED_REUSE_CONTEXT_SCHEMA,
        "materialization": materialization,
        "snippets": snippets,
        "byte_budget": effective_budget,
        "bytes_used": effective_budget - remaining,
        "policy": (
            "Adapt only these host-selected, commit-pinned source slices to the exact "
            "task target; preserve license/provenance and never edit donor files."
        ),
    }


def _reuse_code_tokens(value: Any) -> tuple[str, ...]:
    return tuple(
        token
        for token in _CODE_IDENTIFIER.findall(str(value or ""))
        if token.casefold() not in _REUSE_BOILERPLATE
    )


def _reuse_shingles(tokens: Sequence[str], width: int = 5) -> set[tuple[str, ...]]:
    folded = tuple(token.casefold() for token in tokens)
    if len(folded) < width:
        return set()
    return {
        folded[index : index + width]
        for index in range(len(folded) - width + 1)
    }


def _verify_reuse_application(
    context: Mapping[str, Any],
    staged_root: Path,
    touched_paths: Sequence[str],
) -> dict[str, Any]:
    """Prove that generated source actually incorporates the approved donor code."""

    snippets = context.get("snippets")
    if not isinstance(snippets, list) or not snippets:
        raise CustomModuleGenerationError(
            "Approved reuse context has no code snippets to verify after generation."
        )
    donor_tokens: list[str] = []
    declared_symbols: set[str] = set()
    donor_hashes: list[str] = []
    for snippet in snippets:
        if not isinstance(snippet, Mapping):
            continue
        donor_tokens.extend(_reuse_code_tokens(snippet.get("content")))
        declared_symbols.update(
            str(item).strip()
            for item in snippet.get("symbols", ())
            if str(item).strip()
        )
        digest = str(snippet.get("sha256") or "").strip()
        if digest:
            donor_hashes.append(digest)

    target_tokens: list[str] = []
    verified_paths: list[str] = []
    for raw_path in touched_paths:
        normalized = PurePosixPath(str(raw_path).replace("\\", "/")).as_posix()
        target = (staged_root / normalized).resolve()
        try:
            target.relative_to(staged_root.resolve())
        except ValueError:
            continue
        if not target.is_file() or target.is_symlink():
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        target_tokens.extend(_reuse_code_tokens(text))
        verified_paths.append(normalized)

    donor_folded = {token.casefold(): token for token in donor_tokens}
    target_folded = {token.casefold() for token in target_tokens}
    matched_identifiers = sorted(
        donor_folded[key]
        for key in donor_folded.keys() & target_folded
        if len(key) >= 4
    )
    matched_symbols = sorted(
        symbol
        for symbol in declared_symbols
        if symbol.casefold() in target_folded
    )
    shingle_count = len(
        _reuse_shingles(donor_tokens) & _reuse_shingles(target_tokens)
    )
    applied = bool(matched_symbols or shingle_count)
    receipt = {
        "schema_version": "mmm/reuse-application-receipt-v1",
        "status": "APPLIED" if applied else "NOT_APPLIED",
        "donor_sha256": list(dict.fromkeys(donor_hashes)),
        "touched_paths": verified_paths,
        "matched_declared_symbols": matched_symbols,
        "matched_identifiers": matched_identifiers[:64],
        "matched_token_shingles": shingle_count,
        "policy": (
            "At least one declared donor symbol or one five-token donor code shingle "
            "must survive in generated source; loose identifier overlap is diagnostic only."
        ),
    }
    if not applied:
        raise CustomModuleGenerationError(
            "REUSE_NOT_APPLIED: approved donor code was supplied, but generated changes "
            "contain no attributable donor symbol or code structure."
        )
    return receipt


@task_local_module_contract_owner
def _task_local_module_contract(module: ProductionModule) -> dict[str, Any]:
    return _architecture_task_contract(
        module,
        error_type=CustomModuleGenerationError,
        project_task=project_task_for_coder,
    )


_STAGE_IGNORED_DIRS = {".git", ".gradle", ".minecraft_ai", "build", "run"}
_CONTINUATION_PATH_PREVIEW = 64
_CHECKPOINT_DIRECTORY = ".mmm-custom-checkpoints"
_CHECKPOINT_SCHEMA = "mmm/custom-module-checkpoint-v2"
_CHECKPOINT_KEY = __import__("re").compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_STRATEGY_EPOCH = "mmm/custom-candidate-strategy-v1"
_CHECKPOINT_ACTIVE_LOCK = threading.RLock()
_CHECKPOINT_ACTIVE_PATHS: set[Path] = set()
_CHECKPOINT_LEASE_SCOPE = threading.local()
_CHECKPOINT_PERSISTENCE_SCOPE = threading.local()


class _GenerationCheckpointLease:
    """Exclusive in-process ownership of one persistent staged workspace."""

    def __init__(self, checkpoint_root: Path) -> None:
        self.checkpoint_root = checkpoint_root.resolve()
        self._closed = False
        with _CHECKPOINT_ACTIVE_LOCK:
            if self.checkpoint_root in _CHECKPOINT_ACTIVE_PATHS:
                raise CustomModuleGenerationError(
                    "An identical custom-module checkpoint is already active."
                )
            _CHECKPOINT_ACTIVE_PATHS.add(self.checkpoint_root)

    def close(self) -> None:
        with _CHECKPOINT_ACTIVE_LOCK:
            if self._closed:
                return
            _CHECKPOINT_ACTIVE_PATHS.discard(self.checkpoint_root)
            self._closed = True


def _checkpoint_lease_scoped(method):
    @wraps(method)
    def guarded(*args: Any, **kwargs: Any):
        previous = getattr(_CHECKPOINT_LEASE_SCOPE, "leases", None)
        leases: list[_GenerationCheckpointLease] = []
        _CHECKPOINT_LEASE_SCOPE.leases = leases
        try:
            return method(*args, **kwargs)
        finally:
            for lease in reversed(leases):
                lease.close()
            if previous is None:
                try:
                    del _CHECKPOINT_LEASE_SCOPE.leases
                except AttributeError:
                    pass
            else:
                _CHECKPOINT_LEASE_SCOPE.leases = previous

    return guarded


def _track_checkpoint_lease(lease: _GenerationCheckpointLease) -> None:
    leases = getattr(_CHECKPOINT_LEASE_SCOPE, "leases", None)
    if not isinstance(leases, list):
        lease.close()
        raise CustomModuleGenerationError(
            "Checkpoint lease was opened outside the guarded generation scope."
        )
    leases.append(lease)


def _transfer_checkpoint_lease(lease: _GenerationCheckpointLease) -> None:
    leases = getattr(_CHECKPOINT_LEASE_SCOPE, "leases", None)
    if not isinstance(leases, list) or lease not in leases:
        raise CustomModuleGenerationError(
            "Checkpoint lease is not owned by the active generation scope."
        )
    leases.remove(lease)


@contextmanager
def _active_checkpoint_persistence(
    checkpoint_root: Path,
    staged_root: Path,
    identity_sha256: str,
):
    previous = getattr(_CHECKPOINT_PERSISTENCE_SCOPE, "state", None)
    _CHECKPOINT_PERSISTENCE_SCOPE.state = (
        checkpoint_root,
        staged_root.resolve(),
        identity_sha256,
    )
    try:
        yield
    finally:
        if previous is None:
            try:
                del _CHECKPOINT_PERSISTENCE_SCOPE.state
            except AttributeError:
                pass
        else:
            _CHECKPOINT_PERSISTENCE_SCOPE.state = previous


def persist_active_generation_checkpoint(project_root: str | Path) -> bool:
    """Persist the exact active staged workspace after a source transaction."""

    state = getattr(_CHECKPOINT_PERSISTENCE_SCOPE, "state", None)
    if not isinstance(state, tuple) or len(state) != 3:
        return False
    checkpoint_root, staged_root, identity_sha256 = state
    try:
        current_root = Path(project_root).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    if current_root != staged_root:
        return False
    _persist_generation_checkpoint(
        checkpoint_root,
        staged_root,
        identity_sha256=identity_sha256,
    )
    return True


@atomic_coder_call
def _generate_coder_text(
    router: ModelRouter,
    role: str,
    messages: Sequence[Mapping[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> str:
    """Single custom-generation call seam for coder-specific execution policies."""

    return router.generate_text(role, messages, *args, **kwargs)


def _receipt_required_gates(module: ProductionModule) -> list[str]:
    """Use the same approved gate set as the active task capsule and final receipt."""

    from .small_model_task_capsule_contract import current_task_required_gates

    return list(
        dict.fromkeys(
            (
                *tuple(module.required_gates),
                *tuple(current_task_required_gates()),
            )
        )
    )


def _host_finalize_missing_generation_verification(
    staged_root: Path,
    *,
    generation_verification: dict[str, Any] | None,
    touched_paths: Sequence[str],
    required_gates: Sequence[str],
) -> dict[str, Any] | None:
    """Host-own the final compile evidence when a noncanonical router omitted it.

    Existing receipts are never repaired or replaced here. A malformed receipt must
    remain visible to the fail-closed binding contract. Only a genuinely missing
    receipt can be recovered, and only for one exact Java target with a mandatory
    target_compile gate.
    """

    if generation_verification is not None:
        return generation_verification

    normalized_gates = {
        re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
        for value in required_gates
        if str(value).strip()
    }
    java_paths = tuple(
        dict.fromkeys(
            str(path).replace("\\", "/").strip()
            for path in touched_paths
            if str(path).strip().casefold().endswith(".java")
        )
    )
    if "target_compile" not in normalized_gates or len(java_paths) != 1:
        return None

    target_path = java_paths[0]
    has_gradle_model = any(
        (staged_root / relative).is_file()
        for relative in (
            "gradlew",
            "gradlew.bat",
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
        )
    )
    compile_receipt: dict[str, Any]
    if has_gradle_model:
        from .generation_target_compile import run_generation_target_compile

        try:
            compile_receipt = run_generation_target_compile(
                staged_root,
                target_path=target_path,
            )
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            compile_receipt = {
                "status": "UNAVAILABLE",
                "reason": f"{type(exc).__name__}: {exc}",
                "diagnostics": [],
            }
    else:
        compile_receipt = {
            "status": "UNAVAILABLE",
            "reason": "staged workspace has no Gradle build model yet",
            "diagnostics": [],
        }

    compile_status = str(compile_receipt.get("status") or "").strip().upper()
    if compile_status == "FAIL":
        diagnostics = compile_receipt.get("diagnostics")
        raise CustomModuleGenerationError(
            "GENERATION_TARGET_COMPILE_FAILED: host fallback compiler rejected "
            f"{target_path}: {diagnostics!r}"
        )

    if compile_status == "PASS":
        terminal_status = "PASS"
        validation_status = "PASS"
        termination_reason = "VERIFICATION_PASSED"
        downstream_required_gate = None
    else:
        terminal_status = "DEFERRED_TO_TARGET_COMPILE"
        validation_status = "DEFERRED"
        termination_reason = "VERIFICATION_DEFERRED_TO_TARGET_COMPILE"
        downstream_required_gate = "target_compile"

    return {
        "schema_version": "mmm/generation-verification-v1",
        "status": terminal_status,
        "authority": "generation_tool_loop",
        "validation_status": validation_status,
        "termination_reason": termination_reason,
        "verifier_tool": "target_compile",
        "target_path": target_path,
        "compile_backed_java": True,
        "downstream_required_gate": downstream_required_gate,
        "verifier_origin": "custom_module_host_fallback",
        "compile_receipt": compile_receipt,
    }


def _coder_project_context_budget(
    router: ModelRouter,
    policy: ScalePolicy,
    *,
    fast_mode: bool,
) -> int:
    """Bound the first exact-source page; continuation owns additional context."""

    del fast_mode  # Atomic source-page size is mode-independent.
    hard_cap = min(max(1024, int(policy.model_context_bytes)), 4 * 1024)
    registry = getattr(router, "registry", None)
    resolve_role = getattr(registry, "role", None)
    profile = str(getattr(router, "profile", "") or "").strip()
    if not callable(resolve_role) or not profile:
        return hard_cap
    try:
        config = resolve_role(profile, "coder")
        live_request_bytes = int(request_message_budget(config, ()))
    except Exception:
        return hard_cap
    if live_request_bytes <= 0:
        return hard_cap
    return min(hard_cap, max(1024, live_request_bytes // 2))


class CustomModuleGenerator:
    """Implement one approved module through the canonical tool-capable coder loop."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        policy: ScalePolicy | None = None,
        fast_mode: bool = False,
        project_index: ProjectIndex | None = None,
        checkpoint_root: str | Path | None = None,
    ) -> None:
        self.router = router
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()
        self.fast_mode = fast_mode
        self._cached_index: ProjectIndex | None = project_index
        self._cached_root: Path | None = (
            project_index.root if project_index is not None else None
        )
        self._checkpoint_root = (
            Path(checkpoint_root).expanduser() if checkpoint_root is not None else None
        )
        self._checkpoint_cleanup_lock = threading.RLock()
        self._checkpoint_cleanup_tokens: dict[
            str,
            tuple[str, Path, _GenerationCheckpointLease],
        ] = {}

    @task_capsule_generation_scope
    @generation_authority_scoped
    @_checkpoint_lease_scoped
    def generate(
        self,
        project_root: str | Path,
        *,
        module: ProductionModule,
        research_modules: Iterable[ProductionModule] = (),
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
    ) -> dict[str, Any]:
        module.validate(policy=self.policy)
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise CustomModuleGenerationError(
                "Custom module target must be a regular project directory."
            )

        requested_version = str(minecraft_version or "").strip()
        requested_loader = str(loader or "").strip()
        requested_mappings = str(mappings or "").strip()
        if requested_version or requested_loader or requested_mappings:
            if not requested_version or not requested_loader:
                raise CustomModuleGenerationError(
                    "minecraft_version and loader must be supplied together; mappings are required only when the canonical target contract says they are applicable."
                )
            try:
                adapter = adapter_for_target(requested_version, requested_loader)
                coordinates = validate_target_coordinates(
                    requested_version,
                    requested_loader,
                    requested_mappings,
                    declared_mappings_applicable=adapter.mappings_applicable,
                )
            except (ValueError, TargetContractError) as exc:
                raise CustomModuleGenerationError(str(exc)) from exc
            if coordinates.mappings != str(adapter.yarn_mappings or "").strip():
                raise CustomModuleGenerationError(
                    "Requested mappings disagree with the executable provider target."
                )
        else:
            try:
                adapter = adapter_from_project(root)
            except ValueError as exc:
                raise CustomModuleGenerationError(
                    "Custom generation requires an explicit host target or an unambiguous "
                    "existing project platform lock; historical defaults are disabled."
                ) from exc

        minecraft_version = adapter.minecraft_version
        loader = adapter.loader
        mappings = adapter.yarn_mappings
        java_version = adapter.java_version

        module_contract = _task_local_module_contract(module)
        query = json.dumps(
            module_contract,
            ensure_ascii=False,
            sort_keys=True,
        )
        project_context_budget = _coder_project_context_budget(
            self.router,
            self.policy,
            fast_mode=self.fast_mode,
        )

        research_context = select_module_research_context(
            research_modules,
            query=query,
            byte_budget=min(8 * 1024, project_context_budget),
        )
        before = _project_snapshot(root)
        checkpoint_identity = _generation_checkpoint_identity(
            module_query=query,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
            research_context=research_context,
            router=self.router,
        )
        checkpoint_root, staged_root, checkpoint_resumed, checkpoint_lease = (
            _prepare_generation_checkpoint(
                root,
                identity_sha256=checkpoint_identity,
                configured_root=self._checkpoint_root,
            )
        )
        _track_checkpoint_lease(checkpoint_lease)
        if checkpoint_resumed:
            try:
                resumed_operations, _resumed_paths, resumed_discarded = (
                    _collect_staged_operations(root, staged_root, before)
                )
                if resumed_discarded:
                    raise CustomModuleGenerationError(
                        "Resumable custom-module work contains out-of-scope changes."
                    )
                if resumed_operations:
                    self._validate_operations(resumed_operations)
                    self._validate_total_patch_bytes(resumed_operations)
            except (CustomModuleGenerationError, OSError, ValueError):
                _remove_generation_checkpoint(checkpoint_root)
                staged_root = _initialize_generation_checkpoint(
                    root,
                    checkpoint_root,
                    identity_sha256=checkpoint_identity,
                )
                checkpoint_resumed = False

        # Initial coder grounding must describe the same checkpoint workspace used
        # by source mutation and verification. A resumed checkpoint may differ from
        # the live project root, so build all exact-source context from staged_root.
        index = ProjectIndex(staged_root, policy=self.policy)
        observation_ledger: dict[str, Any] | None = None
        last_snapshot_error: ValueError | None = None
        for snapshot_attempt in range(3):
            try:
                observation_ledger = _collect_initial_observations(
                    index,
                    query=query,
                    byte_budget=project_context_budget,
                )
                break
            except ValueError as exc:
                if not _is_stale_project_index_error(exc):
                    raise
                last_snapshot_error = exc
                index = ProjectIndex(staged_root, policy=self.policy)
                print(
                    "custom module: refreshed changing staged ProjectIndex snapshot",
                    f"attempt={snapshot_attempt + 1}/3",
                    flush=True,
                )
        if observation_ledger is None:
            raise CustomModuleGenerationError(
                "Staged project source kept changing while custom-module context was captured; "
                f"last error: {last_snapshot_error}"
            )

        observation_pages = _observation_context_pages(
            observation_ledger,
            query=query,
            byte_budget=project_context_budget,
        )
        host_grounding = build_coder_grounding(
            module_kind=module.kind,
            source_observation_receipt=observation_ledger["receipt"],
            research_context=research_context,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
        )
        from .generation_implementation_grounding import (
            build_generation_implementation_grounding,
        )

        implementation_grounding = build_generation_implementation_grounding(
            module,
            minecraft_version=minecraft_version,
        )
        if implementation_grounding is not None:
            evidence_bindings = dict(host_grounding.get("evidence_bindings") or {})
            evidence_bindings["implementation_contract"] = {
                "receipt": {
                    "selected_fact_count": implementation_grounding["selected_fact_count"],
                    "grounding_sha256": implementation_grounding["grounding_sha256"],
                    "context_id": implementation_grounding["context_id"],
                },
                "grounding": implementation_grounding,
            }
            host_grounding = {
                **host_grounding,
                "evidence_bindings": evidence_bindings,
            }

        approved_reuse_context = _materialize_owned_reuse_context(
            staged_root,
            module,
        )
        if approved_reuse_context is not None:
            evidence_bindings = dict(host_grounding.get("evidence_bindings") or {})
            evidence_bindings["approved_reuse_source"] = {
                "request_field": "approved_reuse_context",
                "receipt": approved_reuse_context["materialization"],
            }
            host_grounding = {
                **host_grounding,
                "evidence_bindings": evidence_bindings,
            }

        if minecraft_version:
            os.environ["MMM_MINECRAFT_VERSION"] = str(minecraft_version).strip()
        if loader:
            os.environ["MMM_LOADER"] = str(loader).strip()
        if mappings:
            os.environ["MMM_YARN_MAPPINGS"] = str(mappings).strip()
        if java_version:
            os.environ["MMM_JAVA_VERSION"] = str(java_version).strip()

        self.router.bind_agent_workspace(staged_root, require_fresh_evidence=True)
        request = {
            "phase": _implementation_phase(module_contract),
            "task": "Implement the approved Minecraft/Fabric mod feature in the current project.",
            "workspace_project_root": ".",
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mappings": mappings,
                "java": java_version,
            },
            "module": module_contract,
            "project_manifest": index.manifest_receipt(),
            "source_observation_receipt": observation_ledger["receipt"],
            "initial_exact_source_context": observation_pages[0],
            "research_context": research_context,
            "host_grounding": host_grounding,
            "checkpoint": {
                "resumed": checkpoint_resumed,
                "source_state_sha256": _mutable_stage_state_sha256(staged_root),
            },
            "rules": [
                "Implement the feature directly; do not return a file-plan protocol.",
                "Use host_grounding.evidence_bindings.implementation_contract first when present; its API symbols and admitted templates are target authority, not examples to rewrite from memory. When a grounding fact supplies required_imports, import those exact fully-qualified owners and never substitute Yarn, intermediary, neighbouring-version, or remembered package names.",
                "When implementation_contract templates expose symbol_usage/topology_policy, preserve each template's exact receiver/member/argument topology while composing code. Do not swap registry roots, keys, identifiers, owners, receivers, or arguments between admitted templates even when Java types appear compatible.",
                "Keep the first implementation minimal: do not invent extra entrypoints, registries, helper classes, creative tabs/groups, logging, or lifecycle hooks unless the approved task explicitly requires them.",
                "Use workspace/RAG/MCP retrieval only when the host implementation grounding and exact project context do not contain a fact required by the approved task.",
                "Apply real edits with the source-edit tool; target compile feedback is handled inside this same generation run before any fallback repair stage.",
                "Fill the final summary in the supplied fixed template.",
                response_template_prompt("coder_summary"),
                "Edits are limited to src/main/java, src/main/resources, src/test/java and src/gametest.",
                "Build infrastructure, Gradle configuration and host-owned ledgers are read-only.",
                "Do not delete files. Preserve valid source already present in a resumed checkpoint.",
                "Use only the selected Minecraft/loader/mappings/Java target and preserve project conventions.",
            ],
        }
        _apply_authored_request(request, module_contract)
        if approved_reuse_context is not None:
            request["approved_reuse_context"] = approved_reuse_context
            request["rules"][2:2] = [
                "Adapt the pinned approved_reuse_context donor snippets before attempting fresh implementation.",
                "Donor files are read-only evidence; write only the exact task-owned target path.",
                "The final source must retain an attributable verified donor symbol or concrete donor code structure; a fresh rewrite is not reuse.",
            ]
        initial_messages = [
            {
                "role": "system",
                "content": (
                    "You are the implementation coder for one approved Minecraft/Fabric module. "
                    "Prefer exact host-owned implementation facts and the current project over memory. "
                    "Write the smallest source change that satisfies the approved task, then let the "
                    "host target compiler verify it in this same generation run. Do not invent extra "
                    "entrypoints, files, lifecycle hooks, or a second patch/file-plan protocol."
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]

        from .progress_aware_tool_loop import (
            clear_generation_verification_receipt,
            current_generation_verification_receipt,
        )

        summary = ""
        generation_verification: dict[str, Any] | None = None
        continuation_count = 0
        clear_generation_verification_receipt()
        try:
            with _active_checkpoint_persistence(
                checkpoint_root,
                staged_root,
                checkpoint_identity,
            ):
                summary = _generate_coder_text(
                    self.router,
                    "coder",
                    initial_messages,
                    response_format="text",
                    tool_stage="generation",
                    enable_tools=True,
                )
                generation_verification = current_generation_verification_receipt()
            stage_tree_sha256, post_generation_snapshot = _stage_tree_snapshot(staged_root)
            _persist_generation_checkpoint(
                checkpoint_root,
                staged_root,
                identity_sha256=checkpoint_identity,
                stage_tree_sha256=stage_tree_sha256,
            )
        except BaseException as exc:
            try:
                _persist_generation_checkpoint(
                    checkpoint_root,
                    staged_root,
                    identity_sha256=checkpoint_identity,
                )
            except (OSError, ValueError) as checkpoint_exc:
                print(
                    "custom module: checkpoint update failed",
                    f"module={module.module_id}",
                    f"error={type(checkpoint_exc).__name__}",
                    flush=True,
                )

            boundary_kind = completion_boundary_kind(exc)
            if boundary_kind != OUTPUT_EXHAUSTED:
                raise

            progress_operations, _progress_paths, _discarded_paths = (
                _collect_staged_operations(root, staged_root, before)
            )
            if progress_operations:
                self._validate_operations(progress_operations)
                self._validate_total_patch_bytes(progress_operations)
            raise CustomModuleGenerationError(
                "ATOMIC_ACTION_OUTPUT_STALLED: the canonical progress-aware coder loop exhausted "
                "its bounded in-state output recovery; refusing an outer continuation because it "
                "would reset HostRunState over an already-mutated staged workspace."
            ) from exc

        summary_text = _parse_coder_summary(summary)

        operations, touched_paths, discarded_paths = _collect_staged_operations(
            root,
            staged_root,
            before,
            after=post_generation_snapshot,
        )
        if not operations:
            discarded = ", ".join(discarded_paths[:8]) if discarded_paths else "none"
            raise CustomModuleGenerationError(
                "Custom-module coding agent produced no valid source/resource changes. "
                f"Discarded out-of-scope staged paths: {discarded}."
            )

        self._validate_operations(operations)
        self._validate_total_patch_bytes(operations)

        required_gates = _receipt_required_gates(module)
        generation_verification = _host_finalize_missing_generation_verification(
            staged_root,
            generation_verification=generation_verification,
            touched_paths=touched_paths,
            required_gates=required_gates,
        )
        if (
            not isinstance(generation_verification, dict)
            or generation_verification.get("schema_version")
            != "mmm/generation-verification-v1"
            or generation_verification.get("authority") != "generation_tool_loop"
            or generation_verification.get("status")
            not in {"PASS", "DEFERRED_TO_TARGET_COMPILE"}
        ):
            raise CustomModuleGenerationError(
                "GENERATION_VERIFICATION_RECEIPT_MISSING: coder/tool loop and host "
                "fallback completed without trustworthy terminal verification evidence."
            )

        from .generation_verification_contract import (
            classify_generation_verification,
        )

        generation_binding = classify_generation_verification(
            source_status="SOURCE_GENERATED",
            receipt=generation_verification,
            touched_paths=touched_paths,
            required_gates=required_gates,
        )
        if int(generation_binding.get("verifier_tier", 0) or 0) <= 0:
            raise CustomModuleGenerationError(
                "GENERATION_VERIFICATION_RECEIPT_INVALID: terminal verifier evidence "
                "does not bind to the generated source mutation: "
                f"status={generation_binding.get('generation_status')}, "
                f"target={generation_binding.get('receipt_target_path')}, "
                f"target_matches={generation_binding.get('receipt_target_matches')}, "
                f"semantics_valid={generation_binding.get('receipt_semantics_valid')}."
            )

        reuse_application_receipt = None
        if approved_reuse_context is not None:
            reuse_application_receipt = _verify_reuse_application(
                approved_reuse_context,
                staged_root,
                touched_paths,
            )
        receipt = TransactionalSourcePatcher(root).apply(operations)
        if self._cached_index is not None:
            try:
                self._cached_index.update_files(touched_paths)
            except (OSError, ValueError):
                self._cached_index = ProjectIndex(root, policy=self.policy)
        else:
            self._cached_index = ProjectIndex(root, policy=self.policy)
        self._cached_root = root
        self._cached_index.write_manifest()
        checkpoint_token = self._register_generation_checkpoint_cleanup(
            identity_sha256=checkpoint_identity,
            checkpoint_root=checkpoint_root,
            checkpoint_lease=checkpoint_lease,
        )
        result = {
            "schema_version": "mmm/custom-module-result-v3",
            "module_id": module.module_id,
            "kind": module.kind,
            "status": "SOURCE_GENERATED",
            "patch_receipt": receipt,
            "operation_count": len(operations),
            "runtime_tests": [
                "Verify approved mod functionality, compilation, and runtime behavior without crash."
            ],
            "source_observation_receipt": observation_ledger["receipt"],
            "touched_paths": touched_paths,
            "discarded_out_of_scope_paths": discarded_paths,
            "agent_summary": summary_text.strip()[:4096],
            "generation_verification": generation_verification,
            "output_exhaustion_continuations": continuation_count,
            "generation_checkpoint_resumed": checkpoint_resumed,
            "generation_checkpoint": {
                "schema_version": _CHECKPOINT_SCHEMA,
                "status": "AWAITING_LIVE_COMMIT",
                "identity_sha256": checkpoint_identity,
                "cleanup_token": checkpoint_token,
            },
            "required_gates": required_gates,
        }
        if reuse_application_receipt is not None:
            result["reuse_application_receipt"] = reuse_application_receipt
        return result

    def _register_generation_checkpoint_cleanup(
        self,
        *,
        identity_sha256: str,
        checkpoint_root: Path,
        checkpoint_lease: _GenerationCheckpointLease,
    ) -> str:
        token = secrets.token_hex(32)
        with self._checkpoint_cleanup_lock:
            self._checkpoint_cleanup_tokens[token] = (
                identity_sha256,
                checkpoint_root,
                checkpoint_lease,
            )
            _transfer_checkpoint_lease(checkpoint_lease)
        return token

    def finalize_committed_generation_checkpoint(
        self,
        result: Any,
        *,
        project_root: str | Path,
    ) -> bool:
        """Clean one checkpoint only after the durable work-node commit succeeded."""

        checkpoint = result.get("generation_checkpoint") if isinstance(result, dict) else None
        if not isinstance(checkpoint, dict):
            return True
        if checkpoint.get("status") == "CLEANED_AFTER_LIVE_COMMIT":
            return True
        token = checkpoint.get("cleanup_token")
        with self._checkpoint_cleanup_lock:
            owned = (
                self._checkpoint_cleanup_tokens.get(token)
                if isinstance(token, str)
                else None
            )
        if owned is not None:
            return self.acknowledge_generation_checkpoint(result)
        return finalize_persisted_generation_checkpoint(
            result,
            project_root=project_root,
            checkpoint_root=self._checkpoint_root,
        )

    def acknowledge_generation_checkpoint(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        checkpoint = result.get("generation_checkpoint")
        if not isinstance(checkpoint, dict):
            return False
        if checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA:
            return False
        if checkpoint.get("status") == "CLEANED_AFTER_LIVE_COMMIT":
            return "cleanup_token" not in checkpoint
        if checkpoint.get("status") != "AWAITING_LIVE_COMMIT":
            return False
        token = checkpoint.get("cleanup_token")
        identity = checkpoint.get("identity_sha256")
        if not isinstance(token, str) or not _CHECKPOINT_KEY.fullmatch(token) or not isinstance(identity, str):
            return False
        with self._checkpoint_cleanup_lock:
            owned = self._checkpoint_cleanup_tokens.get(token)
            if owned is None or owned[0] != identity:
                checkpoint["status"] = "UNACKNOWLEDGED_AFTER_LIVE_COMMIT"
                checkpoint.pop("cleanup_token", None)
                return False
            try:
                _remove_generation_checkpoint(owned[1])
            except (CustomModuleGenerationError, OSError):
                self._checkpoint_cleanup_tokens.pop(token, None)
                owned[2].close()
                checkpoint["status"] = "PRESERVED_AFTER_CLEANUP_FAILURE"
                checkpoint.pop("cleanup_token", None)
                return False
            self._checkpoint_cleanup_tokens.pop(token, None)
            owned[2].close()
        checkpoint["status"] = "CLEANED_AFTER_LIVE_COMMIT"
        checkpoint.pop("cleanup_token", None)
        return True

    def release_generation_checkpoint(self, result: Any) -> bool:
        return self._finish_generation_checkpoint(result, delete=False)

    def discard_generation_checkpoint(self, result: Any) -> bool:
        return self._finish_generation_checkpoint(result, delete=True)

    def _finish_generation_checkpoint(self, result: Any, *, delete: bool) -> bool:
        if not isinstance(result, dict):
            return False
        checkpoint = result.get("generation_checkpoint")
        if not isinstance(checkpoint, dict):
            return False
        if (
            checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA
            or checkpoint.get("status") != "AWAITING_LIVE_COMMIT"
        ):
            return False
        token = checkpoint.get("cleanup_token")
        identity = checkpoint.get("identity_sha256")
        if not isinstance(token, str) or not _CHECKPOINT_KEY.fullmatch(token) or not isinstance(identity, str):
            checkpoint.pop("cleanup_token", None)
            return False
        with self._checkpoint_cleanup_lock:
            owned = self._checkpoint_cleanup_tokens.get(token)
            if owned is None or owned[0] != identity:
                checkpoint["status"] = "UNOWNED_LOSER_CHECKPOINT"
                checkpoint.pop("cleanup_token", None)
                return False
            self._checkpoint_cleanup_tokens.pop(token, None)
            removed = False
            try:
                if delete:
                    _remove_generation_checkpoint(owned[1])
                    checkpoint["status"] = "DISCARDED_AFTER_OTHER_WINNER"
                    removed = True
                else:
                    checkpoint["status"] = "PRESERVED_FOR_RESUME"
                    removed = True
            except (CustomModuleGenerationError, OSError):
                checkpoint["status"] = "PRESERVED_AFTER_CLEANUP_FAILURE"
            finally:
                owned[2].close()
                checkpoint.pop("cleanup_token", None)
        return removed

    @exact_task_operation_validator(CustomModuleGenerationError)
    def _validate_operations(self, operations: list[dict[str, Any]]) -> None:
        for item in operations:
            if not isinstance(item, dict):
                raise CustomModuleGenerationError("Patch operation must be an object.")
            if item.get("operation") not in {"create", "replace", "edit"}:
                raise CustomModuleGenerationError("Custom module may not delete files.")
            path = _normalized_operation_path(item)
            if custom_module_path_protected(path) or not _agent_mutable_path(path):
                raise CustomModuleGenerationError(
                    f"Custom module path is outside the source/resource scope: {path}"
                )

    def _validate_total_patch_bytes(self, operations: list[dict[str, Any]]) -> None:
        size = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if size > self.policy.max_patch_bytes:
            raise CustomModuleGenerationError(
                "Custom module patch exceeds MMM_MAX_PATCH_BYTES; split the feature or raise explicit host policy."
            )


_agent_mutable_path = custom_module_path_allowed


def _stage_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _STAGE_IGNORED_DIRS}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _checkpoint_router_scope(router: Any) -> dict[str, Any]:
    current = router
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidate_index = getattr(current, "_candidate_index", None)
        strategy = getattr(current, "_strategy", None)
        candidate_count = getattr(current, "_count", None)
        candidate_fields = (candidate_index, strategy, candidate_count)
        if any(value is not None for value in candidate_fields):
            if not (
                type(candidate_index) is int
                and isinstance(strategy, str)
                and strategy.strip()
                and type(candidate_count) is int
                and candidate_count >= 1
                and 0 <= candidate_index < candidate_count
            ):
                raise CustomModuleGenerationError(
                    "Candidate checkpoint identity requires valid index/count/strategy."
                )
            return {
                "mode": "candidate",
                "strategy_epoch": _CHECKPOINT_STRATEGY_EPOCH,
                "candidate_index": candidate_index,
                "candidate_count": candidate_count,
                "strategy": strategy.strip(),
            }
        current = getattr(current, "_router", None)
    return {"mode": "single", "strategy_epoch": _CHECKPOINT_STRATEGY_EPOCH}


def _stage_tree_snapshot(root: Path) -> tuple[str, dict[str, str]]:
    """Hash checkpoint structure and file contents in one filesystem traversal."""

    rows: list[tuple[str, str, str]] = []
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _STAGE_IGNORED_DIRS for part in relative.parts):
            continue
        normalized = relative.as_posix()
        if path.is_symlink():
            rows.append((normalized, "symlink", str(path.readlink())))
        elif path.is_file():
            digest = "sha256:" + content_digest(path).hex()
            rows.append((normalized, "file", digest))
            files[normalized] = digest
        elif path.is_dir():
            rows.append((normalized, "directory", ""))
    return _sha256_json(rows), files


def _checkpoint_tree_state_sha256(root: Path) -> str:
    return _stage_tree_snapshot(root)[0]


def _generation_checkpoint_identity(
    *,
    module_query: str,
    minecraft_version: str,
    loader: str,
    mappings: str,
    research_context: Any,
    router: Any,
) -> str:
    return _sha256_json(
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "module_query_sha256": _sha256_json(module_query),
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mappings": mappings,
            },
            "research_context_sha256": _sha256_json(research_context),
            "router_scope": _checkpoint_router_scope(router),
        }
    )


def _checkpoint_key(identity_sha256: str) -> str:
    prefix, separator, digest = str(identity_sha256).partition(":")
    if prefix != "sha256" or separator != ":" or not _CHECKPOINT_KEY.fullmatch(digest):
        raise ValueError("Custom-module checkpoint identity must be a SHA-256 receipt")
    return digest


def _checkpoint_manifest(checkpoint_root: Path) -> Path:
    return checkpoint_root / "checkpoint.json"


def _checkpoint_base(checkpoint_root: Path) -> Path:
    return checkpoint_root / "base"


def _checkpoint_directory(root: Path, configured_root: Path | None = None) -> Path:
    base = configured_root if configured_root is not None else root.parent / _CHECKPOINT_DIRECTORY
    if base.parent.exists() and base.parent.is_symlink():
        raise CustomModuleGenerationError("Custom-module checkpoint parent may not be a symlink.")
    if base.exists() and (base.is_symlink() or not base.is_dir()):
        raise CustomModuleGenerationError("Custom-module checkpoint root must be a regular host directory.")
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _safe_checkpoint_path(base: Path, key: str) -> Path:
    if not _CHECKPOINT_KEY.fullmatch(key):
        raise CustomModuleGenerationError("Unsafe custom-module checkpoint key")
    checkpoint_root = base / key
    try:
        checkpoint_root.resolve().relative_to(base)
    except ValueError as exc:
        raise CustomModuleGenerationError("Custom-module checkpoint escaped its host-owned root.") from exc
    return checkpoint_root


def _remove_generation_checkpoint(checkpoint_root: Path) -> None:
    declared_base = checkpoint_root.parent
    if declared_base.is_symlink() or not declared_base.is_dir():
        raise CustomModuleGenerationError("Refusing to remove checkpoint through unsafe host root.")
    base = declared_base.resolve()
    if base != declared_base or base.name != _CHECKPOINT_DIRECTORY:
        raise CustomModuleGenerationError("Refusing to remove unrecognized checkpoint path.")
    if not _CHECKPOINT_KEY.fullmatch(checkpoint_root.name) or checkpoint_root.is_symlink():
        raise CustomModuleGenerationError("Refusing to remove unsafe checkpoint path.")
    if checkpoint_root.exists():
        checkpoint_root.resolve().relative_to(base)
        shutil.rmtree(checkpoint_root)
    try:
        base.rmdir()
    except OSError:
        pass


def _persist_generation_checkpoint(
    checkpoint_root: Path,
    staged_root: Path,
    *,
    identity_sha256: str,
    stage_tree_sha256: str | None = None,
) -> None:
    checkpoint_stat = checkpoint_root.lstat()
    staged_stat = staged_root.lstat()
    base_root = _checkpoint_base(checkpoint_root)
    base_stat = base_root.lstat()
    if not all(stat.S_ISDIR(item.st_mode) for item in (checkpoint_stat, staged_stat, base_stat)):
        raise ValueError("Custom-module checkpoint staging root is unsafe")
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "identity_sha256": identity_sha256,
        "base_tree_sha256": _checkpoint_tree_state_sha256(base_root),
        "stage_tree_sha256": (
            stage_tree_sha256
            if stage_tree_sha256 is not None
            else _checkpoint_tree_state_sha256(staged_root)
        ),
    }
    manifest = _checkpoint_manifest(checkpoint_root)
    if manifest.is_symlink():
        raise ValueError("Custom-module checkpoint manifest may not be a symlink")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=checkpoint_root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if checkpoint_root.is_symlink() or manifest.is_symlink():
            raise ValueError("Custom-module checkpoint changed during persistence")
        os.replace(temporary_name, manifest)
        temporary_name = ""
        try:
            directory_fd = os.open(checkpoint_root, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _read_generation_checkpoint_manifest(checkpoint_root: Path) -> dict[str, Any]:
    manifest = _checkpoint_manifest(checkpoint_root)
    if manifest.is_symlink():
        raise ValueError("Custom-module checkpoint manifest may not be a symlink")
    with manifest.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Custom-module checkpoint manifest must be an object")
    return raw


def _initialize_generation_checkpoint(root: Path, checkpoint_root: Path, *, identity_sha256: str) -> Path:
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=False, exist_ok=False)
    base_root = _checkpoint_base(checkpoint_root)
    staged_root = checkpoint_root / "project"
    try:
        shutil.copytree(root, base_root, symlinks=True, ignore=_stage_ignore, copy_function=reflink_or_copy)
        shutil.copytree(base_root, staged_root, symlinks=True, copy_function=reflink_or_copy)
        _persist_generation_checkpoint(checkpoint_root, staged_root, identity_sha256=identity_sha256)
    except BaseException:
        if checkpoint_root.exists() and not checkpoint_root.is_symlink():
            shutil.rmtree(checkpoint_root, ignore_errors=True)
        raise
    return staged_root


def _checkpoint_patch_operations(base_root: Path, staged_root: Path) -> list[dict[str, Any]]:
    operations, _touched, discarded = _collect_staged_operations(
        base_root, staged_root, _project_snapshot(base_root)
    )
    if discarded:
        raise CustomModuleGenerationError("Resumable checkpoint contains out-of-scope source changes.")
    return operations


def _rebase_generation_checkpoint(root: Path, checkpoint_root: Path, *, identity_sha256: str) -> Path:
    base_root = _checkpoint_base(checkpoint_root)
    staged_root = checkpoint_root / "project"
    operations = _checkpoint_patch_operations(base_root, staged_root)
    next_base = Path(tempfile.mkdtemp(prefix=".base-rebase-", dir=checkpoint_root))
    next_stage = Path(tempfile.mkdtemp(prefix=".project-rebase-", dir=checkpoint_root))
    next_base.rmdir()
    next_stage.rmdir()
    try:
        shutil.copytree(root, next_base, symlinks=True, ignore=_stage_ignore, copy_function=reflink_or_copy)
        shutil.copytree(next_base, next_stage, symlinks=True, copy_function=reflink_or_copy)
        if operations:
            TransactionalSourcePatcher(next_stage).apply(operations)
        shutil.rmtree(base_root)
        shutil.rmtree(staged_root)
        os.replace(next_base, base_root)
        os.replace(next_stage, staged_root)
        _persist_generation_checkpoint(checkpoint_root, staged_root, identity_sha256=identity_sha256)
        return staged_root
    finally:
        for temporary in (next_base, next_stage):
            if temporary.exists() and not temporary.is_symlink():
                shutil.rmtree(temporary, ignore_errors=True)


def _prepare_generation_checkpoint(
    root: Path,
    *,
    identity_sha256: str,
    configured_root: Path | None = None,
) -> tuple[Path, Path, bool, _GenerationCheckpointLease]:
    base = _checkpoint_directory(root, configured_root)
    checkpoint_root = _safe_checkpoint_path(base, _checkpoint_key(identity_sha256))
    lease = _GenerationCheckpointLease(checkpoint_root)
    try:
        staged_root = checkpoint_root / "project"
        if checkpoint_root.exists():
            reusable = False
            try:
                raw = _read_generation_checkpoint_manifest(checkpoint_root)
                base_root = _checkpoint_base(checkpoint_root)
                reusable = (
                    raw.get("schema_version") == _CHECKPOINT_SCHEMA
                    and raw.get("identity_sha256") == identity_sha256
                    and base_root.is_dir()
                    and not base_root.is_symlink()
                    and staged_root.is_dir()
                    and not staged_root.is_symlink()
                    and raw.get("base_tree_sha256") == _checkpoint_tree_state_sha256(base_root)
                    and raw.get("stage_tree_sha256") == _checkpoint_tree_state_sha256(staged_root)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                reusable = False
            if reusable and raw.get("base_tree_sha256") != _checkpoint_tree_state_sha256(root):
                try:
                    staged_root = _rebase_generation_checkpoint(
                        root, checkpoint_root, identity_sha256=identity_sha256
                    )
                except (CustomModuleGenerationError, OSError, SourcePatchError, ValueError):
                    reusable = False
            if reusable:
                return checkpoint_root, staged_root, True, lease
            _remove_generation_checkpoint(checkpoint_root)
        staged_root = _initialize_generation_checkpoint(
            root, checkpoint_root, identity_sha256=identity_sha256
        )
    except BaseException:
        lease.close()
        raise
    return checkpoint_root, staged_root, False, lease


def _committed_patch_receipt_matches(
    result: Any,
    *,
    project_root: Path,
) -> bool:
    if not isinstance(result, Mapping):
        return False
    patch = result.get("patch_receipt")
    if not isinstance(patch, Mapping) or patch.get("status") not in {"APPLIED", "UNCHANGED"}:
        return False
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        return False
    root = project_root.expanduser().resolve()
    for item in operations:
        if not isinstance(item, Mapping):
            return False
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return False
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        expected = item.get("after_sha256")
        if expected is None:
            if candidate.exists():
                return False
            continue
        if (
            not isinstance(expected, str)
            or not candidate.is_file()
            or candidate.is_symlink()
            or ("sha256:" + content_digest(candidate).hex()) != expected
        ):
            return False
    return True


def finalize_persisted_generation_checkpoint(
    result: Any,
    *,
    project_root: str | Path,
    checkpoint_root: str | Path | None = None,
) -> bool:
    """Recover cleanup after ledger commit survived but in-memory ownership did not."""

    if not isinstance(result, dict):
        return False
    checkpoint = result.get("generation_checkpoint")
    if not isinstance(checkpoint, dict):
        return True
    if checkpoint.get("status") == "CLEANED_AFTER_LIVE_COMMIT":
        return True
    if (
        checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "AWAITING_LIVE_COMMIT"
    ):
        return False
    identity = checkpoint.get("identity_sha256")
    if not isinstance(identity, str):
        return False
    root = Path(project_root).expanduser().resolve()
    configured = (
        Path(checkpoint_root).expanduser()
        if checkpoint_root is not None
        else None
    )
    base = _checkpoint_directory(root, configured)
    path = _safe_checkpoint_path(base, _checkpoint_key(identity))
    resolved = path.resolve()
    with _CHECKPOINT_ACTIVE_LOCK:
        if resolved in _CHECKPOINT_ACTIVE_PATHS:
            return False

    # The durable work-node receipt can outlive the ephemeral checkpoint.  A later
    # build-repair pass may legitimately change the generated file after cleanup, so
    # only compare the old patch digest when an orphan checkpoint still exists.
    if path.exists():
        if not _committed_patch_receipt_matches(result, project_root=root):
            return False
        try:
            manifest = _read_generation_checkpoint_manifest(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if (
            manifest.get("schema_version") != _CHECKPOINT_SCHEMA
            or manifest.get("identity_sha256") != identity
        ):
            return False
        _remove_generation_checkpoint(path)
    checkpoint["status"] = "CLEANED_AFTER_LIVE_COMMIT"
    checkpoint.pop("cleanup_token", None)
    return True


def _project_snapshot(root: Path) -> dict[str, str]:
    return _stage_tree_snapshot(root)[1]


def _mutable_stage_state_sha256(staged_root: Path) -> str:
    snapshot = {
        path: digest
        for path, digest in _project_snapshot(staged_root).items()
        if _agent_mutable_path(path) and not custom_module_path_protected(path)
    }
    return _sha256_json(snapshot)


def _output_exhaustion_continuation_messages(
    *,
    module: ProductionModule,
    minecraft_version: str,
    loader: str,
    mappings: str,
    java_version: int,
    continuation_index: int,
    state_sha256: str,
    touched_paths: Iterable[str],
    discarded_paths: Iterable[str],
    source_observation_receipt: Mapping[str, Any] | None = None,
    host_grounding: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    return _architecture_continuation_messages(
        {
            "module": module,
            "minecraft_version": minecraft_version,
            "loader": loader,
            "mappings": mappings,
            "java_version": java_version,
            "continuation_index": continuation_index,
            "state_sha256": state_sha256,
            "touched_paths": touched_paths,
            "discarded_paths": discarded_paths,
            "source_observation_receipt": source_observation_receipt,
            "host_grounding": host_grounding,
        },
        task_contract=_task_local_module_contract,
        continuation_path_preview=_CONTINUATION_PATH_PREVIEW,
    )


def _collect_staged_operations(
    original_root: Path,
    staged_root: Path,
    before: dict[str, str],
    *,
    after: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if after is None:
        after = _project_snapshot(staged_root)
    operations: list[dict[str, Any]] = []
    touched: list[str] = []
    discarded: list[str] = []
    for path in sorted(set(before) | set(after)):
        old_sha = before.get(path)
        new_sha = after.get(path)
        if old_sha == new_sha:
            continue
        if not _agent_mutable_path(path) or custom_module_path_protected(path):
            discarded.append(path)
            continue
        if new_sha is None:
            raise CustomModuleGenerationError(
                f"Custom-module coding agent may not delete source/resource files: {path}"
            )
        target = staged_root / Path(path)
        if target.is_symlink() or not target.is_file():
            raise CustomModuleGenerationError(
                f"Custom-module staged target is not a regular file: {path}"
            )
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CustomModuleGenerationError(
                f"Custom-module generated file must be UTF-8 text: {path}: {exc}"
            ) from exc
        if old_sha is None:
            operation = {"operation": "create", "path": path, "content": content}
        else:
            operation = {
                "operation": "replace",
                "path": path,
                "expected_sha256": old_sha,
                "content": content,
            }
        operations.append(operation)
        touched.append(path)
    return operations, touched, discarded


def _parse_coder_summary(text: str) -> str:
    """Parse the fixed coder-summary contract through the shared schema authority."""

    try:
        payload = parse_response_text("coder_summary", text)
    except ValueError as exc:
        raise CustomModuleGenerationError(str(exc)) from exc
    summary = payload["summary"]
    if not isinstance(summary, str):
        raise CustomModuleGenerationError(
            "Coder summary must be a string in the fixed JSON template."
        )
    return summary


def _is_stale_project_index_error(exc: ValueError) -> bool:
    return str(exc).startswith("Project source changed after its context index was built:")


@bounded_initial_observations
def _collect_initial_observations(
    index: ProjectIndex,
    *,
    query: str,
    byte_budget: int,
    diagnostic_paths: Iterable[str] = (),
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    record_keys: set[tuple[str, int, int]] = set()
    source_page_digest = hashlib.sha256()
    cursor = ""
    seen_cursors: set[str] = set()
    page_count = 0
    project_sha256 = ""
    query_sha256 = ""
    while True:
        page = index.select_page(
            query=query,
            diagnostic_paths=diagnostic_paths,
            byte_budget=byte_budget,
            cursor=cursor,
        )
        if _json_size(page) > byte_budget:
            raise CustomModuleGenerationError("Host project context page exceeded its byte budget.")
        current_project_sha256 = str(page["project_sha256"])
        current_query_sha256 = str(page["query_sha256"])
        if page_count == 0:
            project_sha256 = current_project_sha256
            query_sha256 = current_query_sha256
        elif current_project_sha256 != project_sha256 or current_query_sha256 != query_sha256:
            raise CustomModuleGenerationError("Project context pagination changed its bound identity.")
        page_commitment = {
            "page_index": page["page_index"],
            "project_sha256": current_project_sha256,
            "query_sha256": current_query_sha256,
            "start_position": page["start_position"],
            "start_offset": page["start_offset"],
            "next_cursor": page["next_cursor"],
            "files": [
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "content_start_bytes": item["content_start_bytes"],
                    "content_end_bytes": item["content_end_bytes"],
                }
                for item in page["files"]
            ],
        }
        _update_digest(source_page_digest, page_commitment)
        for item in page.get("files", []):
            if isinstance(item, dict) and "path" in item and ("content" in item or "text" in item):
                content_str = str(item.get("content", item.get("text", "")))
                _append_observation(
                    records,
                    record_keys,
                    _exact_observation(
                        path=str(item["path"]),
                        sha256=str(item.get("sha256", "")),
                        start=int(item.get("content_start_bytes", 0)),
                        content=content_str.encode("utf-8"),
                        source_page=int(page.get("page_index", page_count)),
                    ),
                )
        page_count += 1
        if bool(page.get("complete", False)):
            break
        next_cursor = str(page.get("next_cursor", "")).strip()
        if not next_cursor or next_cursor == cursor or next_cursor in seen_cursors:
            raise CustomModuleGenerationError("Project context pagination made no progress.")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    observation_digest = hashlib.sha256()
    for record in records:
        _update_digest(observation_digest, record)
    receipt = {
        "schema_version": "mmm/source-observation-receipt-v1",
        "project_sha256": project_sha256,
        "query_sha256": query_sha256,
        "source_page_count": page_count,
        "observation_count": len(records),
        "source_pages_sha256": "sha256:" + source_page_digest.hexdigest(),
        "observations_sha256": "sha256:" + observation_digest.hexdigest(),
        "policy": {"exact_source_quotes": True, "path_sha256_byte_range_bound": True},
    }
    return {"schema_version": "mmm/source-observation-ledger-v1", "receipt": receipt, "records": records}


_MIN_OBSERVATION_FRAGMENT_BYTES = 128
_OBSERVATION_PAGE_RESERVE_BYTES = 128


def _utf8_fragments(text: str, max_bytes: int) -> tuple[tuple[int, bytes], ...]:
    """Split UTF-8 text at code-point boundaries while retaining byte offsets."""

    limit = max(_MIN_OBSERVATION_FRAGMENT_BYTES, int(max_bytes))
    fragments: list[tuple[int, bytes]] = []
    chars: list[str] = []
    size = 0
    start = 0
    for char in text:
        encoded = char.encode("utf-8")
        if chars and size + len(encoded) > limit:
            payload = "".join(chars).encode("utf-8")
            fragments.append((start, payload))
            start += len(payload)
            chars = []
            size = 0
        chars.append(char)
        size += len(encoded)
    if chars:
        fragments.append((start, "".join(chars).encode("utf-8")))
    return tuple(fragments)


def _split_observation_records(
    records: list[dict[str, Any]],
    byte_budget: int,
) -> list[dict[str, Any]]:
    """Make every exact-source record small enough to coexist with page metadata."""

    fragment_budget = max(
        _MIN_OBSERVATION_FRAGMENT_BYTES,
        min(1024, byte_budget // 5),
    )
    result: list[dict[str, Any]] = []
    for record in records:
        if _json_size(record) <= fragment_budget + 512:
            result.append(record)
            continue
        text = str(record.get("text", ""))
        base_start = int(record.get("content_start_bytes", 0) or 0)
        for relative_start, payload in _utf8_fragments(text, fragment_budget):
            result.append(
                _exact_observation(
                    path=str(record.get("path", "")),
                    sha256=str(record.get("sha256", "")),
                    start=base_start + relative_start,
                    content=payload,
                    source_page=int(record.get("source_page_index", 0) or 0),
                )
            )
    return result


def _observation_context_pages(
    ledger: dict[str, Any],
    *,
    query: str,
    byte_budget: int,
) -> tuple[dict[str, Any], ...]:
    """Build provenance-preserving source pages that are strictly byte bounded."""

    if type(byte_budget) is not int or byte_budget < 1024:
        raise CustomModuleGenerationError(
            "Source-observation byte budget must be an integer >= 1024."
        )

    records = _split_observation_records(list(ledger["records"]), byte_budget)
    query_tokens = _query_tokens(query)
    ranked = sorted(
        records,
        key=lambda record: (
            -_observation_score(record, query_tokens),
            record["path"],
            record["content_start_bytes"],
            record["observation_id"],
        ),
    )

    safe_budget = max(1024, byte_budget - _OBSERVATION_PAGE_RESERVE_BYTES)
    anchors: list[dict[str, Any]] = []
    anchor_target = max(1024, safe_budget * 3 // 5)
    for record in ranked:
        candidate = _observation_page_payload(
            receipt=ledger["receipt"],
            page_index=0,
            page_count=999999,
            anchors=[*anchors, record],
            records=[],
            complete=False,
        )
        if _json_size(candidate) <= anchor_target:
            anchors.append(record)

    if ranked and not anchors:
        candidate = _observation_page_payload(
            receipt=ledger["receipt"],
            page_index=0,
            page_count=999999,
            anchors=[ranked[0]],
            records=[],
            complete=False,
        )
        if _json_size(candidate) > safe_budget:
            raise CustomModuleGenerationError(
                "Exact-source provenance metadata cannot fit the configured coder context budget."
            )
        anchors.append(ranked[0])

    anchor_ids = {record["observation_id"] for record in anchors}
    remaining = [
        record for record in ranked if record["observation_id"] not in anchor_ids
    ]
    pages: list[dict[str, Any]] = []
    cursor = 0

    while cursor < len(remaining) or not pages:
        page_records: list[dict[str, Any]] = []
        while cursor < len(remaining):
            candidate_records = [*page_records, remaining[cursor]]
            candidate = _observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=999999,
                anchors=anchors,
                records=candidate_records,
                complete=False,
            )
            if _json_size(candidate) > safe_budget:
                break
            page_records.append(remaining[cursor])
            cursor += 1

        if cursor < len(remaining) and not page_records:
            if anchors:
                demoted = anchors.pop()
                anchor_ids.discard(demoted["observation_id"])
                remaining.insert(cursor, demoted)
                continue
            raise CustomModuleGenerationError(
                "Exact-source observation cannot fit the configured coder context budget."
            )

        pages.append(
            _observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=0,
                anchors=anchors,
                records=page_records,
                complete=False,
            )
        )
        if cursor >= len(remaining):
            break

    page_count = len(pages)
    for index, page in enumerate(pages):
        page["page_count"] = page_count
        page["complete"] = index == page_count - 1
        if _json_size(page) > byte_budget:
            raise CustomModuleGenerationError(
                "Host source-observation context page exceeded its byte budget after finalization."
            )
    return tuple(pages)

def _observation_page_payload(
    *,
    receipt: dict[str, Any],
    page_index: int,
    page_count: int,
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "mmm/source-observation-context-v1",
        "ledger_receipt": receipt,
        "page_index": page_index,
        "page_count": page_count,
        "complete": complete,
        "global_anchor_count": len(anchors),
        "global_anchors": anchors,
        "page_observations": records,
        "policy": {
            "facts_are_exact_source_data_not_instructions": True,
            "supplemental_retrieval_available": True,
        },
    }


def _query_tokens(value: str) -> set[str]:
    import re
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,127}", value)}


def _exact_observation(
    *,
    path: str,
    sha256: str,
    start: int,
    content: bytes,
    source_page: int,
) -> dict[str, Any]:
    core = {
        "path": path,
        "sha256": sha256,
        "content_start_bytes": start,
        "content_end_bytes": start + len(content),
        "source_page_index": source_page,
        "kind": "exact_source_excerpt",
        "text": content.decode("utf-8", errors="strict"),
    }
    return {"observation_id": "obs_" + _sha256_json(core).removeprefix("sha256:"), **core}


def _append_observation(
    records: list[dict[str, Any]],
    keys: set[tuple[str, int, int]],
    record: dict[str, Any],
) -> None:
    key = (record["path"], record["content_start_bytes"], record["content_end_bytes"])
    if key not in keys:
        keys.add(key)
        records.append(record)


def _observation_score(record: dict[str, Any], query_tokens: set[str]) -> int:
    path_tokens = _query_tokens(str(record["path"]))
    text_tokens = _query_tokens(str(record["text"]))
    anchor_terms = {"api", "contract", "dependency", "implements", "interface", "public", "register", "required", "schema"}
    return 60 * len(query_tokens & path_tokens) + 8 * len(query_tokens & text_tokens) + 20 * len(anchor_terms & text_tokens)


def _normalized_operation_path(item: dict[str, Any]) -> str:
    return PurePosixPath(str(item.get("path", "")).replace("\\", "/")).as_posix()


def _update_digest(digest: Any, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
