from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

from .complete_spec import ProductionModule
from .host_grounding import build_coder_grounding, custom_module_path_protected
from .llama_finish_reason_contract import OUTPUT_EXHAUSTED, completion_boundary_kind
from .model_context_budget import request_message_budget
from .model_router import ModelRouter
from .platform_catalog import adapter_for_target, adapter_from_project
from .project_index import ProjectIndex
from .research_ledger import select_module_research_context
from .scale_policy import ScalePolicy
from .source_patch import SourcePatchError, TransactionalSourcePatcher


class CustomModuleGenerationError(RuntimeError):
    pass


def _task_local_module_contract(module: ProductionModule) -> dict[str, Any]:
    """Project one immutable semantic work item into the coder request.

    Evidence-first planning already owns global proposal state.  Generation receives the
    task-local contract (CodePlan-style per-edit authority) rather than replaying the
    complete module/proposal configuration into every model turn.
    """

    config = module.config if isinstance(module.config, dict) else {}
    evidence_task = config.get("evidence_task")
    if isinstance(evidence_task, dict):
        return {
            "module_id": module.module_id,
            "kind": module.kind,
            "evidence_task": dict(evidence_task),
            "depends_on": list(module.depends_on),
            "required_gates": list(module.required_gates),
        }
    # Compatibility for callers that have not yet been compiled through the
    # evidence-first semantic work graph.  No fields are silently discarded.
    return {
        "module_id": module.module_id,
        "kind": module.kind,
        "config": config,
        "depends_on": list(module.depends_on),
        "required_gates": list(module.required_gates),
    }


_AGENT_MUTABLE_PREFIXES = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
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


def _coder_project_context_budget(
    router: ModelRouter,
    policy: ScalePolicy,
    *,
    fast_mode: bool,
) -> int:
    """Bound source grounding by the live request capacity, not model capability."""

    hard_cap = max(1024, int(policy.model_context_bytes))
    if fast_mode:
        return min(hard_cap, 4 * 1024)
    fallback = min(hard_cap, 12 * 1024)
    registry = getattr(router, "registry", None)
    resolve_role = getattr(registry, "role", None)
    profile = str(getattr(router, "profile", "") or "").strip()
    if not callable(resolve_role) or not profile:
        return fallback
    try:
        config = resolve_role(profile, "coder")
        live_request_bytes = int(request_message_budget(config, ()))
    except Exception:
        return fallback
    if live_request_bytes <= 0:
        return fallback
    # Exact-source grounding is only one component of the request. The model can
    # retrieve more source from the host index after the first turn.
    return min(hard_cap, max(1024, live_request_bytes // 2))


class CustomModuleGenerator:
    """Implement one approved module through the canonical tool-capable coder loop.

    The host owns indexing, source receipts, checkpointing, validation and transactional
    application. The model sees bounded exact source and research receipts, retrieves
    more evidence on demand, and performs edits with normal tools. There is no second
    file-plan, cursor, scalar-repair or tool-disabled recovery protocol here.
    """

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

        requested = tuple(
            str(value or "").strip() for value in (minecraft_version, loader, mappings)
        )
        if any(requested):
            if not all(requested):
                raise CustomModuleGenerationError(
                    "minecraft_version, loader and mappings must be supplied together."
                )
            try:
                adapter = adapter_for_target(requested[0], requested[1])
            except ValueError as exc:
                raise CustomModuleGenerationError(str(exc)) from exc
            if requested[2] != adapter.yarn_mappings:
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

        if self._cached_root == root and self._cached_index is not None:
            index = self._cached_index
        else:
            index = ProjectIndex(root, policy=self.policy)
            self._cached_root = root
            self._cached_index = index

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
                index = ProjectIndex(root, policy=self.policy)
                self._cached_index = index
                self._cached_root = root
                print(
                    "custom module: refreshed changing ProjectIndex snapshot",
                    f"attempt={snapshot_attempt + 1}/3",
                    flush=True,
                )
        if observation_ledger is None:
            raise CustomModuleGenerationError(
                "Project source kept changing while custom-module context was captured; "
                f"last error: {last_snapshot_error}"
            )

        observation_pages = _observation_context_pages(
            observation_ledger,
            query=query,
            byte_budget=project_context_budget,
        )
        research_context = select_module_research_context(
            research_modules,
            query=query,
            byte_budget=min(8 * 1024, project_context_budget),
        )
        host_grounding = build_coder_grounding(
            module_kind=module.kind,
            source_observation_receipt=observation_ledger["receipt"],
            research_context=research_context,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
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
            "phase": "implement_module",
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
                "Use workspace/RAG/MCP tools to retrieve exact source as needed instead of asking for the whole repository.",
                "Apply real edits with the source-edit tool; final text is only a short work summary.",
                "Edits are limited to src/main/java, src/main/resources, src/test/java and src/gametest.",
                "Build infrastructure, Gradle configuration and host-owned ledgers are read-only.",
                "Do not delete files. Preserve valid source already present in a resumed checkpoint.",
                "Use only the selected Minecraft/loader/mappings/Java target and preserve project conventions.",
            ],
        }
        initial_messages = [
            {
                "role": "system",
                "content": (
                    "You are the implementation coder for one approved Minecraft/Fabric module. "
                    "The host keeps the complete project indexed. Retrieve only source needed for "
                    "the current action, edit the staged workspace with tools, and do not invent "
                    "a second patch/file-plan protocol."
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]

        summary = ""
        continuation_count = 0
        seen_output_states: set[str] = set()
        active_messages = initial_messages
        while True:
            try:
                with _active_checkpoint_persistence(
                    checkpoint_root,
                    staged_root,
                    checkpoint_identity,
                ):
                    summary = self.router.generate_text(
                        "coder",
                        active_messages,
                        response_format="text",
                        tool_stage="generation",
                        enable_tools=True,
                    )
                _persist_generation_checkpoint(
                    checkpoint_root,
                    staged_root,
                    identity_sha256=checkpoint_identity,
                )
                break
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

                # Context-pressure recovery belongs exclusively to the canonical
                # progress-aware tool loop. If it bubbles out, preserve the original
                # boundary error instead of re-entering coder with tools disabled.
                boundary_kind = completion_boundary_kind(exc)
                if boundary_kind != OUTPUT_EXHAUSTED:
                    raise

                progress_operations, progress_paths, discarded_paths = (
                    _collect_staged_operations(root, staged_root, before)
                )
                if progress_operations:
                    self._validate_operations(progress_operations)
                    self._validate_total_patch_bytes(progress_operations)
                state_sha256 = _mutable_stage_state_sha256(staged_root)
                if continuation_count >= 5:
                    raise CustomModuleGenerationError(
                        f"Output continuation reached maximum allowed attempts ({continuation_count})."
                    ) from exc
                if state_sha256 in seen_output_states:
                    raise CustomModuleGenerationError(
                        "Output continuation reached a no-source-progress fixed point."
                    ) from exc
                seen_output_states.add(state_sha256)
                continuation_count += 1
                active_messages = _output_exhaustion_continuation_messages(
                    module=module,
                    minecraft_version=minecraft_version,
                    loader=loader,
                    mappings=mappings,
                    java_version=java_version,
                    continuation_index=continuation_count,
                    state_sha256=state_sha256,
                    touched_paths=progress_paths,
                    discarded_paths=discarded_paths,
                )
                print(
                    "custom module: bounded output continuation",
                    f"module={module.module_id}",
                    f"continuation={continuation_count}",
                    f"preserved_paths={len(progress_paths)}",
                    flush=True,
                )

        operations, touched_paths, discarded_paths = _collect_staged_operations(
            root,
            staged_root,
            before,
        )
        if not operations:
            discarded = ", ".join(discarded_paths[:8]) if discarded_paths else "none"
            raise CustomModuleGenerationError(
                "Custom-module coding agent produced no valid source/resource changes. "
                f"Discarded out-of-scope staged paths: {discarded}."
            )

        self._validate_operations(operations)
        self._validate_total_patch_bytes(operations)
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
        return {
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
            "agent_summary": str(summary or "").strip()[:4096],
            "output_exhaustion_continuations": continuation_count,
            "generation_checkpoint_resumed": checkpoint_resumed,
            "generation_checkpoint": {
                "schema_version": _CHECKPOINT_SCHEMA,
                "status": "AWAITING_LIVE_COMMIT",
                "identity_sha256": checkpoint_identity,
                "cleanup_token": checkpoint_token,
            },
            "required_gates": ["JDT", "Gradle", "GameTest", *module.required_gates],
        }

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


_GRADLE_METADATA_FILES = {
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradle.properties",
    "gradle/libs.versions.toml",
}


def _agent_mutable_path(path: str) -> bool:
    normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
    return (
        any(normalized.startswith(prefix) for prefix in _AGENT_MUTABLE_PREFIXES)
        or normalized.startswith(".minecraft_ai/")
        or normalized in _GRADLE_METADATA_FILES
    )


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


def _checkpoint_tree_state_sha256(root: Path) -> str:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _STAGE_IGNORED_DIRS for part in relative.parts):
            continue
        normalized = relative.as_posix()
        if path.is_symlink():
            rows.append((normalized, "symlink", str(path.readlink())))
        elif path.is_file():
            rows.append((normalized, "file", "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()))
        elif path.is_dir():
            rows.append((normalized, "directory", ""))
    return _sha256_json(rows)


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
        "stage_tree_sha256": _checkpoint_tree_state_sha256(staged_root),
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
        shutil.copytree(root, base_root, symlinks=True, ignore=_stage_ignore)
        shutil.copytree(base_root, staged_root, symlinks=True)
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
        shutil.copytree(root, next_base, symlinks=True, ignore=_stage_ignore)
        shutil.copytree(next_base, next_stage, symlinks=True)
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


def _project_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _STAGE_IGNORED_DIRS for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        snapshot[relative.as_posix()] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


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
) -> list[dict[str, str]]:
    touched = sorted({str(path) for path in touched_paths})
    discarded = sorted({str(path) for path in discarded_paths})
    request = {
        "phase": "implement_module",
        "task": "Continue the approved module from the preserved staged workspace; do not restart completed work.",
        "workspace_project_root": ".",
        "target": {
            "minecraft_version": minecraft_version,
            "loader": loader,
            "mappings": mappings,
            "java": java_version,
        },
        "module": _task_local_module_contract(module),
        "continuation": {
            "reason": "previous_tool_enabled_page_exhausted_output",
            "continuation_index": continuation_index,
            "preserved_source_state_sha256": state_sha256,
            "preserved_path_count": len(touched),
            "preserved_paths_preview": touched[:_CONTINUATION_PATH_PREVIEW],
            "discarded_out_of_scope_path_count": len(discarded),
        },
        "rules": [
            "Inspect the current staged workspace before editing; correct prior edits are already persisted.",
            "Retrieve source by path/symbol/RAG only as needed; never reconstruct the whole repository in context.",
            "Use bounded tool actions and continue across tool turns until the module is complete.",
            "Do not repeat an exhausted action and do not put source code in the final summary.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Continue one interrupted Minecraft mod implementation. The host preserved "
                "and hash-checked the staged workspace; resume with normal source/RAG tools."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        },
    ]


def _collect_staged_operations(
    original_root: Path,
    staged_root: Path,
    before: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
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


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise CustomModuleGenerationError("Model output did not contain one parseable JSON object.")


def _is_stale_project_index_error(exc: ValueError) -> bool:
    return str(exc).startswith("Project source changed after its context index was built:")


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


def _observation_context_pages(
    ledger: dict[str, Any],
    *,
    query: str,
    byte_budget: int,
) -> tuple[dict[str, Any], ...]:
    records = list(ledger["records"])
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
    anchors: list[dict[str, Any]] = []
    anchor_bytes = max(512, byte_budget // 2)
    for record in ranked:
        candidate = [*anchors, record]
        if _json_size(candidate) <= anchor_bytes:
            anchors.append(record)
    if ranked and not anchors:
        anchors.append(ranked[0])
    anchor_ids = {record["observation_id"] for record in anchors}
    remaining = [record for record in ranked if record["observation_id"] not in anchor_ids]
    pages: list[dict[str, Any]] = []
    cursor = 0
    safe_budget = max(1024, byte_budget - 128)
    while cursor < len(remaining) or not pages:
        page_records: list[dict[str, Any]] = []
        while cursor < len(remaining):
            candidate = _observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=0,
                anchors=anchors,
                records=[*page_records, remaining[cursor]],
                complete=False,
            )
            if _json_size(candidate) > safe_budget:
                if not page_records:
                    break
                break
            page_records.append(remaining[cursor])
            cursor += 1
        page = _observation_page_payload(
            receipt=ledger["receipt"],
            page_index=len(pages),
            page_count=0,
            anchors=anchors,
            records=page_records,
            complete=False,
        )
        pages.append(page)
        if cursor >= len(remaining):
            break
        if not page_records:
            # One oversized observation stays host-indexed; the coder can re-read it
            # with path/symbol tools rather than forcing it into the initial prompt.
            cursor += 1
    page_count = len(pages)
    for index, page in enumerate(pages):
        page["page_count"] = page_count
        page["complete"] = index == page_count - 1
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
