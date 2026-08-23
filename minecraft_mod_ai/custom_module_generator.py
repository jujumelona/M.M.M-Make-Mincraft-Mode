from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .complete_spec import ProductionModule
from .host_grounding import (
    build_coder_grounding,
    custom_module_path_allowed,
    custom_module_path_protected,
)
from .llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    completion_boundary_error,
    completion_boundary_kind,
)
from .model_router import ModelRouter
from .platform_catalog import adapter_for_target, adapter_from_project
from .project_index import ProjectIndex
from .research_ledger import select_module_research_context
from .scale_policy import ScalePolicy
from .source_patch import SourcePatchError, TransactionalSourcePatcher


class CustomModuleGenerationError(RuntimeError):
    pass


_OBSERVATION_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_ANCHOR_TERMS = {
    "api",
    "contract",
    "dependency",
    "implements",
    "interface",
    "public",
    "register",
    "required",
    "schema",
}
_FABRIC_ROOT_METADATA_PATH = "fabric.mod.json"
_FABRIC_CANONICAL_METADATA_PATH = "src/main/resources/fabric.mod.json"
_MAX_CUSTOM_MODULE_REPAIR_ATTEMPTS = 2
_AGENT_MUTABLE_PREFIXES = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
)
# Host ledgers/trajectory stores are neither implementation source nor model evidence.
# Excluding them also keeps resumable source checkpoints stable when telemetry changes.
_STAGE_IGNORED_DIRS = {".git", ".gradle", ".minecraft_ai", "build", "run"}
_CONTINUATION_PATH_PREVIEW = 64
_CHECKPOINT_DIRECTORY = ".mmm-custom-checkpoints"
_CHECKPOINT_SCHEMA = "mmm/custom-module-checkpoint-v2"
_CHECKPOINT_KEY = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_STRATEGY_EPOCH = "mmm/custom-candidate-strategy-v1"
_RECOVERY_CHUNK_CHARS = 2048
_RECOVERY_SOURCE_CONTEXT_BYTES = 4096
_RECOVERY_SOURCE_EDIT_OPERATIONS = (
    "replace_exact",
    "insert_before",
    "insert_after",
    "create_file",
    "append_file",
)
_PARTIAL_FUNCTION = re.compile(r"<function=\s*apply_source_edit\s*>")
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
    """Release every untransferred lease when generation exits or raises."""

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
    """Expose one exact staged workspace to the post-patch durability hook."""

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
    """Fsync the active manifest immediately after one staged source transaction."""

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
    """Size exact-source grounding from the active coder model's input window."""
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
    except Exception:
        return fallback

    max_context = int(getattr(config, "max_context", 0) or 0)
    max_input = int(getattr(config, "max_input_tokens", 0) or 0)
    max_new = int(getattr(config, "max_new_tokens", 0) or 0)
    input_tokens = max_input if max_input > 0 else max(0, max_context - max_new)
    if input_tokens <= 0:
        return fallback

    reserve_tokens = max(2048, min(8192, max_new // 2 if max_new > 0 else 4096))
    evidence_tokens = max(512, input_tokens - reserve_tokens)
    return min(hard_cap, max(1024, evidence_tokens * 2))


class CustomModuleGenerator:
    """Implement one approved module as a tool-using coding agent.

    The coder receives the feature goal, exact-source grounding and approved research, then
    inspects and edits a disposable project workspace with normal MCP tools. It is never asked
    to predict a file plan, patch protocol state, SHA values, cursors or completion metadata.
    The host validates the staged diff and transactionally applies only module source/resource
    changes to the real project.
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
            Path(checkpoint_root).expanduser()
            if checkpoint_root is not None
            else None
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
            str(value or "").strip()
            for value in (minecraft_version, loader, mappings)
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

        query = json.dumps(
            {
                "module_id": module.module_id,
                "kind": module.kind,
                "config": module.config,
                "depends_on": module.depends_on,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        project_context_budget = _coder_project_context_budget(
            self.router,
            self.policy,
            fast_mode=self.fast_mode,
        )
        if self.fast_mode:
            print(
                "🚀 [Fast-Path] host exact-source grounding limited to 4 KiB.",
                flush=True,
            )

        observation_ledger: dict[str, Any] | None = None
        last_snapshot_error: ValueError | None = None
        for snapshot_attempt in range(3):
            try:
                observation_ledger = _collect_initial_observations(
                    self.router,
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
                    "↻ [CustomModule] ProjectIndex snapshot changed during parallel generation; "
                    f"refreshed context ({snapshot_attempt + 1}/3).",
                    flush=True,
                )
        if observation_ledger is None:
            raise CustomModuleGenerationError(
                "Project source kept changing while custom-module context was being captured; "
                "refusing to generate from a stale snapshot. "
                f"Last error: {last_snapshot_error}"
            )

        observation_pages = _observation_context_pages(
            observation_ledger,
            query=query,
            byte_budget=project_context_budget,
        )
        research_context = select_module_research_context(
            research_modules,
            query=query,
            byte_budget=8 * 1024,
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
            root=root,
            module_query=query,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
            observation_receipt=observation_ledger["receipt"],
            research_context=research_context,
            host_grounding=host_grounding,
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
        self.router.bind_agent_workspace(
            staged_root,
            require_fresh_evidence=True,
        )
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
                "module": {
                    "module_id": module.module_id,
                    "kind": module.kind,
                    "config": module.config,
                    "depends_on": list(module.depends_on),
                    "required_gates": list(module.required_gates),
                },
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
                    "Implement the feature; do not return or invent a file-plan protocol.",
                    "Use available workspace/RAG/MCP tools to inspect any source or research you need before editing.",
                    "Apply real edits with the available source patch tool; your final text is only a short work summary.",
                    "Custom-module edits are limited to src/main/java, src/main/resources, src/test/java and src/gametest.",
                    "Build infrastructure, Gradle wrapper/settings, shell scripts and host-owned ledgers are read-only for this task.",
                    "Do not delete files. If a tool action is rejected, inspect the project and recover with a valid source/resource edit.",
                    "Use only the selected Minecraft/loader/mappings/Java target and preserve existing project conventions.",
                    "Keep gameplay state server-authoritative and persistent where the approved feature requires it.",
                    "Preserve valid source/resource work already present in a resumed checkpoint; inspect before replacing it.",
                ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the implementation coder for one approved Minecraft/Fabric mod module. "
                    "Work directly in the supplied project with MCP tools: inspect the existing code, "
                    "retrieve relevant evidence when needed, and implement the requested feature. "
                    "Do not design a separate file-plan/JSON patch protocol. The host owns safety, "
                    "scope, transactional application and verification."
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        summary = ""
        continuation_count = 0
        scalar_obligation_count = 0
        exhausted_states: set[str] = set()
        active_messages = messages
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
                boundary = completion_boundary_error(exc)
                boundary_kind = completion_boundary_kind(exc)
                partial_context_pressure = bool(
                    boundary_kind == CONTEXT_PRESSURE
                    and boundary is not None
                    and int(getattr(boundary, "partial_bytes", 0) or 0) > 0
                )
                if boundary_kind != OUTPUT_EXHAUSTED and not partial_context_pressure:
                    raise

                # A tool-aware completion may have committed several bounded edits
                # before its final assistant action hit the decode ceiling. Keep the
                # same isolated workspace alive, validate all progress observed so
                # far, and continue from that exact state. Re-copying the project or
                # replaying the durable work node would discard work and can repeat
                # the identical oversized action.
                progress_operations, progress_paths, discarded_paths = (
                    _collect_staged_operations(root, staged_root, before)
                )
                if progress_operations:
                    self._validate_operations(progress_operations)
                    self._validate_total_patch_bytes(progress_operations)

                state_sha256 = _mutable_stage_state_sha256(staged_root)
                if partial_context_pressure or state_sha256 in exhausted_states:
                    # The transport already attempted safe assistant-prefill. If the
                    # exact source state still repeats, convert the incomplete large
                    # action into one schema-bounded scalar obligation. This remains a
                    # model-authored edit, is materialized by the canonical scalar
                    # protocol, and is applied only inside the isolated checkpoint.
                    with _active_checkpoint_persistence(
                        checkpoint_root,
                        staged_root,
                        checkpoint_identity,
                    ):
                        obligation_receipt = _apply_bounded_scalar_obligation(
                            self.router,
                            staged_root=staged_root,
                            module=module,
                            minecraft_version=minecraft_version,
                            loader=loader,
                            mappings=mappings,
                            java_version=java_version,
                            state_sha256=state_sha256,
                            boundary=boundary,
                        )
                    scalar_obligation_count += 1
                    _persist_generation_checkpoint(
                        checkpoint_root,
                        staged_root,
                        identity_sha256=checkpoint_identity,
                    )
                    next_state_sha256 = _mutable_stage_state_sha256(staged_root)
                    if next_state_sha256 == state_sha256:
                        raise CustomModuleGenerationError(
                            "Bounded scalar obligation made no staged source progress."
                        )
                    state_sha256 = next_state_sha256
                    progress_operations, progress_paths, discarded_paths = (
                        _collect_staged_operations(root, staged_root, before)
                    )
                    self._validate_operations(progress_operations)
                    self._validate_total_patch_bytes(progress_operations)
                else:
                    exhausted_states.add(state_sha256)
                    obligation_receipt = None
                continuation_count += 1
                active_messages = _output_exhaustion_continuation_messages(
                    staged_root=staged_root,
                    module=module,
                    minecraft_version=minecraft_version,
                    loader=loader,
                    mappings=mappings,
                    java_version=java_version,
                    continuation_index=continuation_count,
                    state_sha256=state_sha256,
                    touched_paths=progress_paths,
                    discarded_paths=discarded_paths,
                    obligation_receipt=obligation_receipt,
                    boundary_kind=boundary_kind,
                )
                print(
                    "custom module: bounded continuation",
                    f"module={module.module_id}",
                    f"continuation={continuation_count}",
                    f"boundary={boundary_kind}",
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
            "runtime_tests": ["Verify the approved mod functionality, compilation, and runtime behavior without crash."],
            "source_observation_receipt": observation_ledger["receipt"],
            "touched_paths": touched_paths,
            "discarded_out_of_scope_paths": discarded_paths,
            "agent_summary": str(summary or "").strip()[:4096],
            "output_exhaustion_continuations": continuation_count,
            "completion_boundary_continuations": continuation_count,
            "bounded_scalar_obligations": scalar_obligation_count,
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
        """Clean one owned checkpoint only after an outer live-project commit."""

        if not isinstance(result, dict):
            return False
        checkpoint = result.get("generation_checkpoint")
        if not isinstance(checkpoint, dict):
            return False
        if (
            checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA
        ):
            return False
        if checkpoint.get("status") == "CLEANED_AFTER_LIVE_COMMIT":
            return "cleanup_token" not in checkpoint
        if checkpoint.get("status") != "AWAITING_LIVE_COMMIT":
            return False
        token = checkpoint.get("cleanup_token")
        identity = checkpoint.get("identity_sha256")
        if (
            not isinstance(token, str)
            or not _CHECKPOINT_KEY.fullmatch(token)
            or not isinstance(identity, str)
        ):
            return False
        with self._checkpoint_cleanup_lock:
            owned = self._checkpoint_cleanup_tokens.get(token)
            if owned is None or owned[0] != identity:
                # Never return an opaque cleanup capability in a durable/public
                # receipt when this generator cannot prove ownership of it.
                checkpoint["status"] = "UNACKNOWLEDGED_AFTER_LIVE_COMMIT"
                checkpoint.pop("cleanup_token", None)
                return False
            checkpoint_root = owned[1]
            checkpoint_lease = owned[2]
            try:
                _remove_generation_checkpoint(checkpoint_root)
            except (CustomModuleGenerationError, OSError):
                # The live patch has already committed. A cleanup failure must not
                # retain an in-process capability lease forever and block every later
                # generation with the same identity. Preserve the on-disk checkpoint,
                # revoke the opaque token, and release only the lease.
                self._checkpoint_cleanup_tokens.pop(token, None)
                checkpoint_lease.close()
                checkpoint["status"] = "PRESERVED_AFTER_CLEANUP_FAILURE"
                checkpoint.pop("cleanup_token", None)
                return False
            self._checkpoint_cleanup_tokens.pop(token, None)
            checkpoint_lease.close()
        checkpoint["status"] = "CLEANED_AFTER_LIVE_COMMIT"
        checkpoint.pop("cleanup_token", None)
        return True

    def release_generation_checkpoint(self, result: Any) -> bool:
        """Release a failed/losing candidate while preserving its staged work."""

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
        if (
            not isinstance(token, str)
            or not _CHECKPOINT_KEY.fullmatch(token)
            or not isinstance(identity, str)
        ):
            return False
        with self._checkpoint_cleanup_lock:
            owned = self._checkpoint_cleanup_tokens.get(token)
            if owned is None or owned[0] != identity:
                return False
            self._checkpoint_cleanup_tokens.pop(token, None)
            owned[2].close()
        checkpoint["status"] = "PRESERVED_FOR_RESUME"
        checkpoint.pop("cleanup_token", None)
        return True

    def discard_generation_checkpoint(self, result: Any) -> bool:
        """Delete a losing checkpoint after another candidate commits live."""

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
        if (
            not isinstance(token, str)
            or not _CHECKPOINT_KEY.fullmatch(token)
            or not isinstance(identity, str)
        ):
            checkpoint.pop("cleanup_token", None)
            return False
        with self._checkpoint_cleanup_lock:
            owned = self._checkpoint_cleanup_tokens.get(token)
            if owned is None or owned[0] != identity:
                checkpoint["status"] = "UNOWNED_LOSER_CHECKPOINT"
                checkpoint.pop("cleanup_token", None)
                return False
            self._checkpoint_cleanup_tokens.pop(token, None)
            try:
                _remove_generation_checkpoint(owned[1])
            except (CustomModuleGenerationError, OSError):
                checkpoint["status"] = "PRESERVED_AFTER_CLEANUP_FAILURE"
                removed = False
            else:
                checkpoint["status"] = "DISCARDED_AFTER_OTHER_WINNER"
                removed = True
            finally:
                owned[2].close()
                checkpoint.pop("cleanup_token", None)
        return removed

    def _validate_file_plan(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Legacy trace validator; production generation no longer asks for a file plan."""
        if not isinstance(payload, dict):
            raise CustomModuleGenerationError("Custom-module file plan must be an object.")
        raw_files = payload.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise CustomModuleGenerationError(
                "Custom-module file plan must contain at least one file."
            )
        planned: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in raw_files:
            if not isinstance(item, dict):
                raise CustomModuleGenerationError("Custom-module planned file must be an object.")
            path = _canonicalize_planned_path(item.get("path"))
            purpose = str(item.get("purpose", "")).strip()
            if not purpose:
                raise CustomModuleGenerationError(
                    f"Custom-module planned file needs a non-empty purpose: {path}"
                )
            if custom_module_path_protected(path):
                raise CustomModuleGenerationError(
                    "Model file planning may not modify the code-owned research/context ledgers."
                )
            if not custom_module_path_allowed(path):
                raise CustomModuleGenerationError(
                    f"Custom module path is outside the allowed scope: {path}"
                )
            if path not in seen:
                seen.add(path)
                planned.append({"path": path, "purpose": purpose})
        tests_value = payload.get("runtime_tests", [])
        if not isinstance(tests_value, list) or any(not isinstance(v, str) for v in tests_value):
            raise CustomModuleGenerationError("File plan runtime_tests must be a list of strings.")
        return planned, [value.strip() for value in tests_value if value.strip()]

    def _validate_operations(self, operations: list[dict[str, Any]]) -> None:
        for item in operations:
            if not isinstance(item, dict):
                raise CustomModuleGenerationError("Patch operation must be an object.")
            if item.get("operation") not in {"create", "replace", "edit"}:
                raise CustomModuleGenerationError("Custom module may not delete files.")
            path = _normalized_operation_path(item)
            if custom_module_path_protected(path):
                raise CustomModuleGenerationError(
                    "Model patches may not modify the code-owned research ledger or context-observation ledger."
                )
            if not _agent_mutable_path(path):
                raise CustomModuleGenerationError(
                    f"Custom module path is outside the source/resource scope: {path}"
                )

    def _validate_total_patch_bytes(self, operations: list[dict[str, Any]]) -> None:
        size = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if size > self.policy.max_patch_bytes:
            raise CustomModuleGenerationError(
                "Custom module patch exceeds MMM_MAX_PATCH_BYTES; raise the explicit host resource policy or split the feature into dependency-linked modules."
            )


def _agent_mutable_path(path: str) -> bool:
    normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
    return any(normalized.startswith(prefix) for prefix in _AGENT_MUTABLE_PREFIXES)


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
    """Find only stable candidate-search identity; never serialize a router object."""

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
                    "Candidate checkpoint identity requires a valid index, count, and strategy."
                )
            return {
                "mode": "candidate",
                "strategy_epoch": _CHECKPOINT_STRATEGY_EPOCH,
                "candidate_index": candidate_index,
                "candidate_count": candidate_count,
                "strategy": strategy.strip(),
            }
        current = getattr(current, "_router", None)
    return {
        "mode": "single",
        "strategy_epoch": _CHECKPOINT_STRATEGY_EPOCH,
    }


def _checkpoint_tree_state_sha256(root: Path) -> str:
    """Bind a checkpoint to files and symlink metadata without following symlinks."""

    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _STAGE_IGNORED_DIRS for part in relative.parts):
            continue
        normalized = relative.as_posix()
        if path.is_symlink():
            rows.append((normalized, "symlink", str(path.readlink())))
        elif path.is_file():
            rows.append(
                (
                    normalized,
                    "file",
                    "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        elif path.is_dir():
            rows.append((normalized, "directory", ""))
    return _sha256_json(rows)


def _generation_checkpoint_identity(
    *,
    root: Path,
    module_query: str,
    minecraft_version: str,
    loader: str,
    mappings: str,
    observation_receipt: Any,
    research_context: Any,
    host_grounding: Any,
    router: Any,
) -> str:
    """Bind resumable work to the obligation, target, research, and candidate.

    The live source tree and source-observation receipt deliberately do not form the
    key. They are versioned in the checkpoint manifest and three-way rebased on open,
    so an unrelated module committing in parallel cannot orphan completed work.
    """

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
        raise CustomModuleGenerationError(
            "Custom-module checkpoint parent may not be a symlink."
        )
    if base.exists() and (base.is_symlink() or not base.is_dir()):
        raise CustomModuleGenerationError(
            "Custom-module checkpoint root must be a regular host directory."
        )
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _safe_checkpoint_path(base: Path, key: str) -> Path:
    if not _CHECKPOINT_KEY.fullmatch(key):
        raise CustomModuleGenerationError("Unsafe custom-module checkpoint key")
    checkpoint_root = base / key
    try:
        checkpoint_root.resolve().relative_to(base)
    except ValueError as exc:
        raise CustomModuleGenerationError(
            "Custom-module checkpoint escaped its host-owned root."
        ) from exc
    return checkpoint_root


def _remove_generation_checkpoint(checkpoint_root: Path) -> None:
    """Remove only a validated host-owned checkpoint after the patch commits."""

    declared_base = checkpoint_root.parent
    if declared_base.is_symlink() or not declared_base.is_dir():
        raise CustomModuleGenerationError(
            "Refusing to remove a checkpoint through an unsafe host root."
        )
    base = declared_base.resolve()
    if base != declared_base:
        raise CustomModuleGenerationError(
            "Refusing to remove a checkpoint through a redirected host root."
        )
    if base.name != _CHECKPOINT_DIRECTORY or not _CHECKPOINT_KEY.fullmatch(
        checkpoint_root.name
    ):
        raise CustomModuleGenerationError(
            "Refusing to remove an unrecognized custom-module checkpoint path."
        )
    if checkpoint_root.is_symlink():
        raise CustomModuleGenerationError(
            "Refusing to remove a symlinked custom-module checkpoint."
        )
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
    try:
        checkpoint_stat = checkpoint_root.lstat()
        staged_stat = staged_root.lstat()
        base_root = _checkpoint_base(checkpoint_root)
        base_stat = base_root.lstat()
    except OSError as exc:
        raise ValueError("Custom-module checkpoint staging root is missing") from exc
    if (
        not stat.S_ISDIR(checkpoint_stat.st_mode)
        or not stat.S_ISDIR(staged_stat.st_mode)
        or not stat.S_ISDIR(base_stat.st_mode)
    ):
        raise ValueError("Custom-module checkpoint staging root is unsafe")
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "identity_sha256": identity_sha256,
        "base_tree_sha256": _checkpoint_tree_state_sha256(base_root),
        "stage_tree_sha256": _checkpoint_tree_state_sha256(staged_root),
    }
    manifest = _checkpoint_manifest(checkpoint_root)
    legacy_temporary = checkpoint_root / ".checkpoint.json.tmp"
    if legacy_temporary.is_symlink():
        raise ValueError("Custom-module checkpoint temporary may not be a symlink")
    if manifest.is_symlink():
        raise ValueError("Custom-module checkpoint manifest may not be a symlink")
    if manifest.exists() and not stat.S_ISREG(manifest.lstat().st_mode):
        raise ValueError("Custom-module checkpoint manifest must be a regular file")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor = -1
    temporary_name = ""
    temporary_stat: os.stat_result | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".checkpoint-",
            suffix=".tmp",
            dir=checkpoint_root,
        )
        temporary_stat = os.fstat(descriptor)
        if not stat.S_ISREG(temporary_stat.st_mode):
            raise ValueError("Custom-module checkpoint temporary is not regular")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        current_checkpoint_stat = checkpoint_root.lstat()
        current_temporary_stat = os.lstat(temporary_name)
        if (
            not stat.S_ISDIR(current_checkpoint_stat.st_mode)
            or current_checkpoint_stat.st_dev != checkpoint_stat.st_dev
            or current_checkpoint_stat.st_ino != checkpoint_stat.st_ino
            or temporary_stat.st_dev != current_temporary_stat.st_dev
            or temporary_stat.st_ino != current_temporary_stat.st_ino
            or not stat.S_ISREG(current_temporary_stat.st_mode)
        ):
            raise ValueError("Custom-module checkpoint directory changed during persistence")
        if manifest.is_symlink():
            raise ValueError("Custom-module checkpoint manifest became a symlink")
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
    descriptor = os.open(
        manifest,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("Custom-module checkpoint manifest must be regular")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            raw = json.load(handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(raw, dict):
        raise ValueError("Custom-module checkpoint manifest must be an object")
    return raw


def _initialize_generation_checkpoint(
    root: Path,
    checkpoint_root: Path,
    *,
    identity_sha256: str,
) -> Path:
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=False, exist_ok=False)
    base_root = _checkpoint_base(checkpoint_root)
    staged_root = checkpoint_root / "project"
    try:
        shutil.copytree(
            root,
            base_root,
            symlinks=True,
            ignore=_stage_ignore,
        )
        shutil.copytree(
            base_root,
            staged_root,
            symlinks=True,
        )
        _persist_generation_checkpoint(
            checkpoint_root,
            staged_root,
            identity_sha256=identity_sha256,
        )
    except BaseException:
        if checkpoint_root.exists() and not checkpoint_root.is_symlink():
            shutil.rmtree(checkpoint_root, ignore_errors=True)
        raise
    return staged_root


def _checkpoint_patch_operations(
    base_root: Path,
    staged_root: Path,
) -> list[dict[str, Any]]:
    operations, _touched, discarded = _collect_staged_operations(
        base_root,
        staged_root,
        _project_snapshot(base_root),
    )
    if discarded:
        raise CustomModuleGenerationError(
            "Resumable checkpoint contains out-of-scope source changes."
        )
    for operation in operations:
        path = _normalized_operation_path(operation)
        if custom_module_path_protected(path) or not _agent_mutable_path(path):
            raise CustomModuleGenerationError(
                "Resumable checkpoint contains a protected source change."
            )
    return operations


def _fresh_checkpoint_child(checkpoint_root: Path, prefix: str) -> Path:
    child = Path(tempfile.mkdtemp(prefix=prefix, dir=checkpoint_root))
    child.rmdir()
    return child


def _rebase_generation_checkpoint(
    root: Path,
    checkpoint_root: Path,
    *,
    identity_sha256: str,
) -> Path:
    """Replay only checkpoint-authored edits over a newer unrelated base tree."""

    base_root = _checkpoint_base(checkpoint_root)
    staged_root = checkpoint_root / "project"
    operations = _checkpoint_patch_operations(base_root, staged_root)
    next_base: Path | None = _fresh_checkpoint_child(checkpoint_root, ".base-rebase-")
    next_stage: Path | None = _fresh_checkpoint_child(checkpoint_root, ".project-rebase-")
    try:
        assert next_base is not None and next_stage is not None
        shutil.copytree(root, next_base, symlinks=True, ignore=_stage_ignore)
        shutil.copytree(next_base, next_stage, symlinks=True)
        if operations:
            TransactionalSourcePatcher(next_stage).apply(operations)
        shutil.rmtree(base_root)
        shutil.rmtree(staged_root)
        os.replace(next_base, base_root)
        next_base = None
        os.replace(next_stage, staged_root)
        next_stage = None
        _persist_generation_checkpoint(
            checkpoint_root,
            staged_root,
            identity_sha256=identity_sha256,
        )
        return staged_root
    finally:
        for temporary in (next_base, next_stage):
            if temporary is not None and temporary.exists() and not temporary.is_symlink():
                shutil.rmtree(temporary, ignore_errors=True)


def _prepare_generation_checkpoint(
    root: Path,
    *,
    identity_sha256: str,
    configured_root: Path | None = None,
) -> tuple[Path, Path, bool, _GenerationCheckpointLease]:
    """Open an exact resumable workspace or replace a stale owned checkpoint."""

    base = _checkpoint_directory(root, configured_root)
    checkpoint_root = _safe_checkpoint_path(base, _checkpoint_key(identity_sha256))
    lease = _GenerationCheckpointLease(checkpoint_root)
    try:
        staged_root = checkpoint_root / "project"
        if checkpoint_root.exists():
            if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
                raise CustomModuleGenerationError(
                    "Custom-module checkpoint path is not a regular directory."
                )
            try:
                raw = _read_generation_checkpoint_manifest(checkpoint_root)
                base_root = _checkpoint_base(checkpoint_root)
                reusable = (
                    raw.get("schema_version") == _CHECKPOINT_SCHEMA
                    and raw.get("identity_sha256") == identity_sha256
                    and isinstance(raw.get("base_tree_sha256"), str)
                    and isinstance(raw.get("stage_tree_sha256"), str)
                    and base_root.is_dir()
                    and not base_root.is_symlink()
                    and staged_root.is_dir()
                    and not staged_root.is_symlink()
                    and raw["base_tree_sha256"]
                    == _checkpoint_tree_state_sha256(base_root)
                    and raw["stage_tree_sha256"]
                    == _checkpoint_tree_state_sha256(staged_root)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                reusable = False
            if reusable:
                if raw["base_tree_sha256"] != _checkpoint_tree_state_sha256(root):
                    try:
                        staged_root = _rebase_generation_checkpoint(
                            root,
                            checkpoint_root,
                            identity_sha256=identity_sha256,
                        )
                    except (
                        CustomModuleGenerationError,
                        OSError,
                        SourcePatchError,
                        ValueError,
                    ):
                        reusable = False
                if reusable:
                    return checkpoint_root, staged_root, True, lease
            _remove_generation_checkpoint(checkpoint_root)
        staged_root = _initialize_generation_checkpoint(
            root,
            checkpoint_root,
            identity_sha256=identity_sha256,
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
    """Hash only source/resource state that the coding agent is allowed to mutate."""

    snapshot = {
        path: digest
        for path, digest in _project_snapshot(staged_root).items()
        if _agent_mutable_path(path) and not custom_module_path_protected(path)
    }
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _completed_partial_parameter(text: str, key: str, *, maximum: int) -> str:
    marker = f"<parameter={key}>"
    start = text.find(marker)
    if start < 0:
        return ""
    value_start = start + len(marker)
    end = text.find("</parameter>", value_start)
    if end < 0:
        return ""
    value = text[value_start:end].strip()
    if not value or len(value) > maximum or "<parameter=" in value:
        return ""
    tail = text[end + len("</parameter>") :].lstrip()
    if tail and not tail.startswith(("<parameter=", "</function>", "</tool_call>")):
        return ""
    return value


def _partial_source_edit_hint(boundary: Any) -> dict[str, str]:
    """Extract only completed operation/path headers; never expose partial source."""

    partial = getattr(boundary, "partial_message", {})
    if not isinstance(partial, dict):
        return {}
    hints: list[dict[str, str]] = []
    for field in ("reasoning_content", "reasoning", "content"):
        text = partial.get(field)
        if not isinstance(text, str):
            continue
        match = _PARTIAL_FUNCTION.search(text)
        if match is None:
            continue
        fragment = text[match.end() :]
        payload_starts = [
            value
            for value in (
                fragment.find("<parameter=old>"),
                fragment.find("<parameter=new>"),
                fragment.find("<parameter=anchor>"),
                fragment.find("<parameter=content>"),
                fragment.find("<parameter=text>"),
            )
            if value >= 0
        ]
        header = fragment[: min(payload_starts)] if payload_starts else fragment
        operation = _completed_partial_parameter(header, "operation", maximum=32)
        path = _completed_partial_parameter(header, "path", maximum=512)
        hint: dict[str, str] = {}
        aliases = {
            "replace": "replace_exact",
            "create": "create_file",
        }
        operation = aliases.get(operation, operation)
        if operation in _RECOVERY_SOURCE_EDIT_OPERATIONS:
            hint["operation"] = operation
        if path:
            try:
                normalized = _canonicalize_planned_path(path)
            except CustomModuleGenerationError:
                normalized = ""
            if (
                normalized
                and _agent_mutable_path(normalized)
                and not custom_module_path_protected(normalized)
            ):
                hint["path"] = normalized
        if hint:
            hints.append(hint)
    if not hints:
        return {}
    first = hints[0]
    for hint in hints[1:]:
        for key in tuple(first):
            if key in hint and hint[key] != first[key]:
                first.pop(key, None)
    return first


def _bounded_scalar_obligation_schema(hint: dict[str, str]) -> dict[str, Any]:
    required_by_operation = {
        "replace_exact": ["old", "new"],
        "insert_before": ["anchor", "content"],
        "insert_after": ["anchor", "content"],
        "create_file": ["content"],
        "append_file": ["content"],
    }
    operation: dict[str, Any] = {
        "type": "string",
        "enum": list(_RECOVERY_SOURCE_EDIT_OPERATIONS),
    }
    path: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 512,
    }
    if hint.get("operation") in _RECOVERY_SOURCE_EDIT_OPERATIONS:
        operation["const"] = hint["operation"]
    if hint.get("path"):
        path["const"] = hint["path"]
    required = ["operation", "path"]
    hinted_operation = hint.get("operation")
    if hinted_operation in required_by_operation:
        required.extend(required_by_operation[hinted_operation])
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "operation": operation,
            "path": path,
            "old": {
                "type": "string",
                "minLength": 1,
                "maxLength": _RECOVERY_CHUNK_CHARS,
            },
            "new": {"type": "string", "maxLength": _RECOVERY_CHUNK_CHARS},
            "anchor": {
                "type": "string",
                "minLength": 1,
                "maxLength": _RECOVERY_CHUNK_CHARS,
            },
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": _RECOVERY_CHUNK_CHARS,
            },
            "count": {"type": "integer", "const": 1, "default": 1},
        },
    }
    if hinted_operation not in required_by_operation:
        schema["allOf"] = [
            {
                "if": {
                    "properties": {"operation": {"const": operation_name}},
                    "required": ["operation"],
                },
                "then": {"required": fields},
            }
            for operation_name, fields in required_by_operation.items()
        ]
    return schema


def _bounded_recovery_source_context(
    staged_root: Path,
    hint: dict[str, str],
) -> dict[str, Any]:
    """Expose only a small exact excerpt for a validated hinted project path."""

    relative = hint.get("path", "")
    if not relative:
        return {}
    parts = PurePosixPath(relative).parts
    cursor = staged_root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return {"path": relative, "status": "SYMLINK_REJECTED"}
    target = staged_root.joinpath(*parts)
    if target.is_symlink() or not target.is_file():
        return {"path": relative, "status": "NOT_AN_EXISTING_REGULAR_FILE"}
    try:
        descriptor = os.open(
            target,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_BINARY", 0),
        )
    except OSError:
        return {"path": relative, "status": "UNREADABLE"}
    hasher = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            return {"path": relative, "status": "NON_REGULAR_REJECTED"}
        byte_count = opened.st_size
        while chunk := handle.read(64 * 1024):
            hasher.update(chunk)
        if byte_count <= _RECOVERY_SOURCE_CONTEXT_BYTES:
            handle.seek(0)
            excerpt_bytes = handle.read(_RECOVERY_SOURCE_CONTEXT_BYTES)
            excerpt_kind = "complete"
        else:
            half = _RECOVERY_SOURCE_CONTEXT_BYTES // 2
            handle.seek(0)
            head = handle.read(half)
            handle.seek(max(0, byte_count - half))
            tail = handle.read(half)
            excerpt_bytes = head + b"\n...<host excerpt gap>...\n" + tail
            excerpt_kind = "head_tail"
    digest = "sha256:" + hasher.hexdigest()
    try:
        excerpt = excerpt_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "path": relative,
            "status": "NON_UTF8",
            "sha256": digest,
            "byte_count": byte_count,
        }
    return {
        "path": relative,
        "status": "EXACT_UTF8_EXCERPT",
        "sha256": digest,
        "byte_count": byte_count,
        "excerpt_kind": excerpt_kind,
        "source": excerpt,
    }


def _apply_bounded_scalar_obligation(
    router: Any,
    *,
    staged_root: Path,
    module: ProductionModule,
    minecraft_version: str,
    loader: str,
    mappings: str,
    java_version: int,
    state_sha256: str,
    boundary: Any,
) -> dict[str, Any]:
    """Request and apply one small model-authored edit in the isolated checkpoint."""

    hint = _partial_source_edit_hint(boundary)
    active_hint = dict(hint)
    partial_bytes = int(getattr(boundary, "partial_bytes", 0) or 0)
    partial_sha = str(getattr(boundary, "partial_sha256", "") or "")
    schema = _bounded_scalar_obligation_schema(active_hint)
    request = {
        "phase": "implement_module_scalar_obligation",
        "task": (
            "Return exactly one small source edit that advances the approved module. "
            "This is an internal bounded recovery action, not a file plan."
        ),
        "workspace_project_root": ".",
        "target": {
            "minecraft_version": minecraft_version,
            "loader": loader,
            "mappings": mappings,
            "java": java_version,
        },
        "module": {
            "module_id": module.module_id,
            "kind": module.kind,
            "config": module.config,
        },
        "preserved_source_state_sha256": state_sha256,
        "hinted_source_context": _bounded_recovery_source_context(staged_root, hint),
        "truncated_action_receipt": {
            "partial_bytes": partial_bytes,
            "partial_sha256": "sha256:" + partial_sha if partial_sha else "",
            "safe_header_hint": hint,
        },
        "rules": [
            "Return one JSON object matching the supplied schema and nothing else.",
            "Touch exactly one source/resource path and do not delete or replace a whole large file.",
            f"Every old, new, anchor, or content string must be at most {_RECOVERY_CHUNK_CHARS} characters.",
            "Use create_file for a small new-file prefix, append_file for the next chunk, or a bounded exact/anchor edit for an existing file.",
            "Do not repeat or quote the truncated source payload.",
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Produce one bounded Minecraft source-edit obligation. The host will "
                "validate its schema, path, hash preconditions, and isolated scope."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    # The approved research bundle was already consumed by the interrupted coder
    # turn. Avoid the research-evolution wrapper here: it can turn one bounded JSON
    # obligation into another full coder generation and recreate the same bottleneck.
    from .custom_generation_research import _strip_research_router

    obligation_router = _strip_research_router(router)
    from .structured_output import (
        StructuredOutputValidationError,
        validate_structured_output,
    )
    from . import agent_tool_runtime
    from .source_edit_scalar_protocol_contract import materialize_model_source_edit

    seen_rejections: set[str] = set()
    corrective_messages = messages
    while True:
        raw = obligation_router.generate_text(
            "coder",
            corrective_messages,
            response_format="json",
            response_schema=schema,
            tool_stage="generation",
            enable_tools=False,
        )
        model_edit: dict[str, Any] = {}
        try:
            validated = validate_structured_output(
                str(raw or ""),
                response_format="json",
                response_schema=schema,
            )
            decoded = json.loads(validated)
            if not isinstance(decoded, dict):
                raise CustomModuleGenerationError(
                    "Bounded scalar obligation must decode to one object."
                )
            model_edit = decoded
            if active_hint.get("operation") and model_edit.get("operation") != active_hint["operation"]:
                raise CustomModuleGenerationError(
                    "Bounded scalar obligation changed the host-bound operation hint."
                )
            if active_hint.get("path") and model_edit.get("path") != active_hint["path"]:
                raise CustomModuleGenerationError(
                    "Bounded scalar obligation changed the host-bound path hint."
                )

            # This is not a synthesized MCP call: the model authors the complete
            # scalar object, while the canonical protocol supplies only project
            # discovery, scope, and current SHA-256 preconditions.
            patch = materialize_model_source_edit(
                agent_tool_runtime,
                staged_root,
                model_edit,
                bound_project_root=staged_root,
            )
            operations = patch.get("operations")
            if not isinstance(operations, list) or len(operations) != 1:
                raise CustomModuleGenerationError(
                    "Bounded scalar recovery must materialize exactly one source operation."
                )
            receipt = TransactionalSourcePatcher(staged_root).apply(operations)
        except (
            agent_tool_runtime.AgentToolRuntimeError,
            CustomModuleGenerationError,
            json.JSONDecodeError,
            SourcePatchError,
            StructuredOutputValidationError,
            ValueError,
        ) as exc:
            redacted = agent_tool_runtime._redact_text(str(exc))
            compact_error = " ".join(redacted.split())[:512]
            rejection = {
                "schema_version": "mmm/bounded-scalar-rejection-v1",
                "operation": str(model_edit.get("operation", ""))[:32],
                "path": str(model_edit.get("path", ""))[:512],
                "candidate_sha256": _sha256_json(str(raw or "")),
                "error_type": type(exc).__name__,
                "error_sha256": _sha256_json(compact_error),
            }
            rejection_progress = _sha256_json(
                {
                    "stage_state_sha256": _mutable_stage_state_sha256(staged_root),
                    "error_type": type(exc).__name__,
                    "operation": rejection["operation"],
                    "path": rejection["path"],
                }
            )
            if rejection_progress in seen_rejections:
                raise CustomModuleGenerationError(
                    "Bounded scalar correction reached a no-source-progress fixed point."
                ) from exc
            seen_rejections.add(rejection_progress)
            if isinstance(
                exc,
                (agent_tool_runtime.AgentToolRuntimeError, SourcePatchError),
            ) and active_hint.get("operation"):
                # A completed partial header is only a hint. Once the host proves its
                # operation precondition stale, keep the validated path but permit the
                # model to choose another bounded operation on that same path.
                active_hint.pop("operation", None)
                schema = _bounded_scalar_obligation_schema(active_hint)
            correction_request = dict(request)
            correction_request["hinted_source_context"] = (
                _bounded_recovery_source_context(staged_root, active_hint)
            )
            correction_request["truncated_action_receipt"] = {
                **request["truncated_action_receipt"],
                "safe_header_hint": active_hint,
            }
            correction_request["previous_rejection"] = {
                **rejection,
                "instruction": (
                    "Return a different schema-valid scalar edit that matches the "
                    "current exact source context and host preconditions."
                ),
            }
            corrective_messages = [
                messages[0],
                {
                    "role": "user",
                    "content": json.dumps(
                        correction_request,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ]
            continue
        return {
            "schema_version": "mmm/bounded-scalar-obligation-receipt-v1",
            "operation": str(model_edit.get("operation", "")),
            "path": str(model_edit.get("path", "")),
            "partial_bytes": partial_bytes,
            "partial_sha256": "sha256:" + partial_sha if partial_sha else "",
            "correction_count": len(seen_rejections),
            "patch_receipt": receipt,
        }


def _output_exhaustion_continuation_messages(
    *,
    staged_root: Path,
    module: ProductionModule,
    minecraft_version: str,
    loader: str,
    mappings: str,
    java_version: int,
    continuation_index: int,
    state_sha256: str,
    touched_paths: Iterable[str],
    discarded_paths: Iterable[str],
    obligation_receipt: dict[str, Any] | None = None,
    boundary_kind: str = OUTPUT_EXHAUSTED,
) -> list[dict[str, str]]:
    """Build a compact, work-preserving continuation after a truncated tool action.

    The prior assistant bytes are not trusted or replayed. Only edits already present
    in the isolated workspace and their host-observed hashes constitute progress.
    """

    touched = sorted({str(path) for path in touched_paths})
    discarded = sorted({str(path) for path in discarded_paths})
    request = {
        "phase": "implement_module",
        "task": (
            "Continue the approved module from the current staged workspace. "
            "Do not restart completed work."
        ),
        "workspace_project_root": ".",
        "target": {
            "minecraft_version": minecraft_version,
            "loader": loader,
            "mappings": mappings,
            "java": java_version,
        },
        "module": {
            "module_id": module.module_id,
            "kind": module.kind,
            "config": module.config,
            "depends_on": list(module.depends_on),
            "required_gates": list(module.required_gates),
        },
        "continuation": {
            "reason": (
                "partial_source_tool_action_reached_context_boundary"
                if boundary_kind == CONTEXT_PRESSURE
                else "previous_source_tool_action_exhausted_output"
            ),
            "boundary_kind": boundary_kind,
            "continuation_index": continuation_index,
            "preserved_source_state_sha256": state_sha256,
            "preserved_path_count": len(touched),
            "preserved_paths_preview": touched[:_CONTINUATION_PATH_PREVIEW],
            "discarded_out_of_scope_path_count": len(discarded),
            "bounded_scalar_obligation": {
                key: obligation_receipt[key]
                for key in ("schema_version", "operation", "path")
                if obligation_receipt is not None and key in obligation_receipt
            },
        },
        "rules": [
            "Inspect the current workspace state and preserve every correct existing edit.",
            "The next tool call must be exactly one apply_source_edit action for exactly one project-relative path.",
            "Keep each later apply_source_edit call to one bounded scalar action; never emit a whole large file or multiple files in one tool call.",
            "For a large new file, create a prefix of at most 2048 characters and use append_file chunks of at most 2048 characters across tool turns.",
            "For an existing file, prefer replace_exact, insert_before or insert_after over replace_file.",
            "Do not repeat the truncated action. Do not put source code in the final summary.",
            "Continue tool turns until the approved module is implemented, then return only a concise summary.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are continuing one interrupted Minecraft mod implementation. "
                "The host preserved and hash-checked the isolated workspace. Resume "
                "with bounded source-edit tools; never reconstruct completed work from memory."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
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
            original = original_root / Path(path)
            if original.is_symlink() or not original.is_file():
                raise CustomModuleGenerationError(
                    f"Custom-module original target is not a regular file: {path}"
                )
            operation = {
                "operation": "replace",
                "path": path,
                "expected_sha256": old_sha,
                "content": content,
            }
        operations.append(operation)
        touched.append(path)
    return operations, touched, discarded


def _file_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["files", "runtime_tests"],
        "properties": {
            "files": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "purpose"],
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "purpose": {"type": "string", "minLength": 1},
                    },
                },
            },
            "runtime_tests": {"type": "array", "items": {"type": "string"}},
        },
    }


def _file_content_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content", "runtime_tests"],
        "properties": {
            "content": {"type": "string"},
            "runtime_tests": {"type": "array", "items": {"type": "string"}},
        },
    }


def _validate_file_content_payload(payload: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(payload, dict):
        raise CustomModuleGenerationError("Custom-module file content must be an object.")
    content = payload.get("content")
    if not isinstance(content, str):
        raise CustomModuleGenerationError("Custom-module file content must be UTF-8 text.")
    tests = payload.get("runtime_tests", [])
    if not isinstance(tests, list) or any(not isinstance(value, str) for value in tests):
        raise CustomModuleGenerationError("File runtime_tests must be a list of strings.")
    return content, [value.strip() for value in tests if value.strip()]


def _canonicalize_planned_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        raise CustomModuleGenerationError("Custom-module planned path must not be empty.")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts:
        raise CustomModuleGenerationError(
            f"Custom-module planned path must remain project-relative: {raw}"
        )
    path = pure.as_posix()
    if path == _FABRIC_ROOT_METADATA_PATH:
        return _FABRIC_CANONICAL_METADATA_PATH
    return path


def _select_file_context(
    ledger: dict[str, Any],
    *,
    path: str,
    purpose: str,
    byte_budget: int,
) -> dict[str, Any]:
    records = list(ledger.get("records", []))
    query_tokens = {token.lower() for token in _OBSERVATION_TOKEN.findall(f"{path} {purpose}")}
    ranked = sorted(
        records,
        key=lambda record: (
            -_observation_score(record, query_tokens),
            0 if str(record.get("path", "")) == path else 1,
            str(record.get("path", "")),
            int(record.get("content_start_bytes", 0)),
        ),
    )
    selected: list[dict[str, Any]] = []
    base = {
        "schema_version": "mmm/file-source-context-v1",
        "ledger_receipt": ledger.get("receipt", {}),
        "target_path": path,
        "records": selected,
    }
    safe_budget = max(1024, byte_budget - 256)
    for record in ranked:
        selected.append(record)
        if _json_size(base) > safe_budget:
            selected.pop()
            break
    return base


# Legacy parsing/state helpers remain for compatibility tests and old persisted traces.
def _canonicalize_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CustomModuleGenerationError("Custom-module response must be a JSON object.")
    normalized = dict(payload)
    if "tests" in normalized and "runtime_tests" not in normalized:
        normalized["runtime_tests"] = normalized.pop("tests")
    if "cursor" in normalized and "next_cursor" not in normalized:
        normalized["next_cursor"] = normalized.pop("cursor")
    if "patch_operations" in normalized and "operations" not in normalized:
        normalized["operations"] = normalized.pop("patch_operations")
    if "patches" in normalized and "operations" not in normalized:
        normalized["operations"] = normalized.pop("patches")
    operations_value = normalized.get("operations", [])
    if not isinstance(operations_value, list):
        raise CustomModuleGenerationError("Response 'operations' must be a JSON list.")
    normalized["operations"] = [_canonicalize_custom_module_operation(item) for item in operations_value]
    tests_value = normalized.get("runtime_tests", ["Verify mod functionality and compilation without crash."])
    if not isinstance(tests_value, list):
        raise CustomModuleGenerationError("Response 'runtime_tests' must be a JSON list.")
    normalized["runtime_tests"] = tests_value
    complete_value = normalized.get("complete", True)
    if not isinstance(complete_value, bool):
        raise CustomModuleGenerationError("Response 'complete' must be a JSON boolean.")
    normalized["complete"] = complete_value
    next_cursor_value = normalized.get("next_cursor", "")
    if next_cursor_value is None:
        next_cursor_value = ""
    if not isinstance(next_cursor_value, str):
        raise CustomModuleGenerationError("Response 'next_cursor' must be a JSON string.")
    normalized["next_cursor"] = next_cursor_value.strip()
    if "context_page_complete" in normalized:
        page_complete_value = normalized["context_page_complete"]
        if not isinstance(page_complete_value, bool):
            raise CustomModuleGenerationError("Response 'context_page_complete' must be a JSON boolean.")
    else:
        page_complete_value = bool(normalized["operations"]) and not normalized["next_cursor"]
    normalized["context_page_complete"] = page_complete_value
    return normalized


def _generation_fragment_action(
    payload: dict[str, Any],
    *,
    is_last_page: bool,
    has_accumulated_operations: bool,
    current_cursor: str,
    seen_cursors: set[str],
) -> str:
    operations = payload["operations"]
    next_cursor = payload["next_cursor"]
    page_complete = payload["context_page_complete"]
    if next_cursor:
        if next_cursor == current_cursor or next_cursor in seen_cursors:
            raise CustomModuleGenerationError("Custom-module response repeated next_cursor without protocol progress.")
        if page_complete:
            raise CustomModuleGenerationError("Response cannot set context_page_complete=true while also returning an advancing next_cursor for the same observation page.")
        return "cursor"
    if page_complete:
        if is_last_page and not has_accumulated_operations:
            raise CustomModuleGenerationError("Final observation page completed before any patch operation was accumulated.")
        return "page_complete"
    if operations:
        raise CustomModuleGenerationError("Patch operations were returned with context_page_complete=false but without an advancing next_cursor; the response cannot make further progress.")
    keys_str = ", ".join(payload.keys())
    raise CustomModuleGenerationError(
        "Custom-module response fragment made no protocol progress: received object with keys "
        f"[{keys_str}] but no operations, advancing next_cursor, or context_page_complete=true transition."
    )


def _repair_generation_messages(
    generation_messages: list[dict[str, str]],
    error_reason: str,
) -> list[dict[str, str]]:
    return [
        *generation_messages,
        {
            "role": "user",
            "content": (
                "Execution & Validation Failure: the previous response failed host protocol "
                f"validation with reason: {error_reason}. The invalid assistant payload is intentionally omitted so you do not copy its shape. "
                "Repair only the JSON/patch/cursor transition for the host-selected target. A valid response must make progress using patch operations, "
                "a new next_cursor, or an explicit context_page_complete=true transition. Do not emit range-only {start,end} objects. "
                "Fabric metadata belongs at src/main/resources/fabric.mod.json. Do not retrieve new RAG/MCP evidence and do not change the approved feature scope. "
                "Return exactly one valid JSON object using only the supplied host evidence."
            ),
        },
    ]


def _canonicalize_custom_module_operation(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    canonical = dict(item)
    operation = str(canonical.get("operation", "")).strip().lower()
    path = _normalized_operation_path(canonical)
    if operation == "create" and path == _FABRIC_ROOT_METADATA_PATH:
        canonical["path"] = _FABRIC_CANONICAL_METADATA_PATH
    return canonical


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    if "<|channel|>" in cleaned:
        cleaned = cleaned.split("<|channel|>")[-1]
    cleaned_strip = cleaned.strip()
    if "```" in cleaned_strip:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_strip, re.DOTALL)
        if match:
            cleaned = match.group(1)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    curly_match = re.search(r"\{.*\}", text, re.DOTALL)
    if curly_match:
        try:
            value = json.loads(curly_match.group(0))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    raise CustomModuleGenerationError("Model output did not contain one parseable JSON object; refusing a fake complete fallback.")


def _is_stale_project_index_error(exc: ValueError) -> bool:
    return str(exc).startswith("Project source changed after its context index was built:")


def _collect_initial_observations(
    router: ModelRouter,
    index: ProjectIndex,
    *,
    query: str,
    byte_budget: int,
    diagnostic_paths: Iterable[str] = (),
) -> dict[str, Any]:
    del router
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
    query_tokens = {token.lower() for token in _OBSERVATION_TOKEN.findall(query)}
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
    safe_budget = byte_budget - 128
    while cursor < len(remaining) or not pages:
        page_records: list[dict[str, Any]] = []
        while cursor < len(remaining):
            candidate_records = [*page_records, remaining[cursor]]
            candidate = _observation_page_payload(
                receipt=ledger["receipt"],
                page_index=len(pages),
                page_count=0,
                anchors=anchors,
                records=candidate_records,
                complete=False,
            )
            if _json_size(candidate) > safe_budget:
                if not page_records:
                    raise CustomModuleGenerationError("One exact source observation cannot fit the model context page.")
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
        if _json_size(page) > safe_budget:
            raise CustomModuleGenerationError("Global source anchors exceed the model context page budget.")
        pages.append(page)
        if cursor >= len(remaining):
            break
    page_count = len(pages)
    for index, page in enumerate(pages):
        page["page_count"] = page_count
        page["complete"] = index == page_count - 1
        if _json_size(page) > byte_budget:
            raise CustomModuleGenerationError("Observation context page exceeded its byte budget.")
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
            "host_requires_every_page_before_module_completion": True,
        },
    }


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
    path_tokens = {token.lower() for token in _OBSERVATION_TOKEN.findall(str(record["path"]))}
    text_tokens = {token.lower() for token in _OBSERVATION_TOKEN.findall(str(record["text"]))}
    return 60 * len(query_tokens & path_tokens) + 8 * len(query_tokens & text_tokens) + 20 * len(_ANCHOR_TERMS & text_tokens)


def _patch_operation_receipt(operations: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [_normalized_operation_path(item) for item in operations]
    return {
        "schema_version": "mmm/prior-patch-receipt-v1",
        "operation_count": len(operations),
        "operations_sha256": _sha256_json(operations),
        "touched_path_count": len(paths),
        "touched_paths_sha256": _sha256_json(paths),
        "latest_touched_path": paths[-1] if paths else "",
    }


def _normalized_operation_path(item: dict[str, Any]) -> str:
    return PurePosixPath(str(item.get("path", "")).replace("\\", "/")).as_posix()


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _mapping_namespace(value: str) -> str:
    lowered = value.strip().lower()
    if "intermediary" in lowered:
        return "intermediary"
    if "official" in lowered or "mojang" in lowered:
        return "official"
    return "yarn"


def _normalized_generation_failure(value: str) -> str:
    compact = " ".join(value.lower().split())
    compact = re.sub(r"0x[0-9a-f]+|[0-9]+", "#", compact)
    return compact[:2048]
