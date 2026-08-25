from __future__ import annotations

import json
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from .java_lsp import JavaLanguageService
from .model_router import ModelRouter
from .project_index import ProjectIndex
from .runner import GradleRunner
from .scale_policy import ScalePolicy
from .source_patch import (
    SourcePatchError,
    TransactionalSourcePatcher,
    sha256_bytes,
)


class RepairEngineError(RuntimeError):
    pass


_ALLOWED_SUFFIXES = {
    ".java",
    ".json",
    ".mcmeta",
    ".gradle",
    ".properties",
    ".accesswidener",
    ".mixins",
    ".toml",
    ".yaml",
    ".yml",
}


_ACTIVE_REPAIR_PROJECT_INDEX: ContextVar[tuple[Path, ProjectIndex] | None] = ContextVar(
    "mmm_active_repair_project_index",
    default=None,
)


def active_repair_project_index(root: Path, policy: ScalePolicy) -> ProjectIndex:
    """Return the ProjectIndex isolated to the current repair call.

    Repair diagnostics contracts call this helper instead of rebuilding the complete
    project index for every attempt. Calls made outside ``RepairEngine.repair`` retain
    the old fail-safe behavior and construct a fresh index.
    """

    normalized_root = root.expanduser().resolve()
    active = _ACTIVE_REPAIR_PROJECT_INDEX.get()
    if active is not None and active[0] == normalized_root:
        return active[1]
    return ProjectIndex(normalized_root, policy=policy)


class RepairEngine:
    """Diagnostics -> indexed context -> exact patch -> rebuild loop.

    No file-count truncation is used. The whole project is indexed and relevant files
    are selected within an explicit byte budget. By default repair is progress-driven:
    it continues while validation produces a new normalized failure state and stops on
    PASS or repeated machine evidence. ``max_attempts`` is only an explicit caller cap.
    """

    def __init__(
        self,
        *,
        router: ModelRouter,
        gradle_cache: str | Path,
        diagnostics_factory: Callable[[], JavaLanguageService] = JavaLanguageService,
        runner_factory: Callable[[Path], GradleRunner] = GradleRunner,
        policy: ScalePolicy | None = None,
    ) -> None:
        self.router = router
        self.gradle_cache = Path(gradle_cache).expanduser().resolve()
        self.diagnostics_factory = diagnostics_factory
        self.runner_factory = runner_factory
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    def repair(
        self,
        project_root: str | Path,
        *,
        run_gametest: bool = True,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise RepairEngineError("Repair target must be a regular project directory.")
        if max_attempts is not None and (
            type(max_attempts) is not int or max_attempts < 1
        ):
            raise RepairEngineError("max_attempts must be null or a positive integer.")

        # Build the complete project index exactly once for this repair invocation.
        # ContextVar keeps concurrent/nested repairs isolated without storing mutable
        # run state on the reusable RepairEngine instance.
        project_index = ProjectIndex(root, policy=self.policy)
        index_token = _ACTIVE_REPAIR_PROJECT_INDEX.set((root, project_index))
        try:
            receipts: list[dict[str, Any]] = []
            signatures: set[str] = set()
            while True:
                attempt = len(receipts)
                evidence = self._evidence(root, run_gametest=run_gametest)
                if evidence["passed"]:
                    project_index.write_manifest()
                    return {
                        "schema_version": "mmm/repair-result-v2",
                        "status": "PASS",
                        "attempts": attempt,
                        "evidence": evidence,
                        "patch_receipts": receipts,
                    }

                signature = self._signature(evidence)
                if signature in signatures:
                    raise RepairEngineError(
                        "Repair stopped because the same normalized error signature repeated; "
                        "there is no new machine-verifiable progress to justify another model call."
                    )
                signatures.add(signature)

                if max_attempts is not None and attempt >= max_attempts:
                    return {
                        "schema_version": "mmm/repair-result-v2",
                        "status": "FAIL",
                        "attempts": attempt,
                        "stop_reason": "explicit_max_attempts",
                        "evidence": evidence,
                        "patch_receipts": receipts,
                    }

                context = self._context(root, evidence)
                patch = self._request_patch(evidence, context)
                self._validate_patch_scope(patch)
                self._hydrate_repair_preconditions(root, patch)
                try:
                    receipt = TransactionalSourcePatcher(root).apply(patch)
                except SourcePatchError as exc:
                    raise RepairEngineError(
                        f"Generated repair patch was rejected: {exc}"
                    ) from exc

                # Only a successfully committed patch may mutate the in-memory index.
                # The patch contract already rejects duplicate/unsafe paths, so this is
                # the exact minimal touched set needed for the next repair attempt.
                project_index.update_files(tuple(str(item["path"]) for item in patch))
                receipts.append(receipt)
        finally:
            _ACTIVE_REPAIR_PROJECT_INDEX.reset(index_token)

    def _evidence(self, root: Path, *, run_gametest: bool) -> dict[str, Any]:
        try:
            diagnostics = self.diagnostics_factory().diagnostics(root, timeout_seconds=90)
        except Exception as exc:
            diagnostics = {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
                "diagnostics": [],
            }
        try:
            build = self.runner_factory(self.gradle_cache).build(
                root, run_gametest=run_gametest
            ).to_dict()
        except Exception as exc:
            build = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "commands": [],
            }
        diagnostic_errors = [
            item
            for item in diagnostics.get("diagnostics", {}).get("diagnostics", [])
            if isinstance(item, dict) and int(item.get("severity", 1)) <= 2
        ]
        return {
            "passed": build.get("status") == "PASS" and not diagnostic_errors,
            "diagnostics": diagnostics,
            "build": build,
        }

    @staticmethod
    def _signature(evidence: dict[str, Any]) -> str:
        diagnostics = []
        for item in evidence.get("diagnostics", {}).get("diagnostics", []):
            if not isinstance(item, dict):
                continue
            diagnostics.append(
                {
                    "path": item.get("path") or item.get("uri"),
                    "message": item.get("message"),
                    "code": item.get("code"),
                    "severity": item.get("severity"),
                }
            )
        build = evidence.get("build", {})
        return json.dumps(
            {
                "diagnostics": diagnostics,
                "build_status": build.get("status"),
                "build_error": build.get("error"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _context(self, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        diagnostic_paths: list[str] = []
        query_parts: list[str] = []
        for item in evidence.get("diagnostics", {}).get("diagnostics", []):
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("uri")
            if isinstance(path, str):
                diagnostic_paths.append(path)
            message = item.get("message")
            if isinstance(message, str):
                query_parts.append(message)
        build = evidence.get("build", {})
        if isinstance(build.get("error"), str):
            query_parts.append(build["error"])
        for command in build.get("commands", []):
            if isinstance(command, dict) and isinstance(command.get("log_path"), str):
                log = Path(command["log_path"])
                if log.is_file() and not log.is_symlink():
                    text = log.read_text(encoding="utf-8", errors="replace")
                    query_parts.append(text[-32_000:])
        index = active_repair_project_index(root, self.policy)
        return {
            "manifest": index.manifest_receipt(),
            "relevant": index.select(
                query="\n".join(query_parts),
                diagnostic_paths=diagnostic_paths,
            ),
        }

    def _request_patch(
        self,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        active = _ACTIVE_REPAIR_PROJECT_INDEX.get()
        if active is None:
            raise RepairEngineError("Repair model call has no active project index.")
        root, project_index = active
        self.router.bind_agent_workspace(root.parent, require_fresh_evidence=True)
        from .production_tools import ProductionToolService

        manifest = project_index.manifest_receipt()
        ProductionToolService(
            workspace_root=root.parent,
            profile=self.router.profile,
        ).index_project_rag(
            [root.name],
            metadata=_repair_rag_metadata(root, manifest),
            semantic=False,
        )

        prompt = {
            "task": (
                "Repair the approved Minecraft project target using exact minimal patches. "
                "Derive the exact Minecraft version, loader, mappings and Java target from "
                "the project context; never substitute a different target."
            ),
            "constraints": [
                "Return exactly one JSON object with key operations.",
                "Use only create, replace or edit operations.",
                "Every non-create operation must use the supplied exact SHA-256.",
                "Preserve the exact approved loader/version/mappings and requested functionality.",
                "Do not change platform versions merely to make the build pass.",
                "Do not emit shell commands, scripts or markdown.",
                "Use project-index paths; do not assume that omitted content means a file does not exist.",
                "Use live code/project RAG and reviewed MCP evidence for unresolved APIs, symbols, dependency and version facts; inspect retrieval quality and reformulate weak searches.",
                "Treat JDT/Gradle/GameTest failures as new observations and retrieve again when they introduce new uncertainty.",
            ],
            "evidence": evidence,
            "project_context": context,
        }
        text = self.router.generate_text(
            "coder_safe",
            [
                {
                    "role": "system",
                    "content": (
                        "You are a hash-guarded Minecraft source repair planner. "
                        "Inspect evidence with read-only tools and return patch operations; "
                        "the host transaction is the only writer."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            tool_stage="quality",
            response_format="json",
        )
        value = _extract_json(text)
        if set(value) != {"operations"}:
            raise RepairEngineError("Coder repair response must contain only operations.")
        operations = value["operations"]
        if not isinstance(operations, list) or not operations:
            raise RepairEngineError("Coder did not return a non-empty operations list.")
        encoded = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if encoded > self.policy.max_patch_bytes:
            raise RepairEngineError(
                "Repair patch exceeds MMM_MAX_PATCH_BYTES; raise the host policy explicitly."
            )
        return operations

    @staticmethod
    def _validate_patch_scope(operations: list[dict[str, Any]]) -> None:
        paths: list[str] = []
        for item in operations:
            if not isinstance(item, dict):
                raise RepairEngineError("Repair operation must be an object.")
            if item.get("operation") not in {"create", "replace", "edit"}:
                raise RepairEngineError("Automated repair may not delete files.")
            path = Path(str(item.get("path", "")))
            if path.suffix not in _ALLOWED_SUFFIXES and path.name not in {
                "build.gradle",
                "settings.gradle",
                "fabric.mod.json",
            }:
                raise RepairEngineError(f"Repair attempted an unsupported file type: {path}")
            normalized = path.as_posix()
            if normalized.startswith("/") or ".." in path.parts:
                raise RepairEngineError(f"Repair attempted an unsafe path: {path}")
            paths.append(normalized)
        if len(paths) != len(set(paths)):
            raise RepairEngineError("Repair must combine multiple edits to the same path.")

    @staticmethod
    def _hydrate_repair_preconditions(root: Path, operations: list[dict[str, Any]]) -> None:
        for item in operations:
            if not isinstance(item, dict):
                continue
            rel_str = str(item.get("path", "")).strip()
            if not rel_str or rel_str.startswith("/") or ".." in rel_str:
                continue
            target = root / rel_str
            op = str(item.get("operation", "")).strip().lower()
            if op in {"replace", "edit", "delete"}:
                expected = str(item.get("expected_sha256", "")).strip()
                if not expected and target.is_file() and not target.is_symlink():
                    item["expected_sha256"] = sha256_bytes(target.read_bytes())


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
    raise RepairEngineError("Coder repair response did not contain a JSON object.")


def _repair_rag_metadata(project_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    lock_path = project_root / ".minecraft_ai" / "platform-lock.json"
    if not lock_path.is_file() or lock_path.is_symlink():
        raise RepairEngineError("Repair RAG requires the project's exact platform lock.")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairEngineError("Repair RAG platform lock is unreadable.") from exc
    mappings = str(lock.get("yarn_mappings", lock.get("mappings", ""))).strip().lower()
    if "intermediary" in mappings:
        namespace = "intermediary"
    elif "official" in mappings or "mojang" in mappings:
        namespace = "official"
    else:
        namespace = "yarn"
    version = str(lock.get("minecraft_version", "")).strip()
    loader = str(lock.get("loader", "")).strip()
    java_version = str(lock.get("java_version", "")).strip()
    if not version or not loader or not java_version:
        raise RepairEngineError("Repair RAG platform lock is incomplete.")
    return {
        "minecraft_version": version,
        "loader": loader,
        "mapping_namespace": namespace,
        "java_version": java_version,
        "license": os.environ.get("MMM_PROJECT_LICENSE", "project-local").strip() or "project-local",
        "source_commit": str(manifest["sha256"]),
    }
