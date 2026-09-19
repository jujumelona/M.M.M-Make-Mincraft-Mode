from __future__ import annotations

from .fixed_template_generation import generate_fixed_template_text

from .model_response_templates import response_schema, response_template_prompt

import hashlib
import json
import os
import re
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .java_lsp import JavaLanguageService
from .validation_diagnostic_contract import (
    diagnostic_errors as _diagnostic_errors,
    diagnostic_items as _diagnostic_items,
    run_diagnostics as _run_jdt_diagnostics,
)
from .model_router import ModelRouter
from .project_index import ProjectIndex
from .runner import GradleRunner
from .scale_policy import ScalePolicy
from .source_patch import (
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
_REPAIR_LOG_SNIPPET_CHARS = 6000
_REPAIR_LOG_READ_BYTES = 32768
_JAVAC_LINE = re.compile(
    r"^[ \t]*(?:[A-Za-z]:)?[^\r\n]+\.java:\d+:\s*(?:error|warning):",
    re.IGNORECASE,
)
_REPAIR_BUILD_LOG_LIMIT = 4
_REPAIR_QUERY_PART_LIMIT = 12


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



def _read_bounded_build_log(log_path: Any) -> str:
    """Keep compiler diagnostics even when Gradle appends a long stack trace."""

    raw_path = str(log_path or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    try:
        if not path.is_file() or path.is_symlink():
            return ""
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(_REPAIR_LOG_READ_BYTES)
            if size > _REPAIR_LOG_READ_BYTES:
                handle.seek(max(0, size - _REPAIR_LOG_READ_BYTES))
                tail = handle.read(_REPAIR_LOG_READ_BYTES)
            else:
                tail = b""
    except (OSError, ValueError):
        return ""

    normalized = (head + (b"\n" if tail else b"") + tail).decode(
        "utf-8", errors="replace"
    ).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()
    compiler: list[str] = []
    for index, line in enumerate(lines):
        if _JAVAC_LINE.search(line):
            compiler.extend(lines[index : index + 5])
    compiler_text = "\n".join(dict.fromkeys(compiler))
    tail_text = "\n".join(lines)[-_REPAIR_LOG_SNIPPET_CHARS // 2 :]
    combined = "\n".join(
        part for part in (compiler_text, tail_text) if part.strip()
    ).strip()
    return combined[-_REPAIR_LOG_SNIPPET_CHARS:]


def _failed_build_log_diagnostics(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    build = evidence.get("build")
    if not isinstance(build, dict):
        return []
    commands = [item for item in build.get("commands", []) if isinstance(item, dict)]
    failed = [
        item
        for item in commands
        if bool(item.get("timed_out"))
        or (
            isinstance(item.get("exit_code"), int)
            and not isinstance(item.get("exit_code"), bool)
            and item.get("exit_code") != 0
        )
    ]
    if not failed and build.get("status") == "FAIL" and commands:
        failed = [commands[-1]]

    diagnostics: list[dict[str, Any]] = []
    for command in failed[-_REPAIR_BUILD_LOG_LIMIT:]:
        output = _read_bounded_build_log(command.get("log_path"))
        item: dict[str, Any] = {
            "name": command.get("name"),
            "exit_code": command.get("exit_code"),
            "timed_out": bool(command.get("timed_out")),
        }
        if output:
            item["output"] = output
        diagnostics.append(item)
    return diagnostics


class RepairEngine:
    """Diagnostics -> indexed context -> exact patch -> rebuild loop.

    No file-count truncation is used. The whole project is indexed and relevant files
    are selected within an explicit byte budget. Repair remains progress-sensitive:
    repeated verifier signatures terminate semantic fixed points. ``max_attempts``
    is an optional explicit caller policy; None means no arbitrary attempt-count cap.
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

        # With no explicit caller limit, convergence is semantic rather than
        # attempt-count based: a repeated verifier signature terminates the loop.
        attempt_limit = max_attempts

        # Build the complete project index exactly once for this repair invocation.
        # ContextVar keeps concurrent/nested repairs isolated without storing mutable
        # run state on the reusable RepairEngine instance.
        project_index = ProjectIndex(root, policy=self.policy)
        index_token = _ACTIVE_REPAIR_PROJECT_INDEX.set((root, project_index))
        try:
            receipts: list[dict[str, Any]] = []
            signatures: set[str] = set()
            repair_attempts = 0
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
                    return {
                        "schema_version": "mmm/repair-result-v2",
                        "status": "FAIL",
                        "attempts": attempt,
                        "stop_reason": "repeated_signature",
                        "evidence": evidence,
                        "patch_receipts": receipts,
                    }
                signatures.add(signature)

                if attempt_limit is not None and repair_attempts >= attempt_limit:
                    return {
                        "schema_version": "mmm/repair-result-v2",
                        "status": "FAIL",
                        "attempts": attempt,
                        "stop_reason": "explicit_max_attempts",
                        "evidence": evidence,
                        "patch_receipts": receipts,
                    }

                context = self._context(root, evidence)
                repair_attempts += 1
                try:
                    patch = self._request_patch(evidence, context)
                    if not patch:
                        print("  [!] Repair attempt produced no patch operations (retrying)", flush=True)
                        continue
                    self._hydrate_repair_preconditions(root, patch)
                    self._validate_patch_scope(patch)
                    if not patch:
                        print("  [!] Repair operations empty after scope validation (retrying)", flush=True)
                        continue
                    receipt = TransactionalSourcePatcher(root).apply(patch)
                except Exception as exc:
                    print(
                        f"  [!] Repair patch application failed (retrying next attempt): {exc}",
                        flush=True,
                    )
                    continue

                # Only a successfully committed patch may mutate the in-memory index.
                # The patch contract already rejects duplicate/unsafe paths, so this is
                # the exact minimal touched set needed for the next repair attempt.
                project_index.update_files(tuple(str(item["path"]) for item in patch))
                # Keep durable retrieval synchronized with the committed source edit.
                try:
                    from .production_tools import ProjectRAGIndex
                    rag_index = ProjectRAGIndex(root / ".minecraft_ai" / "rag_index")
                    rag_index.build(
                        [root],
                        metadata=_repair_rag_metadata(root, project_index.manifest_receipt()),
                        router=None,
                        semantic=False,
                    )
                except Exception as exc:
                    print(
                        f"  [!] Project RAG refresh after repair edit failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                receipts.append(receipt)
        finally:
            _ACTIVE_REPAIR_PROJECT_INDEX.reset(index_token)

    def _evidence(self, root: Path, *, run_gametest: bool) -> dict[str, Any]:
        diagnostics = _run_jdt_diagnostics(
            self.diagnostics_factory,
            root,
            timeout_seconds=90,
        )
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
        diagnostic_errors = _diagnostic_errors(diagnostics)
        return {
            "passed": build.get("status") == "PASS" and not diagnostic_errors,
            "diagnostics": diagnostics,
            "build": build,
        }

    @staticmethod
    def _signature(evidence: dict[str, Any]) -> str:
        diagnostics = []
        for item in _diagnostic_items(evidence.get("diagnostics")):
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
        build_logs = []
        for item in _failed_build_log_diagnostics(evidence):
            output = str(item.get("output") or "")
            build_logs.append(
                {
                    "name": item.get("name"),
                    "exit_code": item.get("exit_code"),
                    "timed_out": item.get("timed_out"),
                    "output_sha256": (
                        hashlib.sha256(output.encode("utf-8")).hexdigest()
                        if output
                        else ""
                    ),
                }
            )
        return json.dumps(
            {
                "diagnostics": diagnostics,
                "build_status": build.get("status"),
                "build_error": build.get("error"),
                "build_logs": build_logs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _context(self, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        """Build bounded exact repository grounding directly at the repair owner."""

        from .model_context_budget import request_message_budget
        from .repository_grounding import build_repair_repository_context

        diagnostic_paths: list[str] = []
        query_parts: list[str] = []
        for item in _diagnostic_items(evidence.get("diagnostics")):
            if not isinstance(item, dict):
                continue
            path_value = item.get("path") or item.get("uri")
            if isinstance(path_value, str) and path_value.strip():
                diagnostic_paths.append(path_value)
            message = item.get("message")
            if isinstance(message, str) and message.strip():
                query_parts.append(message)

        build = evidence.get("build")
        build = build if isinstance(build, dict) else {}
        build_error = build.get("error")
        if isinstance(build_error, str) and build_error.strip():
            query_parts.append(build_error)

        build_logs = _failed_build_log_diagnostics(evidence)
        for command in build_logs:
            output = command.get("output")
            if isinstance(output, str) and output:
                query_parts.append(output)

        query = "\n".join(query_parts[-_REPAIR_QUERY_PART_LIMIT:]).strip()
        if not query:
            query = json.dumps(
                {
                    "diagnostics_status": (
                        evidence.get("diagnostics", {}).get("status")
                        if isinstance(evidence.get("diagnostics"), dict)
                        else None
                    ),
                    "build_status": build.get("status"),
                },
                sort_keys=True,
            )

        requested = max(1024, int(self.policy.model_context_bytes))
        registry = getattr(self.router, "registry", None)
        resolve = getattr(registry, "role", None)
        profile = str(getattr(self.router, "profile", "") or "").strip()
        if callable(resolve) and profile:
            try:
                config = resolve(profile, "coder_safe")
                live_request_bytes = request_message_budget(config, ())
                byte_budget = min(
                    requested,
                    max(4 * 1024, int(live_request_bytes) // 2),
                )
            except Exception:
                byte_budget = min(requested, 32 * 1024)
        else:
            byte_budget = min(requested, 32 * 1024)

        index = active_repair_project_index(root, self.policy)
        context = build_repair_repository_context(
            self.router,
            index,
            query=query,
            diagnostic_paths=diagnostic_paths,
            byte_budget=byte_budget,
        )
        context["build_logs"] = build_logs
        return context

    def _request_patch(
        self,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prompt = {
            "task": "repair_project",
            "instructions": (
                "Emit minimal source patch operations to resolve build/diagnostic errors. "
                "Derive the exact Minecraft version, loader, mappings and Java target from "
                "the project context; never substitute a different target."
            ),
            "constraints": [
                "Fill the supplied fixed template with the operations field.",
                "Use only create, replace or edit operations.",
                "Every non-create operation must use the supplied exact SHA-256.",
                "Preserve the exact approved loader/version/mappings and requested functionality.",
                "Do not change platform versions merely to make the build pass.",
                "Do not emit shell commands, scripts or markdown.",
                "Use project-index paths; do not assume that omitted content means a file does not exist.",
            ],
            "evidence": evidence,
            "project_context": context,
        }
        try:
            text = generate_fixed_template_text(self.router,
                "coder_safe",
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a hash-guarded Minecraft source repair planner. "
                            "Inspect evidence with read-only tools and return patch operations; "
                            "the host transaction is the only writer."
                            + response_template_prompt("repair")
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                tool_stage="quality",
                response_schema=response_schema("repair"),
            )
        except Exception as exc:
            print(f"  [!] Repair coder model call failed ({type(exc).__name__}: {exc}); skipping attempt", flush=True)
            return []
        value = _extract_json(text)
        operations = value.get("operations") if isinstance(value, dict) and set(value) == {"operations"} else None
        if not isinstance(operations, list) or not operations:
            return []
        encoded = len(json.dumps(operations, ensure_ascii=False).encode("utf-8"))
        if encoded > self.policy.max_patch_bytes:
            return []
        return operations

    @staticmethod
    def _validate_patch_scope(operations: list[dict[str, Any]]) -> None:
        paths: list[str] = []
        valid_operations: list[dict[str, Any]] = []
        for item in operations:
            if not isinstance(item, dict):
                continue
            if item.get("operation") not in {"create", "replace", "edit", "delete"}:
                continue
            if item.get("operation") == "edit" and (not isinstance(item.get("replacements"), list) or not item.get("replacements")):
                continue
            path_str = str(item.get("path", "")).strip()
            if not path_str:
                continue
            path = Path(path_str)
            if path.suffix not in _ALLOWED_SUFFIXES and path.name not in {
                "build.gradle",
                "settings.gradle",
                "fabric.mod.json",
            }:
                continue
            normalized = path.as_posix()
            if normalized.startswith("/") or ".." in path.parts or any(part.startswith(".") for part in path.parts):
                continue
            if normalized in paths:
                continue
            paths.append(normalized)
            valid_operations.append(item)
        operations[:] = valid_operations

    @staticmethod
    def _hydrate_repair_preconditions(root: Path, operations: list[dict[str, Any]]) -> None:
        for index, item in enumerate(operations):
            if not isinstance(item, dict):
                continue
            rel_str = str(item.get("path", "")).strip().replace("\\", "/")
            while rel_str.startswith("./"):
                rel_str = rel_str[2:]
            if not rel_str or rel_str.startswith("/") or ".." in rel_str.split("/"):
                continue
            target = root / rel_str
            if not target.exists() and rel_str.startswith(f"{root.name}/"):
                sub = rel_str[len(root.name) + 1 :]
                if (root / sub).exists():
                    rel_str = sub
                    target = root / rel_str
            item["path"] = rel_str

            op = str(item.get("operation", "")).strip().lower()
            if op in {"modify", "patch", "update", "write_file", "replace_exact", "insert", "insert_before", "insert_after", "append", "prepend", "insert_member", "insert_java_member", "add_member", "add_import", "add_java_import", "import"}:
                op = "replace"
                item["operation"] = op
            elif op in {"write", "add", "create_file", "create_class", "create_type", "create_java_type", "create_java_class"}:
                op = "create"
                item["operation"] = op
            elif op in {"delete", "delete_file", "remove", "remove_file"}:
                op = "delete"
                item["operation"] = op

            # Normalize edit operations with missing/empty replacements to replace
            if op == "edit":
                replacements = item.get("replacements")
                if not isinstance(replacements, list) or not replacements:
                    alt_content = item.get("content") or item.get("new") or item.get("new_text") or item.get("new_content") or item.get("code") or item.get("source") or item.get("body") or item.get("text")
                    if alt_content is not None:
                        op = "replace"
                        item["operation"] = "replace"
                        item["content"] = str(alt_content)
                    elif isinstance(replacements, dict) and "old" in replacements and "new" in replacements:
                        item["replacements"] = [replacements]
                elif isinstance(replacements, list):
                    clean_replacements = []
                    for rep in replacements:
                        if isinstance(rep, dict) and "old" in rep and "new" in rep:
                            clean_replacements.append({
                                "old": str(rep["old"]),
                                "new": str(rep["new"]),
                                "count": int(rep.get("count", 1) or 1),
                            })
                    if clean_replacements:
                        item["replacements"] = clean_replacements
                    else:
                        alt_content = item.get("content") or item.get("new") or item.get("new_text")
                        if alt_content is not None:
                            op = "replace"
                            item["operation"] = "replace"
                            item["content"] = str(alt_content)

            # Normalize content payload
            if op in {"create", "replace"}:
                if "content" not in item:
                    alt_content = item.get("new") or item.get("new_text") or item.get("new_content") or item.get("code") or item.get("source") or item.get("body") or item.get("text")
                    if alt_content is not None:
                        item["content"] = str(alt_content)
                    else:
                        item["content"] = ""

            if op in {"replace", "edit", "delete"}:
                if not target.is_file() or target.is_symlink():
                    if op in {"replace", "edit"}:
                        op = "create"
                        item["operation"] = "create"
                        item.pop("expected_sha256", None)
                else:
                    item["expected_sha256"] = sha256_bytes(target.read_bytes())
            elif op == "create" and target.is_file() and not target.is_symlink():
                op = "replace"
                item["operation"] = "replace"
                item["expected_sha256"] = sha256_bytes(target.read_bytes())

            # Strip disallowed metadata fields so TransactionalSourcePatcher strictly validates
            allowed_fields = {
                "create": {"operation", "path", "content"},
                "replace": {"operation", "path", "expected_sha256", "content"},
                "edit": {"operation", "path", "expected_sha256", "replacements"},
                "delete": {"operation", "path", "expected_sha256"},
            }.get(item.get("operation", ""), set())
            if allowed_fields:
                extra_keys = set(item.keys()) - allowed_fields
                for k in extra_keys:
                    item.pop(k, None)


def _extract_json(text: str) -> Any:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in ("{", "["):
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError:
            continue
    return {}


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
