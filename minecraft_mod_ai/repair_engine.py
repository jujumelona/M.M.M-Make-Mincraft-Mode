from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .java_lsp import JavaLanguageService
from .model_router import ModelRouter
from .project_index import ProjectIndex
from .runner import GradleRunner
from .scale_policy import ScalePolicy
from .source_patch import SourcePatchError, TransactionalSourcePatcher


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


class RepairEngine:
    """Diagnostics -> indexed context -> exact patch -> rebuild loop.

    No file-count truncation is used. The whole project is indexed and relevant files
    are selected within an explicit byte budget. Repair attempts and patch bytes are
    operational policy values, not feature-schema limits.
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
        attempts_limit = self.policy.repair_attempts if max_attempts is None else max_attempts
        if type(attempts_limit) is not int or attempts_limit < 1:
            raise RepairEngineError("max_attempts must be a positive integer.")

        receipts: list[dict[str, Any]] = []
        signatures: set[str] = set()
        for attempt in range(attempts_limit + 1):
            evidence = self._evidence(root, run_gametest=run_gametest)
            if evidence["passed"]:
                ProjectIndex(root, policy=self.policy).write_manifest()
                return {
                    "schema_version": "mmm/repair-result-v2",
                    "status": "PASS",
                    "attempts": attempt,
                    "evidence": evidence,
                    "patch_receipts": receipts,
                }
            if attempt >= attempts_limit:
                break
            signature = self._signature(evidence)
            if signature in signatures:
                raise RepairEngineError(
                    "Repair stopped because the same normalized error signature repeated."
                )
            signatures.add(signature)
            context = self._context(root, evidence)
            patch = self._request_patch(evidence, context)
            self._validate_patch_scope(patch)
            try:
                receipt = TransactionalSourcePatcher(root).apply(patch)
            except SourcePatchError as exc:
                raise RepairEngineError(f"Generated repair patch was rejected: {exc}") from exc
            receipts.append(receipt)

        return {
            "schema_version": "mmm/repair-result-v2",
            "status": "FAIL",
            "attempts": attempts_limit,
            "evidence": self._evidence(root, run_gametest=run_gametest),
            "patch_receipts": receipts,
        }

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
            for item in diagnostics.get("diagnostics", [])
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
        index = ProjectIndex(root, policy=self.policy)
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
        prompt = {
            "task": "Repair the Minecraft Java 1.20.1 Fabric project using exact minimal patches.",
            "constraints": [
                "Return exactly one JSON object with key operations.",
                "Use only create, replace or edit operations.",
                "Every non-create operation must use the supplied exact SHA-256.",
                "Do not delete requested functionality or mix loaders/versions.",
                "Do not emit shell commands, scripts or markdown.",
                "Use project-index paths; do not assume that omitted content means a file does not exist.",
            ],
            "evidence": evidence,
            "project_context": context,
        }
        text = self.router.generate_text(
            "coder",
            [
                {
                    "role": "system",
                    "content": "You are a hash-guarded Fabric source repair agent.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
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
