from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .java_lsp import JavaLanguageService
from .model_router import ModelRouter
from .runner import GradleRunner
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
}


class RepairEngine:
    """Finite diagnostics -> exact patch -> rebuild loop.

    The model never receives shell access. It can only return hash-guarded text patch
    operations, which are applied transactionally and revalidated by machine evidence.
    """

    def __init__(
        self,
        *,
        router: ModelRouter,
        gradle_cache: str | Path,
        diagnostics_factory: Callable[[], JavaLanguageService] = JavaLanguageService,
        runner_factory: Callable[[Path], GradleRunner] = GradleRunner,
    ) -> None:
        self.router = router
        self.gradle_cache = Path(gradle_cache).expanduser().resolve()
        self.diagnostics_factory = diagnostics_factory
        self.runner_factory = runner_factory

    def repair(
        self,
        project_root: str | Path,
        *,
        run_gametest: bool = True,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise RepairEngineError("Repair target must be a regular project directory.")
        if not 1 <= max_attempts <= 5:
            raise RepairEngineError("max_attempts must be 1-5.")
        receipts: list[dict[str, Any]] = []
        signatures: set[str] = set()
        for attempt in range(max_attempts + 1):
            evidence = self._evidence(root, run_gametest=run_gametest)
            if evidence["passed"]:
                return {
                    "schema_version": "mmm/repair-result-v1",
                    "status": "PASS",
                    "attempts": attempt,
                    "evidence": evidence,
                    "patch_receipts": receipts,
                }
            if attempt >= max_attempts:
                break
            signature = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
            if signature in signatures:
                raise RepairEngineError("Repair stopped because the same error signature repeated.")
            signatures.add(signature)
            snapshot = self._snapshot(root, evidence)
            patch = self._request_patch(evidence, snapshot)
            self._validate_patch_scope(patch)
            try:
                receipt = TransactionalSourcePatcher(root).apply(patch)
            except SourcePatchError as exc:
                raise RepairEngineError(f"Generated repair patch was rejected: {exc}") from exc
            receipts.append(receipt)
        return {
            "schema_version": "mmm/repair-result-v1",
            "status": "FAIL",
            "attempts": max_attempts,
            "evidence": self._evidence(root, run_gametest=run_gametest),
            "patch_receipts": receipts,
        }

    def _evidence(self, root: Path, *, run_gametest: bool) -> dict[str, Any]:
        diagnostics: dict[str, Any]
        try:
            diagnostics = self.diagnostics_factory().diagnostics(root, timeout_seconds=90)
        except Exception as exc:
            diagnostics = {
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
                "diagnostics": [],
            }
        try:
            build = self.runner_factory(self.gradle_cache).build(root, run_gametest=run_gametest).to_dict()
        except Exception as exc:
            build = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "commands": []}
        diagnostic_errors = [
            item
            for item in diagnostics.get("diagnostics", [])
            if isinstance(item, dict) and int(item.get("severity", 1)) <= 2
        ]
        passed = build.get("status") == "PASS" and not diagnostic_errors
        return {
            "passed": passed,
            "diagnostics": diagnostics,
            "build": build,
        }

    def _snapshot(self, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        candidates: set[str] = set()
        for item in evidence.get("diagnostics", {}).get("diagnostics", []):
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("uri")
            if isinstance(path, str):
                try:
                    relative = Path(path.removeprefix("file://")).resolve().relative_to(root).as_posix()
                except Exception:
                    continue
                candidates.add(relative)
        if not candidates:
            for path in sorted(root.rglob("*.java"))[:12]:
                if path.is_file() and not path.is_symlink():
                    candidates.add(path.relative_to(root).as_posix())
            for fixed in ("build.gradle", "gradle.properties", "src/main/resources/fabric.mod.json"):
                if (root / fixed).is_file():
                    candidates.add(fixed)
        return TransactionalSourcePatcher(root).snapshot(sorted(candidates)[:24])

    def _request_patch(self, evidence: dict[str, Any], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = {
            "task": "Repair the Minecraft Java 1.20.1 Fabric project using the smallest exact patch.",
            "constraints": [
                "Return exactly one JSON object with key operations.",
                "Use only create, replace, edit or delete operations.",
                "Every non-create operation must use the supplied exact SHA-256.",
                "Do not delete requested functionality or mix loaders/versions.",
                "Do not emit shell commands, scripts or markdown.",
            ],
            "evidence": evidence,
            "snapshot": snapshot,
        }
        text = self.router.generate_text(
            "coder",
            [
                {"role": "system", "content": "You are a bounded Fabric 1.20.1 source repair agent."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format="json",
        )
        value = _extract_json(text)
        operations = value.get("operations")
        if not isinstance(operations, list) or not operations:
            raise RepairEngineError("Coder did not return a non-empty operations list.")
        return operations

    def _validate_patch_scope(self, operations: list[dict[str, Any]]) -> None:
        if len(operations) > 32:
            raise RepairEngineError("A repair attempt may touch at most 32 files.")
        for item in operations:
            if not isinstance(item, dict):
                raise RepairEngineError("Repair operation must be an object.")
            path = Path(str(item.get("path", "")))
            if path.suffix not in _ALLOWED_SUFFIXES and path.name not in {
                "build.gradle",
                "settings.gradle",
                "fabric.mod.json",
            }:
                raise RepairEngineError(f"Repair attempted an unsupported file type: {path}")
            if item.get("operation") == "delete":
                raise RepairEngineError("Automated repair may not delete files.")


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
