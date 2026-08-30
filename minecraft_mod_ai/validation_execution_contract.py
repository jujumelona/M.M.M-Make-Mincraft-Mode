from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Any

from .research_validation_fingerprint_performance import content_digest
from .runner import BuildRunnerError
from .validation_diagnostic_contract import (
    diagnostic_errors as _diagnostic_errors,
    run_diagnostics as _run_jdt_diagnostics,
)

_CACHE_LOCK = threading.RLock()
_SUCCESSFUL_BUILDS: dict[tuple[Any, ...], Any] = {}
_JDT_RESULTS: dict[tuple[Any, ...], dict[str, Any]] = {}
_CACHE_LIMIT = 24

_SKIP_TOP_LEVEL = {
    ".git",
    ".gradle",
    "build",
    "logs",
    "node_modules",
    "run",
}
_SKIP_STATE_DIRECTORIES = {
    ".minecraft_ai/logs",
    ".minecraft_ai/runtime",
    ".minecraft_ai/validation-cache",
}
_SKIP_PREFIXES = tuple(f"{value}/" for value in sorted(_SKIP_STATE_DIRECTORIES))
_SKIP_WRAPPER_PATHS = {
    "gradlew",
    "gradlew.bat",
    "gradle/wrapper/gradle-wrapper.jar",
    "gradle/wrapper/gradle-wrapper.properties",
}
_JAVA_CONFIG_FILES = ("build.gradle", "settings.gradle", "gradle.properties")


def _bounded_put(mapping: dict[Any, Any], key: Any, value: Any) -> None:
    mapping.pop(key, None)
    mapping[key] = value
    while len(mapping) > _CACHE_LIMIT:
        mapping.pop(next(iter(mapping)))


def _canonical_project_root(project_root: str | Path) -> Path:
    """Resolve one existing project directory without traversing symlink aliases."""

    lexical = Path(os.path.abspath(os.fspath(Path(project_root).expanduser())))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("Validation project root may not traverse symbolic links.")
    try:
        root = lexical.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise FileNotFoundError(lexical) from exc
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _is_build_input(relative: str) -> bool:
    path = Path(relative)
    if path.parts and path.parts[0] in _SKIP_TOP_LEVEL:
        return False
    if relative in _SKIP_WRAPPER_PATHS:
        return False
    return not any(relative.startswith(prefix) for prefix in _SKIP_PREFIXES)


def _skip_build_directory(relative: str) -> bool:
    path = Path(relative)
    if path.parts and path.parts[0] in _SKIP_TOP_LEVEL:
        return True
    return relative in _SKIP_STATE_DIRECTORIES


def _iter_build_inputs(root: Path) -> tuple[tuple[str, Path], ...]:
    """Walk build inputs deterministically and reject relevant symlink aliases."""

    values: list[tuple[str, Path]] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current)
        kept_directories: list[str] = []
        for name in sorted(dirnames):
            child = directory / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                if not _skip_build_directory(relative):
                    raise ValueError(
                        f"Build input directory traverses a symbolic link: {relative}"
                    )
                continue
            if not _skip_build_directory(relative):
                kept_directories.append(name)
        dirnames[:] = kept_directories

        for name in sorted(filenames):
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                if _is_build_input(relative):
                    raise ValueError(
                        f"Build input file is a symbolic link: {relative}"
                    )
                continue
            if not path.is_file():
                continue
            if _is_build_input(relative):
                values.append((relative, path))
    return tuple(values)


def project_build_fingerprint(project_root: str | Path) -> str:
    """Hash exact build inputs while pruning outputs, caches, logs, and symlinks."""

    root = _canonical_project_root(project_root)
    digest = hashlib.sha256()
    digest.update(b"mmm/build-input-fingerprint-v4\0")
    for relative, path in _iter_build_inputs(root):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest(path))
        digest.update(b"\0")
    return digest.hexdigest()


def _validated_project_file(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or not raw.parts or any(
        part in {"", ".", ".."} for part in raw.parts
    ):
        raise ValueError("Validation input path must be canonical and project-relative.")

    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Validation input path traversed a symbolic link.")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise ValueError("Validation input escaped the project root.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _java_fingerprint(
    project_root: str | Path,
    relative_files: Iterable[str] | None,
) -> tuple[str, tuple[str, ...]]:
    root = _canonical_project_root(project_root)

    if relative_files is None:
        candidates = sorted(
            (
                path
                for path in root.rglob("*.java")
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.as_posix(),
        )
        relative = tuple(path.relative_to(root).as_posix() for path in candidates)
        paths = tuple(_validated_project_file(root, value) for value in relative)
    else:
        relative = tuple(
            sorted(set(str(value).replace("\\", "/") for value in relative_files))
        )
        paths = tuple(_validated_project_file(root, value) for value in relative)

    digest = hashlib.sha256()
    digest.update(b"mmm/java-validation-fingerprint-v4\0")
    for config_name in _JAVA_CONFIG_FILES:
        config = root / config_name
        if not config.exists() and not config.is_symlink():
            continue
        config = _validated_project_file(root, config_name)
        digest.update(config_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest(config))
        digest.update(b"\0")
    for rel, path in zip(relative, paths, strict=True):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest(path))
        digest.update(b"\0")
    return digest.hexdigest(), relative


def _validated_log_file(root: Path, log_path: str | Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(log_path).expanduser())))
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("GameTest log escaped the project root.") from exc
    return _validated_project_file(root, relative)


def gametest_resource_errors(
    project_root: str | Path,
    log_path: str | Path,
) -> tuple[str, ...]:
    """Return resource errors, failing closed when GameTest evidence is unavailable."""

    try:
        root = _canonical_project_root(project_root)
        fabric = _validated_project_file(root, "src/main/resources/fabric.mod.json")
    except (FileNotFoundError, OSError, ValueError) as exc:
        return (
            "GameTest resource validation unavailable: fabric.mod.json or project root "
            f"is missing or unsafe ({type(exc).__name__}: {exc}).",
        )
    try:
        payload = json.loads(fabric.read_text(encoding="utf-8"))
        raw_mod_id = payload["id"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return ("GameTest resource validation unavailable: fabric.mod.json is unreadable or invalid.",)
    if (
        type(raw_mod_id) is not str
        or not raw_mod_id
        or raw_mod_id != raw_mod_id.strip()
        or any(ord(character) < 0x20 for character in raw_mod_id)
    ):
        return ("GameTest resource validation unavailable: fabric.mod.json has an invalid mod id.",)

    try:
        path = _validated_log_file(root, log_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return (
            "GameTest resource validation unavailable: GameTest log is missing or unsafe "
            f"({type(exc).__name__}: {exc}).",
        )

    namespace = f"{raw_mod_id}:"
    markers = (
        "couldn't parse element",
        "parsing error loading",
        "failed to parse",
        "couldn't load",
        "error loading",
        "unknown item",
        "unknown block",
        "unknown registry",
    )
    findings: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                lowered = raw.lower()
                if namespace not in raw or not any(marker in lowered for marker in markers):
                    continue
                compact = raw.strip()
                if compact and compact not in findings:
                    findings.append(compact[:2000])
    except OSError as exc:
        return (f"GameTest resource validation unavailable: {type(exc).__name__}: {exc}",)
    return tuple(findings)


def _jdt_cache_profile(self: Any, *, timeout_seconds: int) -> tuple[Any, ...]:
    """Capture every service/runtime setting that can change a JDT receipt."""

    return (
        tuple(str(value) for value in getattr(self, "command", ())),
        int(getattr(self, "diagnostic_page_max_files", 0)),
        int(getattr(self, "diagnostic_page_max_source_bytes", 0)),
        float(getattr(self, "diagnostic_quiet_seconds", 0.0)),
        int(timeout_seconds),
        os.environ.get("JAVA_HOME", ""),
        os.environ.get("JDK_HOME", ""),
        os.environ.get("PATH", ""),
    )


def _install_jdt_cache(java_lsp_module: Any) -> None:
    cls = java_lsp_module.JavaLanguageService
    original = cls.diagnostics
    if getattr(original, "_mmm_exact_java_cache", False):
        return

    @wraps(original)
    def cached_diagnostics(
        self: Any,
        project_root: str | Path,
        *,
        relative_files: Iterable[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        try:
            root = _canonical_project_root(project_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise java_lsp_module.JDTLanguageServerError(
                f"JDT validation project root is missing or unsafe: {exc}"
            ) from exc
        full_scope = relative_files is None

        files = tuple(java_lsp_module._java_files(root, relative_files))
        normalized = tuple(path.relative_to(root).as_posix() for path in files)
        fingerprint, fingerprint_files = _java_fingerprint(root, normalized)
        if fingerprint_files != normalized:
            raise java_lsp_module.JDTLanguageServerError(
                "JDT validation file identity changed during cache preparation."
            )

        profile = _jdt_cache_profile(self, timeout_seconds=timeout_seconds)
        key = (str(root), fingerprint, normalized, profile)
        with _CACHE_LOCK:
            cached = _JDT_RESULTS.get(key)
            if cached is not None:
                return copy.deepcopy(cached)

        result = original(
            self,
            root,
            relative_files=normalized,
            timeout_seconds=timeout_seconds,
        )

        if full_scope:
            final_files = tuple(java_lsp_module._java_files(root, None))
            final_normalized = tuple(
                path.relative_to(root).as_posix() for path in final_files
            )
            if final_normalized != normalized:
                raise java_lsp_module.JDTLanguageServerError(
                    "Java source set changed during JDT validation; result is not certifiable."
                )

        final_fingerprint, final_files = _java_fingerprint(root, normalized)
        if final_files != normalized or final_fingerprint != fingerprint:
            raise java_lsp_module.JDTLanguageServerError(
                "Java/config inputs changed during JDT validation; result is not certifiable."
            )

        with _CACHE_LOCK:
            _bounded_put(_JDT_RESULTS, key, copy.deepcopy(result))
        return result

    cached_diagnostics._mmm_exact_java_cache = True
    cached_diagnostics._mmm_snapshot_stable_java_cache = True
    cls.diagnostics = cached_diagnostics


def _install_progressive_repair(repair_module: Any) -> None:
    cls = repair_module.RepairEngine
    original_request_patch = cls._request_patch
    if not getattr(original_request_patch, "_mmm_tracks_repair_scope", False):

        @wraps(original_request_patch)
        def scoped_request_patch(
            self: Any,
            evidence: dict[str, Any],
            context: dict[str, Any],
        ) -> Any:
            operations = original_request_patch(self, evidence, context)
            self._mmm_last_java_paths = tuple(
                sorted(
                    str(item.get("path", "")).replace("\\", "/")
                    for item in operations
                    if str(item.get("path", "")).lower().endswith(".java")
                )
            )
            return operations

        scoped_request_patch._mmm_tracks_repair_scope = True
        cls._request_patch = scoped_request_patch

    original_evidence = cls._evidence
    if getattr(original_evidence, "_mmm_progressive_evidence", False):
        return

    def progressive_evidence(
        self: Any,
        root: Path,
        *,
        run_gametest: bool,
    ) -> dict[str, Any]:
        relative_files = getattr(self, "_mmm_last_java_paths", ()) or None
        diagnostics = _run_jdt_diagnostics(
            self.diagnostics_factory,
            root,
            relative_files=relative_files,
            timeout_seconds=90,
        )

        errors = _diagnostic_errors(diagnostics)
        if errors:
            return {
                "passed": False,
                "diagnostics": diagnostics,
                "build": {
                    "status": "SKIPPED",
                    "error": "Gradle/GameTest deferred until JDT diagnostics are clean.",
                    "commands": [],
                },
            }

        try:
            build = self.runner_factory(self.gradle_cache).build(
                root,
                run_gametest=run_gametest,
            ).to_dict()
        except (BuildRunnerError, OSError, TimeoutError) as exc:
            build = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "commands": [],
            }
        return {
            "passed": build.get("status") == "PASS" and not errors,
            "diagnostics": diagnostics,
            "build": build,
        }

    progressive_evidence._mmm_progressive_evidence = True
    cls._evidence = progressive_evidence


def install(
    runner_module: Any,
    java_lsp_module: Any,
    repair_module: Any,
) -> None:
    """Install exact-input JDT caching and progressive validation evidence.

    Target-aware Gradle distribution/build caching is owned exclusively by
    ``runner_parallel_validation_contract``. Keeping it out of this contract avoids
    legacy wrapper stacking and stale Gradle API assumptions.
    """

    del runner_module
    _install_jdt_cache(java_lsp_module)
    _install_progressive_repair(repair_module)
