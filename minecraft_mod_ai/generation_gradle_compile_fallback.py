from __future__ import annotations

"""Compile-only Gradle fallback for generation-time Java verification.

The fallback deliberately runs only ``compileJava``. A non-zero Gradle exit is
coder-repairable only when the log contains positive javac source diagnostics. Dependency
resolution, wrapper, network, timeout, and other Gradle infrastructure failures remain
verifier-unavailable evidence and must never be fed back as a source-repair request.
"""

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_from_project
from .runner import BuildRunnerError, CommandResult, GradleRunner, _exclusive_cache_lock

_SOURCE_ERROR = re.compile(r"(?mi)^.*\.java:\d+(?::\d+)?:\s*error:\s+.+$")


@dataclass(frozen=True)
class CompileVerificationReport:
    status: str
    gradle_version: str
    commands: tuple[CommandResult, ...]
    error: str | None = None
    source_diagnostics: bool = False

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def available(self) -> bool:
        return self.status in {"PASS", "FAIL"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "gradle_version": self.gradle_version,
            "commands": [asdict(command) for command in self.commands],
            "error": self.error,
            "source_diagnostics": self.source_diagnostics,
            "verification_scope": "compileJava",
        }


def has_javac_source_diagnostics(log_text: str) -> bool:
    """Require concrete file/line javac errors before authorizing coder repair."""

    return bool(_SOURCE_ERROR.search(str(log_text or "")))


def _read_log(path: str | Path, *, max_bytes: int = 2 * 1024 * 1024) -> str:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        return ""
    try:
        with target.open("rb") as handle:
            size = target.stat().st_size
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            raw = handle.read(max_bytes)
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


class GradleCompileVerifier:
    """Run the pinned project toolchain only far enough to compile Java sources."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        download_timeout_seconds: int = 300,
        command_timeout_seconds: int = 1200,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.runner = GradleRunner(
            self.cache_dir,
            download_timeout_seconds=download_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
        )
        self.command_timeout_seconds = command_timeout_seconds

    def verify(self, project_root: Path) -> CompileVerificationReport:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with _exclusive_cache_lock(
            self.cache_dir,
            timeout_seconds=max(300, self.command_timeout_seconds * 3),
        ):
            return self._verify_locked(Path(project_root).expanduser().resolve())

    def _verify_locked(self, project_root: Path) -> CompileVerificationReport:
        if not (project_root / "build.gradle").is_file():
            raise BuildRunnerError(f"Not a generated Gradle project: {project_root}")
        try:
            adapter = adapter_from_project(project_root)
        except ValueError as exc:
            raise BuildRunnerError(
                f"Project platform lock is missing, mixed, or unsupported: {exc}"
            ) from exc

        gradle_version = adapter.gradle
        gradle_sha256 = adapter.gradle_sha256
        logs = project_root / ".minecraft_ai" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        gradle = self.runner._ensure_gradle(gradle_version, gradle_sha256)
        environment = os.environ.copy()
        environment["GRADLE_USER_HOME"] = str(self.cache_dir / "gradle-user-home")
        environment["CI"] = "true"
        commands: list[CommandResult] = []

        if not self.runner._wrapper_is_current(project_root, gradle_version, gradle_sha256):
            wrapper = self.runner._run(
                name="verifier_wrapper",
                executable=gradle,
                arguments=(
                    "--no-daemon",
                    "wrapper",
                    "--gradle-version",
                    gradle_version,
                    "--gradle-distribution-sha256-sum",
                    gradle_sha256,
                    "--stacktrace",
                ),
                cwd=project_root,
                env=environment,
                log_path=logs / "gradle-verifier-wrapper.log",
            )
            commands.append(wrapper)
            if wrapper.exit_code != 0:
                return CompileVerificationReport(
                    status="UNAVAILABLE",
                    gradle_version=gradle_version,
                    commands=tuple(commands),
                    error="Gradle wrapper preparation failed before source compilation.",
                )

        compile_result = self.runner._run(
            name="compile_java_verifier",
            executable=gradle,
            arguments=("--no-daemon", "compileJava", "--stacktrace"),
            cwd=project_root,
            env=environment,
            log_path=logs / "gradle-compile-java-verifier.log",
        )
        commands.append(compile_result)
        if compile_result.exit_code == 0:
            return CompileVerificationReport(
                status="PASS",
                gradle_version=gradle_version,
                commands=tuple(commands),
            )

        if compile_result.timed_out:
            return CompileVerificationReport(
                status="UNAVAILABLE",
                gradle_version=gradle_version,
                commands=tuple(commands),
                error="Gradle compileJava timed out before trustworthy source diagnostics completed.",
            )

        log_text = _read_log(compile_result.log_path)
        source_diagnostics = has_javac_source_diagnostics(log_text)
        if source_diagnostics:
            return CompileVerificationReport(
                status="FAIL",
                gradle_version=gradle_version,
                commands=tuple(commands),
                error="Gradle compileJava reported javac source errors.",
                source_diagnostics=True,
            )
        return CompileVerificationReport(
            status="UNAVAILABLE",
            gradle_version=gradle_version,
            commands=tuple(commands),
            error=(
                "Gradle compileJava failed without concrete javac file/line diagnostics; "
                "treating the fallback as infrastructure-unavailable rather than source-invalid."
            ),
        )


__all__ = [
    "CompileVerificationReport",
    "GradleCompileVerifier",
    "has_javac_source_diagnostics",
]
