"""Keep active Gradle validation bound to the project's resolved Java toolchain."""

from __future__ import annotations

import os
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

_ACTIVE_JAVA_HOME: ContextVar[Path | None] = ContextVar(
    "mmm_runner_target_java_home",
    default=None,
)


def _prepend_java_home(env: dict[str, str], java_home: Path) -> dict[str, str]:
    updated = dict(env)
    updated["JAVA_HOME"] = str(java_home)
    path_key = next((key for key in updated if key.upper() == "PATH"), "PATH")
    current_path = updated.get(path_key, "")
    java_bin = str(java_home / "bin")
    updated[path_key] = java_bin + os.pathsep + current_path if current_path else java_bin
    return updated


def install(runner_module: Any) -> None:
    """Install target-Java preflight after all runner validation overlays are finalized.

    Parallel validation replaces ``GradleRunner._build_locked`` and therefore cannot rely
    on the base runner's ``_prepare_build_context`` Java selection.  This contract keeps
    that final path target-aware without mutating process-global ``os.environ``: the
    resolved JDK travels through a ``ContextVar`` and is copied into each Gradle command
    environment for the duration of one project build.
    """

    cls = runner_module.GradleRunner

    current_run = cls._run
    if not getattr(current_run, "_mmm_target_java_environment", False):
        base_run = current_run

        @wraps(base_run)
        def target_java_run(
            self: Any,
            *,
            name: str,
            executable: Path,
            arguments: tuple[str, ...],
            cwd: Path,
            env: dict[str, str],
            log_path: Path,
        ) -> Any:
            java_home = _ACTIVE_JAVA_HOME.get()
            command_env = (
                _prepend_java_home(env, java_home)
                if java_home is not None
                else env
            )
            return base_run(
                self,
                name=name,
                executable=executable,
                arguments=arguments,
                cwd=cwd,
                env=command_env,
                log_path=log_path,
            )

        target_java_run._mmm_target_java_environment = True
        cls._run = target_java_run

    current_build_locked = cls._build_locked
    if getattr(current_build_locked, "_mmm_target_java_preflight", False):
        return
    base_build_locked = current_build_locked

    @wraps(base_build_locked)
    def target_java_build_locked(
        self: Any,
        project_root: Path,
        *,
        run_gametest: bool,
    ) -> Any:
        root = Path(project_root).expanduser()
        if root.is_symlink() or not root.is_dir() or not (root / "build.gradle").is_file():
            return base_build_locked(self, project_root, run_gametest=run_gametest)

        try:
            adapter = runner_module.adapter_from_project(root.resolve())
        except ValueError:
            return base_build_locked(self, project_root, run_gametest=run_gametest)

        try:
            required_java = int(str(adapter.java_version).strip())
        except (TypeError, ValueError) as exc:
            raise runner_module.BuildRunnerError(
                f"Project target has an invalid Java version: {adapter.java_version!r}"
            ) from exc
        if required_java <= 0:
            raise runner_module.BuildRunnerError(
                f"Project target has an invalid Java version: {adapter.java_version!r}"
            )

        try:
            java_home = runner_module._resolve_project_java_home(required_java)
        except runner_module.JDTWorkspaceBootstrapError as exc:
            return runner_module.BuildReport(
                status="UNAVAILABLE",
                gradle_version=str(adapter.gradle),
                commands=(),
                jar_path=None,
                gametest_report=None,
                error=(
                    f"Java {required_java} toolchain unavailable for target "
                    f"{adapter.minecraft_version}: {exc}"
                ),
            )

        token = _ACTIVE_JAVA_HOME.set(Path(java_home).expanduser().resolve())
        try:
            return base_build_locked(
                self,
                root.resolve(),
                run_gametest=run_gametest,
            )
        finally:
            _ACTIVE_JAVA_HOME.reset(token)

    target_java_build_locked._mmm_target_java_preflight = True
    cls._build_locked = target_java_build_locked


__all__ = ["install"]
