"""Resolve project-target Java environments without mutating runtime-owned methods."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def target_java_environment(
    *,
    runner_module: Any,
    adapter: Any,
    environment: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    "Return a command environment bound to the project's declared Java toolchain.\n\n    Adapters that do not declare Java metadata retain the incoming environment for\n    compatibility with validation surfaces that predate target-Java selection. An\n    explicitly declared Java version remains a strict contract and is resolved to the\n    target JDK before any Gradle command is launched.\n    "
    return _architecture_impl_target_java_environment((runner_module, adapter, environment))


def target_java_build_environment(
    *, runner_module: Any, adapter: Any, version: str
) -> tuple[dict[str, str], Any | None]:
    """Prepare the target JDK environment or an UNAVAILABLE build report."""
    environment, error = target_java_environment(
        runner_module=runner_module,
        adapter=adapter,
        environment=os.environ.copy(),
    )
    if error is None:
        return environment, None
    return environment, runner_module.BuildReport(
        status="UNAVAILABLE",
        gradle_version=version,
        commands=(),
        jar_path=None,
        gametest_report=None,
        error=error,
    )


__all__ = ["target_java_build_environment", "target_java_environment"]

def _architecture_impl_target_java_environment(_ctx):
    (runner_module, adapter, environment) = _ctx
    """Return a command environment bound to the project's declared Java toolchain.

    Adapters that do not declare Java metadata retain the incoming environment for
    compatibility with validation surfaces that predate target-Java selection. An
    explicitly declared Java version remains a strict contract and is resolved to the
    target JDK before any Gradle command is launched.
    """
    raw_java_version = getattr(adapter, "java_version", None)
    if raw_java_version is None or not str(raw_java_version).strip():
        return dict(environment), None

    try:
        required_java = int(str(raw_java_version).strip())
    except (TypeError, ValueError) as exc:
        raise runner_module.BuildRunnerError(
            f"Project target has an invalid Java version: {raw_java_version!r}"
        ) from exc
    if required_java <= 0:
        raise runner_module.BuildRunnerError(
            f"Project target has an invalid Java version: {raw_java_version!r}"
        )

    try:
        java_home = Path(
            runner_module._resolve_project_java_home(required_java)
        ).expanduser().resolve()
    except runner_module.JDTWorkspaceBootstrapError as exc:
        target_version = getattr(adapter, "minecraft_version", "unknown")
        return dict(environment), (
            f"Java {required_java} toolchain unavailable for target "
            f"{target_version}: {exc}"
        )

    updated = dict(environment)
    updated["JAVA_HOME"] = str(java_home)
    path_key = next((key for key in updated if key.upper() == "PATH"), "PATH")
    current_path = updated.get(path_key, "")
    java_bin = str(java_home / "bin")
    updated[path_key] = java_bin + os.pathsep + current_path if current_path else java_bin
    return updated, None

