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
    """Return a command environment bound to the project's declared Java toolchain.

    The helper is intentionally pure with respect to ``runner_module`` and process-global
    ``os.environ``.  Runtime-owned Gradle methods remain installed by the existing parallel
    validation contract; this function only resolves the target JDK and prepares the
    per-command environment consumed by that contract.
    """
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
        java_home = Path(
            runner_module._resolve_project_java_home(required_java)
        ).expanduser().resolve()
    except runner_module.JDTWorkspaceBootstrapError as exc:
        return dict(environment), (
            f"Java {required_java} toolchain unavailable for target "
            f"{adapter.minecraft_version}: {exc}"
        )

    updated = dict(environment)
    updated["JAVA_HOME"] = str(java_home)
    path_key = next((key for key in updated if key.upper() == "PATH"), "PATH")
    current_path = updated.get(path_key, "")
    java_bin = str(java_home / "bin")
    updated[path_key] = java_bin + os.pathsep + current_path if current_path else java_bin
    return updated, None


__all__ = ["target_java_environment"]
