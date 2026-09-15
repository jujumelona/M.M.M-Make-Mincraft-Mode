from __future__ import annotations

"""Keep JDT's launcher JVM, project JDK, and Gradle daemon JVM distinct.

Eclipse JDT LS exposes ``java.configuration.runtimes`` for project JDKs and
``java.import.gradle.java.home`` for the JVM that runs the Gradle daemon.  A
language-server launcher JVM is a third concern and is configured separately by
``MMM_JDTLS_JAVA_HOME`` in :mod:`minecraft_mod_ai.java_lsp`.

This installer pins Gradle import to the already-resolved project JDK instead of
letting Buildship inherit the JDT launcher/host ``JAVA_HOME``.  That prevents a
host JDK (for example Java 21) from compiling a project whose selected release is
newer (for example Java 25), while preserving the project's declared release
rather than silently downgrading it.
"""

from pathlib import Path
from typing import Any


def _project_home_from_configuration(configuration: dict[str, Any]) -> str | None:
    java = configuration.get("java")
    if not isinstance(java, dict):
        return None
    runtime_configuration = java.get("configuration")
    if not isinstance(runtime_configuration, dict):
        return None
    runtimes = runtime_configuration.get("runtimes")
    if not isinstance(runtimes, list):
        return None
    defaults = [
        item
        for item in runtimes
        if isinstance(item, dict) and item.get("default") is True and item.get("path")
    ]
    if len(defaults) != 1:
        return None
    return str(defaults[0]["path"])


def install(java_lsp_module: Any) -> None:
    """Install the project-JDK/Gradle-JVM separation exactly once."""
    current = java_lsp_module._jdt_configuration
    if getattr(current, "_mmm_java_toolchain_separation", False):
        return

    def separated_configuration(project_java_home: Path | None = None) -> dict[str, Any]:
        configuration = current(project_java_home)
        if not isinstance(configuration, dict):
            raise TypeError("JDT configuration must be a mapping.")

        resolved_home = (
            str(Path(project_java_home).expanduser().resolve())
            if project_java_home is not None
            else _project_home_from_configuration(configuration)
        )
        if not resolved_home:
            raise java_lsp_module.JDTWorkspaceBootstrapError(
                "JDT workspace bootstrap failure: project JDK was resolved without a "
                "usable home, so the Gradle daemon JVM cannot be selected safely."
            )

        java = configuration.setdefault("java", {})
        if not isinstance(java, dict):
            raise TypeError("JDT java configuration must be a mapping.")
        java_import = java.setdefault("import", {})
        if not isinstance(java_import, dict):
            raise TypeError("JDT java.import configuration must be a mapping.")
        gradle = java_import.setdefault("gradle", {})
        if not isinstance(gradle, dict):
            raise TypeError("JDT java.import.gradle configuration must be a mapping.")
        gradle_java = gradle.setdefault("java", {})
        if not isinstance(gradle_java, dict):
            raise TypeError("JDT java.import.gradle.java configuration must be a mapping.")

        # Official JDT/vscode-java setting: java.import.gradle.java.home.
        # This controls the Gradle daemon JVM and must follow the project JDK,
        # not the JVM used merely to launch JDT LS.
        gradle_java["home"] = resolved_home
        return configuration

    separated_configuration._mmm_java_toolchain_separation = True
    separated_configuration._mmm_original = current
    java_lsp_module._jdt_configuration = separated_configuration


__all__ = ["install"]
