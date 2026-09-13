from __future__ import annotations

"""Install late exact-project-JDK provisioning at the Java verification boundary.

Colab runtime setup can run before a target Minecraft version has selected its
required Java major.  The verification boundary therefore has to treat
``MMM_JAVA_VERSION`` as late-bound: first reuse any already-visible exact JDK,
then provision that exact major only when JDT actually needs it.
"""

import os
from pathlib import Path
from types import ModuleType
from typing import Callable

from . import jdtls_bootstrap


_MARKER = "__mmm_late_project_jdk_provisioning__"


def _provision_exact_project_jdk(required_major: int) -> Path:
    if required_major <= 0:
        raise jdtls_bootstrap.JDTLSBootstrapError(
            f"Project JDK major must be positive; got {required_major}."
        )

    # _install_project_jdk is lock-protected and re-checks existing exact-major
    # runtimes before downloading, so concurrent verifier processes converge on
    # one cached installation rather than racing duplicate downloads.
    home = jdtls_bootstrap._install_project_jdk(required_major)
    java = jdtls_bootstrap._jdk_java(home)
    observed_major = jdtls_bootstrap._java_major(str(java))
    if observed_major != required_major:
        raise jdtls_bootstrap.JDTLSBootstrapError(
            "Project JDK validation failed after late provisioning: "
            f"requested Java {required_major}, found Java {observed_major} at {home}."
        )

    resolved = home.resolve()
    os.environ["MMM_PROJECT_JAVA_HOME"] = str(resolved)
    return resolved


def install(java_lsp_module: ModuleType) -> None:
    """Make project-JDK resolution provision the late-bound exact Java major."""

    current: Callable[[int | None], Path] = getattr(
        java_lsp_module, "_resolve_project_java_home"
    )
    if getattr(current, _MARKER, False):
        return

    bootstrap_error = getattr(java_lsp_module, "JDTWorkspaceBootstrapError")
    requested_major = getattr(java_lsp_module, "_requested_project_java_major")

    def resolve_project_java_home(required_major: int | None = None) -> Path:
        required = required_major if required_major is not None else int(requested_major())
        try:
            return current(required)
        except bootstrap_error as discovery_error:
            try:
                _provision_exact_project_jdk(required)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as provisioning_error:
                raise bootstrap_error(
                    "JDT workspace bootstrap failure: exact project JDK provisioning failed "
                    f"after local discovery missed MMM_JAVA_VERSION={required}; "
                    f"provisioning_error={type(provisioning_error).__name__}: "
                    f"{provisioning_error}"
                ) from provisioning_error

            # Re-enter the canonical resolver instead of trusting installation
            # success.  This preserves the existing exact-major validation and
            # gives JDT the same canonical path it would have discovered locally.
            try:
                return current(required)
            except bootstrap_error as validation_error:
                raise bootstrap_error(
                    "JDT workspace bootstrap failure: exact project JDK provisioning "
                    f"completed for Java {required}, but canonical resolution still failed; "
                    f"initial_discovery={discovery_error}; post_provision={validation_error}"
                ) from validation_error

    setattr(resolve_project_java_home, _MARKER, True)
    setattr(resolve_project_java_home, "__wrapped__", current)
    java_lsp_module._resolve_project_java_home = resolve_project_java_home


__all__ = ["install"]
