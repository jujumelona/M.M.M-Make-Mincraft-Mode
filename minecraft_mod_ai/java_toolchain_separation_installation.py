from __future__ import annotations

"""Compatibility entrypoint for native JDT project-toolchain separation.

The project JDK and Gradle daemon JVM are now configured directly by
``java_lsp._jdt_configuration``.  This module remains as a no-op compatibility hook so
older bootstrap code does not need runtime rebinding.
"""

from typing import Any


def install(java_lsp_module: Any) -> None:
    _ = java_lsp_module


__all__ = ["install"]
