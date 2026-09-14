from __future__ import annotations

"""Compatibility hook for the retired JDT-to-Gradle diagnostic fallback.

Java semantic verification must fail with its own infrastructure diagnostic when the
JDT verifier is unavailable. Gradle remains a build/project-model owner and is never
invoked as a semantic-verifier fallback from this boundary.
"""

from typing import Any


def install(service_type: type[Any]) -> None:
    """Intentionally leave ``ProductionToolService.java_diagnostics`` unchanged."""


__all__ = ["install"]
