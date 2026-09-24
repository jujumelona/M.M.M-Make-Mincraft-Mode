from __future__ import annotations


class CustomModuleGenerationError(RuntimeError):
    """Raised when one custom-module generation transaction cannot proceed safely."""


__all__ = ["CustomModuleGenerationError"]
