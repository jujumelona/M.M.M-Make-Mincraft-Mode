"""Shared template runtime errors."""


class TemplateBlocked(ValueError):
    """Raised when a template cannot be completed without inventing information."""


__all__ = ["TemplateBlocked"]
