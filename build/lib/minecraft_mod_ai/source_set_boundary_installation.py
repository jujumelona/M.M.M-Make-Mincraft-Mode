from __future__ import annotations

from typing import Any


def install(java_lsp_module: Any) -> None:
    """Verify the canonical Java diagnostics owner already enforces source-set safety.

    Source-set safety is part of ``JavaLanguageService.diagnostics`` itself. Runtime
    finalization must never replace that method with another wrapper because doing so
    creates a second mutable validation owner and expands the runtime mutation surface.
    """

    diagnostics = java_lsp_module.JavaLanguageService.diagnostics
    if not getattr(diagnostics, "__mmm_source_set_boundary__", False):
        raise RuntimeError(
            "JavaLanguageService.diagnostics is missing the canonical source-set boundary"
        )


__all__ = ["install"]
