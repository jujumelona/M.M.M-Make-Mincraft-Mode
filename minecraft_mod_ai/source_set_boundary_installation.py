from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Any

from .source_set_boundary_contract import (
    SourceSetBoundaryError,
    assert_server_safe_source_sets,
)


def install(java_lsp_module: Any) -> None:
    """Install the source-set guard as the outer Java validation boundary.

    This intentionally sits outside the exact-input JDT cache: a partial repair
    validation may cache only edited files, while a client-boundary violation can
    be introduced elsewhere in the project. The guard therefore re-evaluates the
    project source-set graph before every certifiable diagnostics request.
    """

    cls = java_lsp_module.JavaLanguageService
    original = cls.diagnostics
    if getattr(original, "__mmm_source_set_boundary__", False):
        return

    @wraps(original)
    def guarded_diagnostics(
        self: Any,
        project_root: str | Path,
        *,
        relative_files: Iterable[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        try:
            assert_server_safe_source_sets(project_root)
        except (SourceSetBoundaryError, FileNotFoundError, OSError, UnicodeError) as exc:
            raise java_lsp_module.JDTLanguageServerError(
                f"Java source-set preflight failed: {exc}"
            ) from exc
        return original(
            self,
            project_root,
            relative_files=relative_files,
            timeout_seconds=timeout_seconds,
        )

    guarded_diagnostics.__mmm_source_set_boundary__ = True
    cls.diagnostics = guarded_diagnostics


__all__ = ["install"]
