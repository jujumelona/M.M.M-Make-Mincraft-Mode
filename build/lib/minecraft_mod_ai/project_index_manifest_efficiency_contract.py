from __future__ import annotations

from typing import Any

_NATIVE_CONTRACTS = (
    ("manifest_receipt", "_mmm_cached_manifest_receipt"),
    ("_ranked_files", "_mmm_cached_relevance_order"),
    ("update_files", "_mmm_incremental_sorted_update"),
    ("write_manifest", "_mmm_incremental_manifest_io"),
)


def install(project_index_module: Any) -> None:
    """Verify ProjectIndex owns its efficiency behavior without runtime patching."""

    cls = project_index_module.ProjectIndex
    missing = [
        f"{method}.{marker}"
        for method, marker in _NATIVE_CONTRACTS
        if not getattr(getattr(cls, method), marker, False)
    ]
    if missing:
        raise RuntimeError(
            "ProjectIndex efficiency behavior must be implemented natively; missing: "
            + ", ".join(missing)
        )


__all__ = ["install"]
