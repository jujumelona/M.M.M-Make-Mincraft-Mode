from __future__ import annotations

"""Project-local RAG index augmentation for pre-design planning.

External mod catalogue retrieval is owned exclusively by
catalog_first_grounded_rag and the CurseForge provider. Do not independently
search remote sources or add a second retrieval pass in this module.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .rag_index import ProjectRAGIndex


def _workspace_root(router: Any | None = None) -> Path | None:
    attached = str(getattr(router, "_mmm_workspace_root", "") or "").strip()
    if attached:
        return Path(attached).expanduser().resolve()
    configured = os.environ.get("MMM_WORKSPACE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return None


def _index_path(workspace: Path, router: Any | None = None) -> Path:
    attached = str(getattr(router, "_mmm_project_rag_index", "") or "").strip()
    if attached:
        return Path(attached).expanduser().resolve()
    configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
    if configured and not str(
        getattr(router, "_mmm_workspace_root", "") or ""
    ).strip():
        return Path(configured).expanduser().resolve()
    return (workspace / ".minecraft_ai" / "project-index.db").resolve()


def _index_metadata() -> dict[str, Any]:
    return {
        "minecraft_version": os.environ.get(
            "MMM_MINECRAFT_VERSION", "target-neutral"
        ).strip()
        or "target-neutral",
        "loader": os.environ.get("MMM_LOADER", "fabric").strip() or "fabric",
        "mapping_namespace": os.environ.get(
            "MMM_MAPPING_NAMESPACE", "yarn"
        ).strip()
        or "yarn",
        "java_version": os.environ.get("MMM_JAVA_VERSION", "21").strip() or "21",
        "license": "project-local",
        "source_commit": "WORKTREE",
    }


def _ensure_local_index(agentic_module: Any, router: Any) -> dict[str, Any]:
    workspace = _workspace_root(router)
    if workspace is None:
        return {
            "status": "workspace_unconfigured",
            "built": False,
            "index_path": "",
            "reason": "local project root was not explicitly attached or configured",
        }
    if not workspace.exists() or not workspace.is_dir():
        return {
            "status": "workspace_missing",
            "built": False,
            "index_path": "",
            "workspace": str(workspace),
        }

    existing = agentic_module._existing_code_index()
    if existing is not None:
        existing_path = Path(existing).expanduser().resolve()
        legacy_configured = os.environ.get("MMM_PROJECT_RAG_INDEX", "").strip()
        attached_scope = bool(
            str(getattr(router, "_mmm_workspace_root", "") or "").strip()
        )
        try:
            existing_path.relative_to(workspace)
        except ValueError:
            if attached_scope or not legacy_configured:
                existing_path = None
        if existing_path is not None:
            return {
                "status": "available",
                "built": False,
                "index_path": str(existing_path),
                "workspace": str(workspace),
            }

    index_path = _index_path(workspace, router)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = ProjectRAGIndex(index_path)
    max_files_raw = os.environ.get("MMM_RAG_MAX_FILES", "").strip()
    max_files = int(max_files_raw) if max_files_raw else None
    try:
        receipt = index.build(
            [workspace],
            metadata=_index_metadata(),
            router=None,
            semantic=False,
            max_files=max_files,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "build_failed",
            "built": False,
            "index_path": str(index_path),
            "workspace": str(workspace),
            "error": f"{type(exc).__name__}: {exc}",
        }

    os.environ["MMM_PROJECT_RAG_INDEX"] = str(index_path)
    if router is not None:
        router._mmm_project_rag_index = str(index_path)
    return {
        "status": "available",
        "built": True,
        "index_path": str(index_path),
        "workspace": str(workspace),
        "receipt": receipt,
    }


def _augment_bundle(
    agentic_module: Any,
    payload: Mapping[str, Any],
    *,
    local_index: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the authoritative CurseForge evidence and attach local index."""
    result = dict(payload)
    result["local_index"] = dict(local_index)
    if local_index.get("status") == "available":
        result["code_index_status"] = "available"
        result["code_index_path"] = str(local_index.get("index_path", ""))
    result["research_sha256"] = agentic_module._sha256(
        {key: value for key, value in result.items() if key != "research_sha256"}
    )
    return result


def install(agentic_module: Any) -> None:
    """Install one local-index wrapper; never replace CurseForge retrieval."""
    original = agentic_module._forced_rag_bundle
    if getattr(original, "__mmm_grounded_rag__", False):
        return

    def grounded_rag_bundle(
        router: Any, research_brief: Mapping[str, Any]
    ) -> dict[str, Any]:
        local_index = _ensure_local_index(agentic_module, router)
        payload = original(router, research_brief)
        return _augment_bundle(agentic_module, payload, local_index=local_index)

    grounded_rag_bundle.__name__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__qualname__ = "_forced_rag_bundle_grounded"
    grounded_rag_bundle.__mmm_grounded_rag__ = True
    grounded_rag_bundle.__wrapped__ = original
    agentic_module._forced_rag_bundle = grounded_rag_bundle


__all__ = ["install"]
