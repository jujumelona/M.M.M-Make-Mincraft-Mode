from __future__ import annotations

"""Separate retrieval evidence from host-selected mutation targeting."""

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any


def observed_context_may_bind(
    observed_context: Any,
    *,
    binding_enabled: bool,
) -> bool:
    return bool(binding_enabled and observed_context is not None)


def context_is_host_pinned(context: Any) -> bool:
    return bool(context is not None and getattr(context, "target_pinned", False))


def context_is_localized(context: Any) -> bool:
    return bool(context is not None and getattr(context, "is_mutation_ready", False))


def materialized_create_context(
    path: str,
    operation: str,
    arguments: Mapping[str, Any],
    create_operations: Collection[str],
    context_factory: Any,
) -> Any:
    if not path or operation not in create_operations:
        return None
    content = arguments.get("content")
    return context_factory(
        target_path=path,
        source_body=content if isinstance(content, str) else None,
        is_new_file=False,
        evidence_source="mutation_receipt",
        writable_paths=(path,),
        target_pinned=True,
    )



def authored_design_execution_requested(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether the current coder turn belongs to saved authored design execution."""

    for message in reversed(messages):
        if str(message.get("role") or "").strip().casefold() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        return str(payload.get("phase") or "").strip() == "implement_authored_design"
    return False


def _workspace_existing_context(
    workspace_root: Any,
    path: str,
    context_factory: Any,
) -> Any:
    clean = str(path or "").strip().replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    if (
        not clean
        or clean.startswith("/")
        or ":" in clean
        or ".." in Path(clean).parts
        or not clean.casefold().endswith((".java", ".kt"))
    ):
        return None
    try:
        root = Path(str(workspace_root)).expanduser().resolve()
        candidate = (root / clean).resolve()
        candidate.relative_to(root)
        raw = candidate.read_bytes()
        source = raw.decode("utf-8")
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return None
    return context_factory(
        target_path=clean,
        source_body=source,
        base_revision_sha=hashlib.sha256(raw).hexdigest(),
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(clean,),
        target_pinned=True,
    )


def recover_stale_existing_context(
    state: Any,
    *,
    workspace_root: Any,
    path: str,
    existing: Any,
    context_factory: Any,
) -> Any:
    """Bind the live admitted file after an exact-edit precondition goes stale."""

    if existing is not None:
        return existing
    context = _workspace_existing_context(workspace_root, path, context_factory)
    if context is None:
        return None
    state.mutation_context = context
    state.created_paths.add(str(context.target_path))
    return context
