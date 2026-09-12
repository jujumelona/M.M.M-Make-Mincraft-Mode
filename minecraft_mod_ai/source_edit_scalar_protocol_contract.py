from __future__ import annotations

"""Canonical model-facing source edit protocol compatibility facade.

The stable implementation lives in ``_source_edit_scalar_protocol_contract_core``.
This facade normalizes compatibility-expanded model payloads before delegating to the
strict core contract.  Operation semantics remain fail-closed: aliases may collapse
only when they describe one unambiguous value, and unknown fields are still rejected
by the core validator.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import _source_edit_scalar_protocol_contract_core as _core

SOURCE_EDIT_PARAMETER_ALIASES = _core.SOURCE_EDIT_PARAMETER_ALIASES
SOURCE_EDIT_SCHEMA = _core.SOURCE_EDIT_SCHEMA


def _is_placeholder(value: Any) -> bool:
    return value is None or value == ""


def _consume_alias_group(
    runtime_module: Any,
    payload: dict[str, Any],
    *,
    canonical: str,
    aliases: tuple[str, ...],
) -> None:
    """Collapse one compatibility alias group without hiding real conflicts."""

    keys = (canonical, *aliases)
    present = [(key, payload[key]) for key in keys if key in payload]
    if not present:
        return

    meaningful = [(key, value) for key, value in present if not _is_placeholder(value)]
    if meaningful:
        chosen_key, chosen_value = meaningful[0]
        conflicts = [
            (key, value)
            for key, value in meaningful[1:]
            if value != chosen_value
        ]
        if conflicts:
            conflict_keys = [chosen_key, *(key for key, _ in conflicts)]
            raise runtime_module.AgentToolRuntimeError(
                "Conflicting model-facing source-write aliases for "
                f"{canonical!r}: {conflict_keys}"
            )
    else:
        # Empty text is valid for create_file/replacement content.  If every supplied
        # spelling is merely empty/None, preserve an actual string when available.
        chosen_value = next(
            (value for _, value in present if isinstance(value, str)),
            present[0][1],
        )

    for key, _ in present:
        payload.pop(key, None)
    payload[canonical] = chosen_value


def _consume_standard_aliases(runtime_module: Any, payload: dict[str, Any]) -> None:
    grouped: dict[str, list[str]] = {}
    for alias, canonical in SOURCE_EDIT_PARAMETER_ALIASES.items():
        grouped.setdefault(canonical, []).append(alias)
    for canonical, aliases in grouped.items():
        _consume_alias_group(
            runtime_module,
            payload,
            canonical=canonical,
            aliases=tuple(aliases),
        )


def _canonicalize_create_file_payload(
    runtime_module: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    # Path aliases are normal aliases regardless of operation.
    _consume_alias_group(
        runtime_module,
        payload,
        canonical="path",
        aliases=("file", "target_path", "target_file"),
    )

    # Compatibility adapters historically expanded one create payload into several
    # equivalent spellings.  For create_file they all mean exactly one thing: content.
    _consume_alias_group(
        runtime_module,
        payload,
        canonical="content",
        aliases=(
            "text",
            "new",
            "new_text",
            "new_content",
            "replacement",
            "code",
            "body",
        ),
    )

    # These are schema-known helper slots used by other operations.  Once operation is
    # explicitly create_file they carry no executable meaning, so compatibility-filled
    # values must not turn an otherwise unambiguous create into an invalid-field error.
    for helper in (
        "old",
        "old_text",
        "anchor",
        "count",
        "package_name",
        "declaration",
        "import_name",
        "member",
    ):
        payload.pop(helper, None)

    payload["operation"] = "create_file"
    return payload


def _canonicalize_compat_payload(
    runtime_module: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload)
    operation = _core._normalize_operation(
        runtime_module,
        normalized.get("operation"),
        normalized,
    )

    if operation == "create_file":
        return _canonicalize_create_file_payload(runtime_module, normalized)

    _consume_standard_aliases(runtime_module, normalized)
    normalized["operation"] = operation
    return normalized


def _reject_model_java_create_file(runtime_module: Any, payload: Mapping[str, Any]) -> None:
    if payload.get("operation") != "create_file":
        return
    path = payload.get("path")
    if not isinstance(path, str):
        return
    normalized_path = path.strip().replace("\\", "/")
    if normalized_path.casefold().endswith(".java"):
        raise runtime_module.AgentToolRuntimeError(
            "Model create_file cannot create or replace Java source; use "
            "create_java_type, add_java_import, and insert_java_member instead: "
            f"{normalized_path}"
        )


def materialize_model_source_edit(
    runtime_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
    *,
    bound_project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compile one semantic model edit into one strict core host patch."""

    if not isinstance(payload, Mapping):
        # Preserve the core contract/error wording for non-object arguments.
        return _core.materialize_model_source_edit(
            runtime_module,
            workspace_root,
            payload,
            bound_project_root=bound_project_root,
        )

    normalized = _canonicalize_compat_payload(runtime_module, payload)
    _reject_model_java_create_file(runtime_module, normalized)
    return _core.materialize_model_source_edit(
        runtime_module,
        workspace_root,
        normalized,
        bound_project_root=bound_project_root,
    )


__all__ = [
    "SOURCE_EDIT_PARAMETER_ALIASES",
    "SOURCE_EDIT_SCHEMA",
    "materialize_model_source_edit",
]
