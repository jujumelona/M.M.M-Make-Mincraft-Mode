from __future__ import annotations

"""Canonical model-facing source edit protocol compatibility facade."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import _source_edit_scalar_protocol_contract_core as _core
from .mutation_authority import current_mutation_error

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
    _consume_alias_group(
        runtime_module,
        payload,
        canonical="path",
        aliases=("file", "target_path", "target_file"),
    )
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


def _enforce_host_mutation_authority(runtime_module: Any, payload: Mapping[str, Any]) -> None:
    error = current_mutation_error(
        payload.get("path"),
        operation=payload.get("operation"),
    )
    if error is not None:
        raise runtime_module.AgentToolRuntimeError(error)


def materialize_model_source_edit(
    runtime_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
    *,
    bound_project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compile one semantic model edit into one strict host-authorized patch."""

    if not isinstance(payload, Mapping):
        return _core.materialize_model_source_edit(
            runtime_module,
            workspace_root,
            payload,
            bound_project_root=bound_project_root,
        )

    normalized = _canonicalize_compat_payload(runtime_module, payload)
    _enforce_host_mutation_authority(runtime_module, normalized)
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
