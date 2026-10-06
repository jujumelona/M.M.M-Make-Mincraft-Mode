from __future__ import annotations

"""Canonical contracts for deterministic Typed PlanIR host-generation modules."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

_STATE_STORE_KEYS = frozenset({
    "namespace",
    "schema_version",
    "migrations",
    "malformed_policy",
    "transfer_on_respawn",
})
_NETWORK_SYNC_KEYS = frozenset({
    "sync_interval_ticks",
    "max_payload_bytes",
    "__covers",
})


def _active_records(
    structured_sections: Mapping[str, Any] | None,
    section: str,
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    if not isinstance(structured_sections, Mapping):
        return {}
    from .authored_structured_design import active_concern_records

    return active_concern_records(structured_sections, section)


def _state_variable_rows(
    structured_sections: Mapping[str, Any] | None,
    state_section: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    if isinstance(state_section, Mapping):
        specification = state_section.get("specification")
        if isinstance(specification, Mapping):
            rows = specification.get("variables", ())
            if isinstance(rows, Sequence) and not isinstance(
                rows,
                (str, bytes, bytearray),
            ):
                return tuple(
                    dict(row)
                    for row in rows
                    if isinstance(row, Mapping)
                )
    active = _active_records(structured_sections, "state_model")
    return tuple(
        dict(row)
        for row in active.get("variables", ())
        if isinstance(row, Mapping)
    )


def _migration_rows(
    structured_sections: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    active = _active_records(structured_sections, "persistence")
    return tuple(
        row
        for row in active.get("migration", ())
        if isinstance(row, Mapping)
    )


def _state_schema_version(variables: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(row) for row in variables],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "state-" + hashlib.sha256(payload).hexdigest()[:16]


def _canonical_migrations(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema_version: str,
) -> list[dict[str, Any]]:
    migrations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, row in enumerate(rows):
        source = str(row.get("source_version") or "").strip()
        operation = str(row.get("operation") or "").strip()
        if not source:
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_VERSION_REQUIRED: {index}"
            )
        if operation not in {
            "preserve",
            "rename_key",
            "delete_key",
            "set_default",
        }:
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_OPERATION_INVALID: {operation!r}"
            )
        if source == schema_version:
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_SELF_LOOP: {index} uses current schema "
                f"{schema_version!r} as its source"
            )

        source_key = row.get("source_key")
        destination_key = row.get("destination_key")
        value = row.get("value")
        if operation in {"rename_key", "delete_key"} and not str(
            source_key or ""
        ).strip():
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_SOURCE_KEY_REQUIRED: {index}"
            )
        if operation in {"rename_key", "set_default"} and not str(
            destination_key or ""
        ).strip():
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_DESTINATION_KEY_REQUIRED: {index}"
            )

        identity = (
            source,
            operation,
            str(source_key or ""),
            str(destination_key or ""),
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )
        if identity in seen:
            raise ValueError(
                f"TYPED_PLAN_MIGRATION_DUPLICATE: {index} duplicates an "
                "earlier canonical migration operation"
            )
        seen.add(identity)
        migrations.append({
            "from_version": source,
            "to_version": schema_version,
            "operation": operation,
            "source_key": source_key,
            "destination_key": destination_key,
            "value": value,
        })
    return migrations


def canonical_typed_state_store_config(
    *,
    structured_sections: Mapping[str, Any] | None = None,
    state_section: Mapping[str, Any] | None = None,
    namespace: str = "authored_state",
    transfer_required: bool = False,
) -> dict[str, Any]:
    normalized_namespace = str(namespace or "").strip()
    if not normalized_namespace:
        raise ValueError("TYPED_STATE_STORE_NAMESPACE_REQUIRED")
    variables = _state_variable_rows(structured_sections, state_section)
    schema_version = _state_schema_version(variables)
    return {
        "namespace": normalized_namespace,
        "schema_version": schema_version,
        "migrations": _canonical_migrations(
            _migration_rows(structured_sections),
            schema_version=schema_version,
        ),
        "malformed_policy": "backup_and_reset",
        "transfer_on_respawn": bool(transfer_required),
    }


def normalize_typed_state_store_config(
    raw_config: Mapping[str, Any],
    *,
    structured_sections: Mapping[str, Any] | None = None,
    state_section: Mapping[str, Any] | None = None,
    covers: Sequence[str] = (),
) -> dict[str, Any]:
    if not isinstance(raw_config, Mapping):
        raise ValueError("TYPED_STATE_STORE_CONFIG_INVALID")
    unknown = set(raw_config) - _STATE_STORE_KEYS
    if unknown:
        raise ValueError(
            "TYPED_STATE_STORE_CONFIG_UNKNOWN_FIELDS: "
            + ", ".join(sorted(str(key) for key in unknown))
        )

    normalized_covers = {
        str(ref).strip()
        for ref in covers
        if str(ref).strip()
    }
    if normalized_covers:
        transfer_required = "persistence.transfers" in normalized_covers
    else:
        transfer_required = raw_config.get("transfer_on_respawn") is True

    canonical = canonical_typed_state_store_config(
        structured_sections=structured_sections,
        state_section=state_section,
        namespace=str(raw_config.get("namespace") or "authored_state"),
        transfer_required=transfer_required,
    )
    for key in (
        "schema_version",
        "migrations",
        "malformed_policy",
        "transfer_on_respawn",
    ):
        if key in raw_config and raw_config[key] != canonical[key]:
            raise ValueError(
                f"TYPED_STATE_STORE_CONFIG_DRIFT: {key} does not match "
                "canonical structured-state authority"
            )
    return canonical


def normalize_typed_network_sync_config(
    raw_config: Mapping[str, Any],
    *,
    covers: Sequence[str] = (),
) -> dict[str, Any]:
    if not isinstance(raw_config, Mapping):
        raise ValueError("TYPED_NETWORK_SYNC_CONFIG_INVALID")
    unknown = set(raw_config) - _NETWORK_SYNC_KEYS
    if unknown:
        raise ValueError(
            "TYPED_NETWORK_SYNC_CONFIG_UNKNOWN_FIELDS: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    interval = raw_config.get("sync_interval_ticks", 20)
    max_payload = raw_config.get("max_payload_bytes", 32767)
    if type(interval) is not int or interval <= 0:
        raise ValueError("TYPED_NETWORK_SYNC_INTERVAL_INVALID")
    if type(max_payload) is not int or max_payload <= 0:
        raise ValueError("TYPED_NETWORK_SYNC_PAYLOAD_LIMIT_INVALID")
    normalized_covers = tuple(dict.fromkeys(
        str(ref).strip()
        for ref in (
            covers
            if covers
            else raw_config.get("__covers", ())
        )
        if str(ref).strip()
    ))
    return {
        "sync_interval_ticks": interval,
        "max_payload_bytes": max_payload,
        "__covers": list(normalized_covers),
    }


def normalize_typed_resource_policy_config(
    raw_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw_config, Mapping):
        raise ValueError("TYPED_RESOURCE_POLICY_CONFIG_INVALID")
    if raw_config:
        raise ValueError(
            "TYPED_RESOURCE_POLICY_CONFIG_UNKNOWN_FIELDS: deterministic "
            "resource policy is derived from structured design"
        )
    return {}


__all__ = [
    "canonical_typed_state_store_config",
    "normalize_typed_network_sync_config",
    "normalize_typed_resource_policy_config",
    "normalize_typed_state_store_config",
]
