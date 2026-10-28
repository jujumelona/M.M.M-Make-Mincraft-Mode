"""Prepare source-owned UI implementation work without inventing state effects."""
from __future__ import annotations

from copy import deepcopy

from .source_ui_actions import (
    SOURCE_UI_CAPABILITY, deferred_ui_interactions, source_ui_capabilities,
    source_ui_preflight_sections,
)
from .request_record_ownership import same_action_source


def validated_deferred_ui_interactions(sections, request_scope):
    identifiers = {row["requirement_id"] for row in request_scope.get("requirements", ())}
    deferred = deferred_ui_interactions(sections)
    for row in deferred:
        if row["source_requirement_id"] not in identifiers:
            raise ValueError("SOURCE_UI_SOURCE_UNKNOWN: " + row["source_requirement_id"])
    return deferred


def production_ui_capabilities(capabilities, sections, request_scope):
    deferred = validated_deferred_ui_interactions(sections, request_scope)
    return source_ui_capabilities(capabilities, sections) if deferred else deepcopy(dict(capabilities))


def action_preflight_sections(sections, capabilities):
    return production_checking_sections(sections, enabled=SOURCE_UI_CAPABILITY in (capabilities or {}))


def production_checking_sections(sections, *, enabled=True):
    result = source_ui_preflight_sections(sections, enabled=enabled)
    deferred = deferred_ui_interactions(sections) if enabled else []
    state = result.get("state_model", {}).get("specification", {})
    for concern in ("updates", "transitions"):
        if concern in state:
            state[concern] = [row for row in state[concern]
                              if not any(same_action_source(row, action) for action in deferred)]
    return result


def has_action_execution(algorithm, state, sections):
    return bool(algorithm.get("atomic_mutations") or state.get("transitions")
                or state.get("updates") or deferred_ui_interactions(sections))


def needs_grounded_action_body(prefer_grounded, body):
    return prefer_grounded and body is None


def admit_action_body(function, record, sections, metadata, state_types, capabilities, validator):
    from .source_ui_actions import source_ui_body

    delegated = source_ui_body(record, sections, capabilities)
    if delegated is None:
        validator(function, metadata, state_types, capabilities)
    elif function["body"] != delegated:
        raise ValueError("SOURCE_UI_DELEGATE_BODY_CHANGED")
