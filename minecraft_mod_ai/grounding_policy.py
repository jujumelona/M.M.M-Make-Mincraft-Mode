from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .host_grounding import _SCHEMA_VERSION as _HOST_GROUNDING_SCHEMA

_HOST_BASELINE_CAUSAL_FACTS = frozenset(
    {"project_observed", "code_evidence", "evidence_ready"}
)
_HOST_AUTHORITY_ROLES = frozenset({"developer", "system", "tool"})
_TASK_CAPSULE_SCHEMA = "mmm/small-model-task-capsule"


def host_baseline_evidence_ready(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether host-validated baseline evidence is actionable for this turn.

    A project receipt with explicit ``observation_count`` must prove at least one
    observation. Older receipts that predate the counter remain valid when both integrity
    hashes are present. Fresh targets transported separately from their grounding still
    need at least one host-approved research fact; a host-authored task capsule that embeds
    its grounding is already one atomic host decision and does not need that extra receipt.
    """

    decoded = tuple(_decoded_message(message) for message in messages)
    fresh_task_present = _fresh_task_present(decoded)
    for role, payload in decoded:
        if payload is None:
            continue
        grounding = _find_host_grounding(payload)
        if grounding is None:
            continue
        embedded_fresh_authority = bool(
            role in _HOST_AUTHORITY_ROLES and _fresh_task_payload(payload)
        )
        require_research = fresh_task_present and not embedded_fresh_authority
        if _grounding_ready(grounding, require_research=require_research):
            return True
    return False



def _fresh_task_present(decoded: Sequence[tuple[str, Any | None]]) -> bool:
    return any(_fresh_task_payload(payload) for _, payload in decoded)

def host_baseline_causal_facts(
    messages: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Translate validated host-owned project grounding into causal evidence facts."""

    if not host_baseline_evidence_ready(messages):
        return frozenset()
    return _HOST_BASELINE_CAUSAL_FACTS


def _decoded_message(message: Mapping[str, Any]) -> tuple[str, Any | None]:
    return str(message.get("role") or "").strip().casefold(), _message_payload(message)


def _message_payload(message: Mapping[str, Any]) -> Any | None:
    content = message.get("content")
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        return None
    raw = content.strip()
    if not raw.startswith("{"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _find_host_grounding(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if str(value.get("schema_version", "")).strip() == _HOST_GROUNDING_SCHEMA:
            return value
        direct = value.get("host_grounding")
        if isinstance(direct, Mapping):
            found = _find_host_grounding(direct)
            if found is not None:
                return found
        for child in value.values():
            found = _find_host_grounding(child)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _find_host_grounding(child)
            if found is not None:
                return found
    return None


def _fresh_task_payload(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if str(payload.get("schema_version") or "").strip() != _TASK_CAPSULE_SCHEMA:
        return False
    if str(payload.get("reuse_action") or "").strip().casefold() != "fresh":
        return False
    target = payload.get("mutation_target")
    return isinstance(target, Mapping) and bool(str(target.get("path") or "").strip())


def _receipt_hashes_ready(receipt: Any) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    if "observation_count" in receipt:
        count = receipt.get("observation_count")
        if type(count) is not int or count <= 0:
            return False
    return bool(
        str(receipt.get("project_sha256", "")).strip()
        and str(receipt.get("observations_sha256", "")).strip()
    )


def _fresh_implementation_evidence_ready(bindings: Mapping[str, Any]) -> bool:
    """Accept either reviewed research facts or immutable host implementation facts."""

    for key in ("implementation_contract", "approved_research_rag"):
        evidence = bindings.get(key)
        if not isinstance(evidence, Mapping):
            continue
        receipt = evidence.get("receipt")
        if not isinstance(receipt, Mapping):
            continue
        count = receipt.get("selected_fact_count")
        if type(count) is int and count > 0:
            return True
    return False


def _grounding_policy_ready(policy: Any) -> bool:
    return bool(
        isinstance(policy, Mapping)
        and policy.get("resolved_before_first_coder_decode") is True
        and policy.get("baseline_grounding_owned_by_host") is True
        and policy.get("baseline_grounding_optional_for_model") is False
        and policy.get("model_tool_choice_required_for_baseline") is False
    )


def _grounding_ready(
    grounding: Mapping[str, Any], *, require_research: bool = False
) -> bool:
    policy = grounding.get("policy")
    bindings = grounding.get("evidence_bindings")
    if not _grounding_policy_ready(policy) or not isinstance(bindings, Mapping):
        return False
    project = bindings.get("project_exact_rag")
    if not isinstance(project, Mapping) or not _receipt_hashes_ready(project.get("receipt")):
        return False
    return not require_research or _fresh_implementation_evidence_ready(bindings)


__all__ = ["host_baseline_causal_facts", "host_baseline_evidence_ready"]
