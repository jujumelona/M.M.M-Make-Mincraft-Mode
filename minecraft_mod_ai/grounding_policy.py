from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .host_grounding import _SCHEMA_VERSION as _HOST_GROUNDING_SCHEMA

_HOST_BASELINE_CAUSAL_FACTS = frozenset(
    {"project_observed", "code_evidence", "evidence_ready"}
)


def host_baseline_evidence_ready(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether host-validated baseline evidence is ready for the coder.

    A host receipt is baseline evidence only when it proves that at least one project
    observation was actually collected. A hash of an empty observation set is a valid
    integrity receipt, but it is not evidence and must not suppress the coder's retrieval
    phase.

    Target localization and write authority remain separate concerns and are enforced by
    the task capsule/tool loop. Additional retrieval remains available whenever the host
    baseline is incomplete.
    """

    for message in messages:
        payload = _message_payload(message)
        if payload is None:
            continue
        grounding = _find_host_grounding(payload)
        if grounding is not None and _grounding_ready(grounding):
            return True
    return False


def host_baseline_causal_facts(
    messages: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Translate validated host-owned project grounding into causal evidence facts."""

    if not host_baseline_evidence_ready(messages):
        return frozenset()
    return _HOST_BASELINE_CAUSAL_FACTS


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


def _positive_observation_count(receipt: Mapping[str, Any]) -> bool:
    value = receipt.get("observation_count")
    return type(value) is int and value > 0


def _grounding_ready(grounding: Mapping[str, Any]) -> bool:
    policy = grounding.get("policy")
    bindings = grounding.get("evidence_bindings")
    if not isinstance(policy, Mapping) or not isinstance(bindings, Mapping):
        return False
    if not (
        policy.get("resolved_before_first_coder_decode") is True
        and policy.get("baseline_grounding_owned_by_host") is True
        and policy.get("baseline_grounding_optional_for_model") is False
        and policy.get("model_tool_choice_required_for_baseline") is False
    ):
        return False
    project = bindings.get("project_exact_rag")
    if not isinstance(project, Mapping):
        return False
    receipt = project.get("receipt")
    if not isinstance(receipt, Mapping) or not _positive_observation_count(receipt):
        return False
    return bool(
        str(receipt.get("project_sha256", "")).strip()
        and str(receipt.get("observations_sha256", "")).strip()
    )


__all__ = ["host_baseline_causal_facts", "host_baseline_evidence_ready"]
