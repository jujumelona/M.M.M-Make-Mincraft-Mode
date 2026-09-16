from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .host_grounding import _SCHEMA_VERSION as _HOST_GROUNDING_SCHEMA
_HOST_BASELINE_CAUSAL_FACTS = frozenset(
    {"project_observed", "code_evidence", "evidence_ready"}
)


def host_baseline_evidence_ready(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether host-validated evidence is actionable before the first coder decode.

    A project snapshot proves repository identity, but it does not by itself prove the
    Minecraft/Fabric API details needed to author a fresh Java implementation. When the
    approved research binding contains no selected facts, keep fresh retrieval mandatory
    so the coder must query the version-pinned code/API evidence surface before mutation.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, Mapping):
            payload: Any = content
        elif isinstance(content, str):
            raw = content.strip()
            if not raw.startswith("{"):
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
        else:
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


def _selected_research_fact_count(bindings: Mapping[str, Any]) -> int:
    research = bindings.get("approved_research_rag")
    if not isinstance(research, Mapping):
        return 0
    receipt = research.get("receipt")
    if not isinstance(receipt, Mapping):
        return 0
    try:
        selected = int(receipt.get("selected_fact_count", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, selected)


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
    if not isinstance(receipt, Mapping):
        return False
    project_ready = bool(
        str(receipt.get("project_sha256", "")).strip()
        and str(receipt.get("observations_sha256", "")).strip()
    )
    if not project_ready:
        return False

    # Repository identity alone is not actionable API evidence for fresh generated
    # Java. CustomModuleGenerator binds require_fresh_evidence=True, so an empty
    # approved research selection must leave the baseline unsatisfied and force one
    # version-pinned RAG/API retrieval before ACT.
    return _selected_research_fact_count(bindings) > 0


__all__ = ["host_baseline_causal_facts", "host_baseline_evidence_ready"]
