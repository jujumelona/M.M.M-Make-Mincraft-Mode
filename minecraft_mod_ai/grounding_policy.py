from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .host_grounding import _SCHEMA_VERSION as _HOST_GROUNDING_SCHEMA

_HOST_BASELINE_CAUSAL_FACTS = frozenset(
    {"project_observed", "code_evidence", "evidence_ready"}
)
_TASK_CAPSULE_SCHEMA = "mmm/small-model-task-capsule"


def host_baseline_evidence_ready(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether host-validated evidence is actionable before the first coder decode.

    Existing host-owned project grounding remains sufficient for ordinary coder turns.
    Fresh Java creation is narrower: when the active task capsule says ``reuse_action`` is
    ``fresh``, repository identity alone does not establish the Minecraft/Fabric API shape
    needed for a new source file. In that case an empty approved-research selection keeps
    the existing fresh-retrieval path active before mutation.
    """

    payloads = tuple(_message_payload(message) for message in messages)
    require_fresh_java_evidence = any(
        payload is not None and _contains_fresh_java_task(payload)
        for payload in payloads
    )
    for payload in payloads:
        if payload is None:
            continue
        grounding = _find_host_grounding(payload)
        if grounding is None or not _grounding_ready(grounding):
            continue
        if require_fresh_java_evidence:
            bindings = grounding.get("evidence_bindings")
            if not isinstance(bindings, Mapping):
                continue
            if _selected_research_fact_count(bindings) <= 0:
                continue
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


def _contains_fresh_java_task(value: Any) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("schema_version", "")).strip() == _TASK_CAPSULE_SCHEMA:
            reuse_action = str(value.get("reuse_action") or "").strip().casefold()
            target = value.get("mutation_target")
            path = (
                str(target.get("path") or "").replace("\\", "/").strip()
                if isinstance(target, Mapping)
                else ""
            )
            if reuse_action == "fresh" and path.endswith(".java"):
                return True
        return any(_contains_fresh_java_task(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_fresh_java_task(child) for child in value)
    return False


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
    return bool(
        str(receipt.get("project_sha256", "")).strip()
        and str(receipt.get("observations_sha256", "")).strip()
    )


__all__ = ["host_baseline_causal_facts", "host_baseline_evidence_ready"]
