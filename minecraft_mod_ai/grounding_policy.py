from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

_HOST_GROUNDING_SCHEMA = "mmm/host-owned-coder-grounding-v1"


def host_baseline_evidence_ready(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether host-validated coder evidence already satisfies baseline grounding.

    Supplemental model retrieval remains available; this only prevents the router
    from making a retrieval tool invocation mandatory when the host has already
    bound exact project evidence before the first coder decode.
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
