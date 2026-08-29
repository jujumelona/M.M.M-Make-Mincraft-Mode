from __future__ import annotations

"""Single-pass semantic compilation for the approved authored requirement graph.

The semantic model gets one schema-constrained compilation turn. The host then validates
source grounding, authored-clause coverage, and semantic identifiers as one atomic result.
Invalid semantic content is never repaired, merged, or partially reused.
"""

import json
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from . import evidence_first_planning as _evidence
from . import semantic_requirement_authority as _authority

_INSTALLED = False
_MARKER = "_mmm_semantic_single_pass_v1"


def _single_pass_messages(
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Reuse the authority instructions while removing repair protocol from the request."""

    messages = [dict(item) for item in _authority._model_messages(clauses)]
    if len(messages) >= 2:
        try:
            payload = json.loads(str(messages[1].get("content") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            payload.pop("repair_diagnostics", None)
            messages[1]["content"] = _authority._canonical(payload)
    return messages


def _call_semantic_model_single_pass(
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> Any:
    """Compile one atomic semantic batch using a schema-bearing request."""

    max_clause_index = max(int(clause["clause_index"]) for clause in clauses)
    parameters = _authority._semantic_schema(max_clause_index)
    messages = _single_pass_messages(clauses)

    native = getattr(router, "generate_tool_decision", None)
    if callable(native):
        return native(
            "planner",
            messages,
            tool_name="compile_semantic_requirements",
            parameters=parameters,
            description=(
                "Compile every independently observable authored behavior into semantic "
                "leaf requirements. The host owns exact source grounding."
            ),
        )

    raw = router.generate_text(
        "planner",
        messages,
        response_format="json",
        response_schema=parameters,
        enable_tools=False,
    )
    return _authority._parse_json(raw)


def _generate_approved_nodes_single_pass(
    prompt: str,
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compile and approve all authored requirements with exactly one model call."""

    del prompt
    try:
        payload = _call_semantic_model_single_pass(router, clauses)
    except Exception as exc:
        raise _evidence.EvidencePlanError(
            "semantic compilation failed before host approval: "
            + _authority._canonical(
                {
                    "error_code": "REQ_MODEL_RESPONSE",
                    "error": f"{type(exc).__name__}: {exc}",
                    "generation_policy": "single_pass_constrained",
                }
            )
        ) from exc

    nodes, invalid_clauses, diagnostics = _authority._evaluate_batch(payload, clauses)
    if invalid_clauses:
        raise _evidence.EvidencePlanError(
            "semantic compilation did not satisfy the host contract: "
            + _authority._canonical(
                {
                    "invalid_clause_indices": sorted(invalid_clauses),
                    "diagnostics": diagnostics,
                    "generation_policy": "single_pass_constrained",
                }
            )
        )
    return _authority._assign_local_ids(nodes)


def install() -> None:
    """Make semantic compilation single-pass and remove repair from the normal path."""

    global _INSTALLED
    if _INSTALLED:
        return

    current_generate = _authority._generate_approved_nodes
    if not getattr(current_generate, _MARKER, False):
        generate = _generate_approved_nodes_single_pass
        setattr(generate, _MARKER, True)
        generate.__wrapped__ = current_generate  # type: ignore[attr-defined]
        _authority._generate_approved_nodes = generate

    current_build = _authority._build_catalog
    if not getattr(current_build, _MARKER, False):

        @wraps(current_build)
        def build_catalog(
            prompt: str,
            nodes: Sequence[Mapping[str, Any]],
            clauses: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
            payload = current_build(prompt, nodes, clauses)
            audit = dict(payload.get("semantic_audit") or {})
            audit.update(
                {
                    "normal_model_turns": 1,
                    "max_repair_turns": 0,
                    "generation_policy": "single_pass_constrained",
                }
            )
            payload["semantic_audit"] = audit
            payload["catalog_sha256"] = ""
            payload["catalog_sha256"] = _evidence._hash_without(
                payload,
                "catalog_sha256",
            )
            return payload

        setattr(build_catalog, _MARKER, True)
        _authority._build_catalog = build_catalog

    _INSTALLED = True


__all__ = [
    "_call_semantic_model_single_pass",
    "_generate_approved_nodes_single_pass",
    "_single_pass_messages",
    "install",
]
