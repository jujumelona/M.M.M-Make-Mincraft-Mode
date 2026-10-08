"""Host-owned source-reuse handoff for the default authored production lane.

Discovery candidates are never executable code. Only proven donor decisions
may be materialized into the final workspace. The small model gets a bounded
receipt, never authority to invent or select a donor.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def authored_capability_graph(
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
) -> dict[str, Any]:
    from .minecraft_knowledge_nodes import detect_features

    # Known taxonomy only supplies search keywords, never the request's scope.
    # Include the actual authored request to preserve uncatalogued mechanics.
    features = [
        str(name).replace("_", " ")
        for name in detect_features(requested_prompt)
        if str(name) != "base_mod"
    ]
    full_request = " ".join(requested_prompt.split())
    nodes = list(dict.fromkeys(
        ["custom gameplay " + full_request[:150]]
        + ["minecraft fabric " + name for name in features]
    ))
    return {
        "nodes": nodes,
        "search_terms": [
            {"capability": node, "terms": [full_request[:200], node]}
            for node in nodes
        ],
        "coverage_policy": "discovery_hints_only; authored design remains authoritative",
    }


def resolve_authored_source_reuse(
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
    *,
    minecraft_version: str,
    loader: str,
) -> dict[str, Any]:
    from .grounded_source_reuse import build_repository_reuse_plan

    graph = authored_capability_graph(requested_prompt, structured_sections)
    frozen = {
        "plan_sha256": "sha256:" + _sha(graph),
        "capability_graph": graph,
    }
    design = {
        "_pre_retrieval_plan": frozen,
        "_platform_selection": {
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
            },
        },
    }
    receipt = build_repository_reuse_plan(design)
    return {
        **receipt,
        "bound_target": {
            "minecraft_version": minecraft_version,
            "loader": loader,
        },
        "origin": "default_authored_production",
    }


def verified_reuse_context(plan: Mapping[str, Any]) -> str:
    """Compact, authority-preserving context for a low-memory local model."""
    decisions = plan.get("capabilities", ())
    summaries = []
    for row in decisions if isinstance(decisions, list) else ():
        if not isinstance(row, Mapping):
            continue
        source_id = str(row.get("source_id") or "")
        summaries.append({
            "capability": str(row.get("capability") or "")[:160],
            "source": source_id if row.get("mode") == "source_transplant" else "",
            "mode": row.get("mode", "fresh"),
        })
    return (
        "\n\nHOST SOURCE-REUSE RECEIPTS (read-only facts, not new requests):\n"
        + json.dumps(summaries, ensure_ascii=False, separators=(",", ":"))
        + "\nOnly source_transplant rows have passed code inspection and compile proof. "
        "Fresh rows need an independent implementation. Never treat a repository search "
        "result or a feature description as working code.\n"
    )


__all__ = [
    "authored_capability_graph",
    "resolve_authored_source_reuse",
    "verified_reuse_context",
]
