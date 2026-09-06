from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from minecraft_mod_ai import evidence_first_planning as planning


def request_catalog(
    prompt: str,
    requirements: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a validated downstream request-catalog fixture without parsing raw prompt text.

    Tests that exercise post-planning consumers must supply the same authority shape that
    production receives from ``planning_state_handoff``.  This helper deliberately takes
    explicit semantic requirements; it is not a test-only prompt parser.
    """

    authored = str(prompt)
    if not authored:
        raise ValueError("test request prompt must not be empty")
    raw_requirements = list(requirements or ({},))
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_requirements, start=1):
        statement = str(raw.get("statement") or authored).strip()
        source_text = str(raw.get("source_text") or authored)
        start = authored.find(source_text)
        if start < 0:
            raise ValueError(f"test source_text is not an exact prompt span: {source_text!r}")
        end = start + len(source_text)
        requirement_id = str(raw.get("requirement_id") or f"req_{index:03d}")
        capability = str(raw.get("capability") or f"researched.test_{index:03d}").strip()
        capability = capability.removeprefix("capability:")
        canonical = f"capability:{capability}"
        acceptance = [
            str(item).strip()
            for item in raw.get("acceptance", [f"Observe the authored behavior: {statement}"])
            if str(item).strip()
        ]
        implementation_capabilities = [
            str(item).strip()
            for item in raw.get("implementation_capabilities", [capability])
            if str(item).strip()
        ]
        implementation_obligations = [
            str(item).strip()
            for item in raw.get(
                "implementation_obligations",
                [f"Implement the authored requirement deterministically: {statement}"],
            )
            if str(item).strip()
        ]
        search_queries = [
            str(item).strip()
            for item in raw.get("search_queries", [])
            if str(item).strip()
        ]
        depends_on = [
            str(item).strip()
            for item in raw.get("depends_on", [])
            if str(item).strip()
        ]
        semantic_type = str(raw.get("semantic_type") or "researched_gameplay_requirement")
        row = {
            "requirement_id": requirement_id,
            "capability": capability,
            "statement": statement,
            "semantic_statement": statement,
            "mandatory": True,
            "provenance_role": "authored",
            "source_span": {
                "source_id": "requested_prompt",
                "char_start": start,
                "char_end": end,
                "text": source_text,
                "text_sha256": planning._sha(source_text),
            },
            "evidence_refs": [],
            "derived_from": [],
            "depends_on": depends_on,
            "provides": [canonical],
            "gameplay_capabilities": [capability],
            "implementation_capabilities": implementation_capabilities,
            "implementation_obligations": implementation_obligations,
            "artifact_task_ids": [
                planning._stable_id(
                    "task",
                    item,
                    {"requirement_id": requirement_id, "layer": "test_fixture"},
                )
                for item in implementation_capabilities
            ],
            "semantic_type": semantic_type,
            "unlock_policy": {
                "required_capabilities": [],
                "required_requirement_refs": depends_on,
                "optional_capabilities": [],
                "optional_requirement_refs": [],
                "policy": "grounded_planning_state_only",
            },
            "artifact_obligations": list(raw.get("artifact_obligations", [])),
            "design_resolution_obligations": implementation_obligations,
            "runtime_acceptance": acceptance,
            "semantic_status": "RESOLVED",
            "unresolved_spans": [],
            "acceptance": acceptance,
            "observable_behavior": {
                "given": "the authored preconditions are established",
                "when": statement,
                "then": acceptance[0],
            },
            "template_profile": {
                "template_id": "grounded_researched_requirement",
                "architecture_owner": "planning_state",
            },
            "search_queries": search_queries,
            "reuse_candidates": list(raw.get("reuse_candidates", [])),
            "detailed_plan_ref": str(raw.get("detailed_plan_ref") or f"detail_{index:03d}"),
            "engineering_worksheet": raw.get("engineering_worksheet"),
        }
        rows.append(row)

    ids = {row["requirement_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("test request requirement IDs must be unique")
    if any(dep not in ids for row in rows for dep in row["depends_on"]):
        raise ValueError("test request dependency references an unknown requirement")

    catalog: dict[str, Any] = {
        "prompt_sha256": planning._sha(authored),
        "prompt_char_length": len(authored),
        "purpose": authored,
        "requirements": rows,
        "constraints": [],
        "non_goals": [],
        "deployment_expectations": [],
        "requirement_graph": {
            "node_ids": [row["requirement_id"] for row in rows],
            "edges": [
                {"from": dependency, "to": row["requirement_id"]}
                for row in rows
                for dependency in row["depends_on"]
            ],
        },
        "dependency_provenance": [],
        "semantic_audit": {
            "status": "APPROVED",
            "authored_clause_count": len(rows),
            "covered_clause_count": len(rows),
            "unresolved_clause_count": 0,
            "unsupported_design_choice_count": 0,
            "generation_policy": "test_explicit_grounded_catalog",
            "source_grounding_owner": "test_fixture_explicit_spans",
            "capability_id_owner": "test_fixture_explicit_semantics",
            "dependency_owner": "test_fixture_explicit_semantics",
            "implementation_architecture_owner": "grounded_detailed_plan",
            "research_query_owner": "host_fixture",
        },
        "planning_state_sha256": "sha256:" + "0" * 64,
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = planning._hash_without(catalog, "catalog_sha256")
    planning._validate_request_catalog(catalog, prompt=authored)
    return catalog


def design_with_catalog(
    prompt: str,
    design: Mapping[str, Any] | None = None,
    requirements: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    value = dict(design or {})
    value["_evidence_request_catalog"] = request_catalog(prompt, requirements)
    return value


__all__ = ["design_with_catalog", "request_catalog"]
