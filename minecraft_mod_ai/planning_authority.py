from __future__ import annotations

"""Native request and retrieval-planning authority.

This module owns the pre-design semantic boundary without runtime rebinding. The model
may interpret already-host-owned request clauses and propose retrieval queries, while the
host owns exact source text, offsets, stable IDs, validation, dependency DAG integrity,
and the active planning scope.

Native function calling is used when the router supports it. The fallback protocol is
strict Markdown parsed by the host; free-form JSON is never required from the model.
"""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator

from . import authored_scope_research_contract as _retrieval
from . import evidence_first_planning as _evidence
from . import evidence_request_guard as _guard
from . import semantic_requirement_authority as _semantic


_SEMANTIC_FIELDS = (
    "source_clause_index",
    "capability_id",
    "source_anchor",
    "semantic_statement",
    "given",
    "when",
    "then",
)


def _semantic_text_messages(
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    base = _semantic._model_messages(clauses)
    system = str(base[0]["content"]) + (
        "\n\nOUTPUT PROTOCOL: do not emit JSON. Emit one Markdown block per semantic leaf:\n"
        "### requirement\n"
        "source_clause_index: <integer>\n"
        "capability_id: <lower-case dotted semantic id>\n"
        "source_anchor: <short phrase grounded in the authored clause>\n"
        "semantic_statement: <normalized meaning>\n"
        "given: <observable precondition>\n"
        "when: <player action/event>\n"
        "then: <observable outcome>\n"
        "Repeat the block for every independent behavior. Do not add other headings."
    )
    clause_text = "\n\n".join(
        f"## clause {int(clause['clause_index'])}\n{str(clause['text'])}"
        for clause in clauses
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": clause_text},
    ]


def _parse_semantic_markdown(text: str) -> dict[str, Any]:
    requirements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.casefold() == "### requirement":
            if current is not None:
                requirements.append(current)
            current = {}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().casefold()
        value = value.strip()
        if key not in _SEMANTIC_FIELDS:
            continue
        if key == "source_clause_index":
            try:
                current[key] = int(value)
            except ValueError as exc:
                raise _evidence.EvidencePlanError(
                    f"REQ_MODEL_RESPONSE: invalid source_clause_index {value!r}"
                ) from exc
        else:
            current[key] = value
    if current is not None:
        requirements.append(current)
    if not requirements:
        raise _evidence.EvidencePlanError(
            "REQ_MODEL_RESPONSE: semantic Markdown contained no requirement blocks"
        )
    missing = [
        (index, [field for field in _SEMANTIC_FIELDS if field not in item])
        for index, item in enumerate(requirements)
        if any(field not in item for field in _SEMANTIC_FIELDS)
    ]
    if missing:
        raise _evidence.EvidencePlanError(
            f"REQ_MODEL_RESPONSE: semantic Markdown is incomplete: {missing}"
        )
    return {"requirements": requirements}


def _call_semantic_compiler(router: Any, clauses: Sequence[Mapping[str, Any]]) -> Any:
    max_clause_index = max(int(clause["clause_index"]) for clause in clauses)
    schema = _semantic._semantic_schema(max_clause_index)
    native = getattr(router, "generate_tool_decision", None)
    if callable(native):
        return native(
            "planner",
            _semantic._model_messages(clauses),
            tool_name="compile_semantic_requirements",
            parameters=schema,
            description=(
                "Compile every independently observable authored behavior into semantic "
                "leaf requirements. The host owns source grounding and IDs."
            ),
        )
    raw = router.generate_text(
        "planner",
        _semantic_text_messages(clauses),
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    return _parse_semantic_markdown(str(raw))


def _compile_semantic_catalog(prompt: str, router: Any | None) -> dict[str, Any]:
    if router is None:
        catalog = _semantic.build_approved_requirement_catalog(prompt, router=None)
        return dict(catalog)
    if not isinstance(prompt, str) or not prompt.strip():
        raise _evidence.EvidencePlanError(
            "REQ_SOURCE_EMPTY: semantic authority requires a non-empty prompt."
        )
    clauses = _semantic._clause_records(prompt)
    try:
        payload = _call_semantic_compiler(router, clauses)
    except Exception as exc:
        if isinstance(exc, _evidence.EvidencePlanError):
            raise
        raise _evidence.EvidencePlanError(
            f"semantic compilation failed before host approval: {type(exc).__name__}: {exc}"
        ) from exc
    nodes, invalid_clauses, diagnostics = _semantic._evaluate_batch(payload, clauses)
    if invalid_clauses:
        raise _evidence.EvidencePlanError(
            "semantic compilation did not satisfy the host contract: "
            + _semantic._canonical(
                {
                    "invalid_clause_indices": sorted(invalid_clauses),
                    "diagnostics": diagnostics,
                    "generation_policy": "single_pass_native_authority",
                }
            )
        )
    catalog = _semantic._build_catalog(
        prompt,
        _semantic._assign_local_ids(nodes),
        clauses,
    )
    audit = dict(catalog.get("semantic_audit") or {})
    audit.update(
        {
            "normal_model_turns": 1,
            "max_repair_turns": 0,
            "generation_policy": "single_pass_native_authority",
        }
    )
    catalog["semantic_audit"] = audit
    catalog["catalog_sha256"] = ""
    catalog["catalog_sha256"] = _evidence._hash_without(catalog, "catalog_sha256")
    _semantic.validate_approved_requirement_catalog(catalog, prompt=prompt)
    return catalog


def _retrieval_text_messages(
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    base = _retrieval._retrieval_plan_messages(prompt, requirements)
    system = str(base[0]["content"]) + (
        "\n\nOUTPUT PROTOCOL: do not emit JSON. For every supplied requirement emit:\n"
        "### <requirement_id>\n"
        "depends_on: <comma-separated requirement IDs, or none>\n"
        "query: <English retrieval query>\n"
        "query: <another English retrieval query>\n"
        "Use 2-5 query lines. Emit every requirement exactly once."
    )
    rendered = []
    for raw in requirements:
        span = raw.get("source_span")
        source = str(span.get("text") or "") if isinstance(span, Mapping) else ""
        rendered.append(
            "\n".join(
                (
                    f"## {raw.get('requirement_id', '')}",
                    f"capability: {raw.get('capability', '')}",
                    f"semantic_statement: {raw.get('semantic_statement', '')}",
                    f"source_text: {source}",
                )
            )
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(rendered)},
    ]


def _parse_retrieval_markdown(text: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("### "):
            if current is not None:
                rows.append(current)
            current = {
                "requirement_id": line[4:].strip(),
                "depends_on": [],
                "search_queries": [],
            }
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().casefold()
        value = value.strip()
        if key == "depends_on":
            if value.casefold() not in {"", "none", "null", "n/a", "-"}:
                current["depends_on"] = [
                    item.strip() for item in value.split(",") if item.strip()
                ]
        elif key == "query" and value:
            current["search_queries"].append(value)
    if current is not None:
        rows.append(current)
    if not rows:
        raise ValueError("retrieval planner Markdown contained no requirement blocks")
    return {"requirements": rows}


def _call_retrieval_planner(
    router: Any,
    prompt: str,
    requirements: Sequence[Mapping[str, Any]],
) -> Any:
    ids = [str(item.get("requirement_id") or "") for item in requirements]
    schema = _retrieval._retrieval_plan_schema(ids)
    native = getattr(router, "generate_tool_decision", None)
    if callable(native):
        return native(
            "planner",
            _retrieval._retrieval_plan_messages(prompt, requirements),
            tool_name="plan_requirement_retrieval",
            parameters=schema,
            description=(
                "Create authored prerequisite edges and English atomic retrieval queries "
                "for every frozen requirement."
            ),
        )
    raw = router.generate_text(
        "planner",
        _retrieval_text_messages(prompt, requirements),
        response_format="text",
        response_schema=None,
        tool_stage="research_query_planning",
        enable_tools=False,
    )
    return _parse_retrieval_markdown(str(raw))


def _enrich_retrieval_plan(
    prompt: str,
    catalog: Mapping[str, Any],
    router: Any | None,
) -> dict[str, Any]:
    if router is None:
        return dict(catalog)
    requirements = catalog.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return dict(catalog)
    payload = _call_retrieval_planner(router, prompt, requirements)
    plan = _retrieval._normalize_retrieval_plan(prompt, requirements, payload)

    enriched = deepcopy(dict(catalog))
    enriched_requirements: list[dict[str, Any]] = []
    edges: list[list[str]] = []
    for raw in enriched["requirements"]:
        item = dict(raw)
        rid = str(item.get("requirement_id") or "")
        planned = plan[rid]
        item["depends_on"] = list(planned["depends_on"])
        item["search_queries"] = list(planned["search_queries"])
        edges.extend([[dep, rid] for dep in item["depends_on"]])
        enriched_requirements.append(item)
    enriched["requirements"] = enriched_requirements
    enriched["requirement_graph"] = {
        "node_ids": [str(item["requirement_id"]) for item in enriched_requirements],
        "edges": edges,
    }
    audit = dict(enriched.get("semantic_audit") or {})
    audit["normal_model_turns"] = int(audit.get("normal_model_turns") or 1) + 1
    audit["retrieval_query_planning"] = "atomic_english_multi_query"
    audit["dependency_edge_count"] = len(edges)
    enriched["semantic_audit"] = audit
    enriched["catalog_sha256"] = ""
    enriched["catalog_sha256"] = _evidence._hash_without(enriched, "catalog_sha256")
    _semantic.validate_approved_requirement_catalog(enriched, prompt=prompt)
    return enriched


def build_authoritative_request_catalog(
    prompt: str,
    router: Any | None,
) -> dict[str, Any]:
    """Compile request meaning and retrieval intent before any design/RAG execution."""

    return _enrich_retrieval_plan(
        prompt,
        _compile_semantic_catalog(prompt, router),
        router,
    )


@contextmanager
def authoritative_request_scope(
    prompt: str,
    catalog: Mapping[str, Any],
) -> Iterator[None]:
    """Expose one immutable request catalog to downstream read-only planning helpers."""

    token = _guard._ACTIVE_REQUEST_CATALOG.set((prompt, deepcopy(dict(catalog))))
    try:
        yield
    finally:
        _guard._ACTIVE_REQUEST_CATALOG.reset(token)


__all__ = [
    "authoritative_request_scope",
    "build_authoritative_request_catalog",
]
