"""Request-first feature decomposition with local task-template RAG.

The worksheet is supporting design, never the authority for which requested
features survive. Model pages are bounded; the persisted feature ledger is not
limited to the four records in a worksheet concern.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .fixed_template_generation import generate_fixed_template_value
from .rag_index import ProjectRAGIndex
from .task_template_catalog import RUNTIME_TEMPLATE_ROOT


def _template_rag() -> ProjectRAGIndex:
    roots = [RUNTIME_TEMPLATE_ROOT / name for name in ("design", "translation", "minecraft")]
    identity = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*.yaml")):
            identity.update(str(path.relative_to(RUNTIME_TEMPLATE_ROOT)).encode())
            identity.update(path.read_bytes())
    revision = identity.hexdigest()
    directory = Path(os.environ.get("MMM_REQUEST_RAG_DIR", ".mmm/request-rag"))
    index = ProjectRAGIndex(directory / "templates.sqlite")
    index.build(roots, metadata={
        "minecraft_version": "target-neutral", "loader": "target-neutral",
        "mapping_namespace": "official", "java_version": "not-bound",
        "license": "MIT", "source_commit": revision,
        "source_kind": "host_task_templates",
    })
    return index


def _references(index: ProjectRAGIndex | None, query: str) -> list[dict[str, Any]]:
    if index is None:
        return []
    try:
        hits = index.search(query, limit=3)
    except (OSError, ValueError, RuntimeError):
        return []
    return [{**asdict(hit), "text": hit.text[:1200],
             "authority": "task_template_reference_only"} for hit in hits]


def _request_chunks(prompt: str):
    """Bound input pages without dropping any source characters."""
    start = 0
    while start < len(prompt):
        end = min(start + 768, len(prompt))
        if end < len(prompt):
            boundary = max(prompt.rfind("\n", start + 384, end),
                           prompt.rfind(" ", start + 384, end))
            if boundary > start:
                end = boundary + 1
        yield start, prompt[start:end]
        start = end


def author_request_scope(router: Any, prompt: str, *, budget: Any = None) -> dict[str, Any]:
    """Identify requested features, then retrieve references for each feature."""
    try:
        index = _template_rag()
    except (OSError, ValueError, RuntimeError) as exc:
        index = None
        print(f"REQUEST_RAG: unavailable ({type(exc).__name__}); preserving request", flush=True)
    references = _references(index, prompt)
    if not references:
        references = _references(index, "gameplay request content entity core loop")
    features: list[dict[str, Any]] = []
    for start, chunk in _request_chunks(prompt):
        context = (
            "Original request (scope authority):\n" + chunk
            + "\n\nLocal task-template RAG (design guidance, no target API authority):\n"
            + json.dumps(references, ensure_ascii=False, separators=(",", ":"))
        )
        if budget is not None:
            budget.consume("request_scope.cardinality")
        count = generate_fixed_template_value(router, "planner", (
            {"role": "system", "content": (
                "Decompose the user's Minecraft mod request into distinct playable features. "
                "Count every requested mechanic, including later exploration/combat/progression. "
                "Count features, not worksheet fields, implementation classes or quality checks. "
                "Do not reduce the entire request to its first subsystem or add new mechanics."
            )},
            {"role": "user", "content": context + "\nChoose the number of distinct features in this source page."},
        ), response_schema={
            "type": "object", "properties": {"feature_count": {
                "type": "string", "enum": [str(n) for n in range(1, 33)],
            }}, "required": ["feature_count"], "additionalProperties": False,
        }, enable_tools=False)
        prior: list[dict[str, str]] = []
        for ordinal in range(int(count["feature_count"])):
            if budget is not None:
                budget.consume("request_scope.feature")
            row = generate_fixed_template_value(router, "planner", (
                {"role": "system", "content": (
                    "Identify one requested feature in source order. Write a complete compact "
                    "English statement and a short English mod/source search query. Quote the "
                    "original words identifying it. Keep distinct features distinct. "
                    "RAG examples inform implementation; they do not add user requirements."
                )},
                {"role": "user", "content": context
                 + f"\nFeature {ordinal + 1} of {count['feature_count']}."
                 + "\nAlready identified features (do not repeat):\n"
                 + json.dumps(prior, ensure_ascii=False)},
            ), response_schema={
                "type": "object", "properties": {
                    "statement": {"type": "string", "minLength": 1, "maxLength": 256},
                    "search_query": {"type": "string", "minLength": 1, "maxLength": 96},
                    "source_quote": {"type": "string", "minLength": 1, "maxLength": 192},
                }, "required": ["statement", "search_query", "source_quote"],
                "additionalProperties": False,
            }, enable_tools=False)
            prior.append(dict(row))
            quote = row["source_quote"]
            offset = chunk.find(quote)
            # A model paraphrase cannot masquerade as an exact source quotation.
            source_text = quote if offset >= 0 else chunk
            source_start = start + max(0, offset)
            digest = hashlib.sha256(
                f"{prompt}\0{source_start}\0{ordinal}\0{row['statement']}".encode()
            ).hexdigest()[:16]
            features.append({
                "requirement_id": "request_" + digest,
                "statement": row["statement"], "search_query": row["search_query"],
                "source_span": {"text": source_text, "start": source_start,
                                "end": source_start + len(source_text)},
                "rag_references": _references(index, row["search_query"]),
            })
    print(f"REQUEST_SCOPE: features={len(features)}; original request preserved", flush=True)
    return {"schema_version": "mmm/request-scope-v1", "requested_prompt": prompt,
            "requirements": features}


def request_scope_context(scope: Mapping[str, Any]) -> str:
    rows = scope.get("requirements", ())
    if not rows:
        return ""
    return (
        "\n\nORIGINAL REQUEST FEATURES (preserve every feature across all subsystems):\n"
        + json.dumps([{
            "requirement_id": row["requirement_id"], "statement": row["statement"],
            "source_text": row.get("source_span", {}).get("text", ""),
        } for row in rows], ensure_ascii=False, separators=(",", ":"))
        + "\nA worksheet concern is a bounded supporting view. It cannot exclude "
        "features that belong to another subsystem. Use complete compact sentences.\n"
    )
