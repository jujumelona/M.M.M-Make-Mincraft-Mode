"""Request-first feature decomposition with local task-template RAG.

The original request remains the scope authority. Each small-model response
identifies one feature or ends the page; a predicted count never forces it to
invent or repeat features. Recovery preserves source text instead of aborting.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .fixed_template_generation import generate_fixed_template_value
from .rag_index import ProjectRAGIndex
from .task_template_catalog import RUNTIME_TEMPLATE_ROOT
from .structured_output import StructuredOutputValidationError

_FEATURE_SCHEMA = {
    "type": "object", "properties": {
        "status": {"type": "string", "enum": ["complete", "feature"]},
        "statement": {"type": "string", "maxLength": 256},
        "search_query": {"type": "string", "maxLength": 96},
        "source_quote": {"type": "string", "maxLength": 192},
        "implementation_role": {"type": "string", "enum": ["gameplay", "interface", "infrastructure"]},
    }, "required": ["status", "statement", "search_query", "source_quote", "implementation_role"],
    "additionalProperties": False,
}


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
        hits = index.search(query, limit=2)
    except (OSError, ValueError, RuntimeError, sqlite3.Error):
        return []
    return [{**asdict(hit), "text": hit.text[:800],
             "authority": "task_template_reference_only"} for hit in hits]


def _request_chunks(prompt: str, *, limit: int = 768):
    """Bound input pages without dropping any source characters."""
    start = 0
    while start < len(prompt):
        end = min(start + limit, len(prompt))
        if end < len(prompt):
            boundary = max(prompt.rfind("\n", start + limit // 2, end),
                           prompt.rfind(" ", start + limit // 2, end))
            if boundary > start:
                end = boundary + 1
        yield start, prompt[start:end]
        start = end


def _source_quote_bounds(source: str, quote: str) -> tuple[int, int] | None:
    """Find a verbatim source span allowing only whitespace differences."""
    words = str(quote).split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, source)
    return (match.start(), match.end()) if match else None


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(" ".join(str(row[key]).split()).casefold()
                 for key in ("statement", "source_quote"))


def _uncovered_source(chunk: str, prior: list[dict[str, Any]]) -> str:
    remaining = chunk
    for quote in dict.fromkeys(row["source_quote"] for row in prior):
        bounds = _source_quote_bounds(remaining, quote)
        if bounds is not None:
            left, right = bounds
            remaining = remaining[:left] + " " * (right - left) + remaining[right:]
    return remaining


def _feature_messages(context: str, chunk: str, prior: list[dict[str, Any]], repair_note: str):
    return (
        {"role": "system", "content": (
            "Identify the next separately actionable Minecraft mod feature in source order. "
            "Include exploration, combat and progression wherever requested. "
            "Return status=feature with a complete compact English statement, a short "
            "English mod/source search query, and a verbatim original source quote. "
            "Classify it as gameplay, interface, or infrastructure. Use interface only "
            "when the feature itself is a screen or UI. Distinct features may share "
            "a quote. When all features are covered, return status=complete with "
            "empty statement, search_query and source_quote. Constraints alone need "
            "no additional feature. Do not repeat features or invent mechanics to "
            "fill a count. RAG examples provide guidance, not user requirements."
        )},
        {"role": "user", "content": context
         + "\nAlready identified features (do not repeat):\n"
         + json.dumps([{"statement": row["statement"], "source_quote": row["source_quote"]}
                       for row in prior[-8:]], ensure_ascii=False, separators=(",", ":"))
         + "\nStill-unquoted original words (guidance; shared quotes may have further features):\n"
         + _uncovered_source(chunk, prior) + repair_note},
    )


def _new_feature(row: Mapping[str, Any], chunk: str, identities: set[tuple[str, str]]) -> bool:
    return (
        all(str(row[key]).strip() for key in ("statement", "search_query", "source_quote"))
        and _source_quote_bounds(chunk, row["source_quote"]) is not None
        and _identity(row) not in identities
    )


def _feature_record(prompt: str, start: int, chunk: str, row: Mapping[str, Any], index: Any):
    quote = str(row["source_quote"])
    bounds = _source_quote_bounds(chunk, quote)
    offset, end = bounds if bounds is not None else (0, len(chunk))
    source_text = chunk[offset:end]
    source_start = start + offset
    digest = hashlib.sha256(
        f"{prompt}\0{source_start}\0{row['statement']}".encode()
    ).hexdigest()[:16]
    return {
        "requirement_id": "request_" + digest,
        "statement": row["statement"], "search_query": row["search_query"],
        "implementation_role": row["implementation_role"],
        "source_span": {"text": source_text, "start": source_start,
                        "end": source_start + len(source_text)},
        "rag_references": _references(index, str(row["search_query"])),
    }


def _uncovered_spans(chunk: str, prior: list[dict[str, Any]]):
    covered = bytearray(len(chunk))
    for quote in dict.fromkeys(row["source_quote"] for row in prior):
        bounds = _source_quote_bounds(chunk, quote)
        if bounds is not None:
            offset, end = bounds
            covered[offset:end] = bytes([1]) * (end - offset)
    offset = 0
    while offset < len(chunk):
        if covered[offset]:
            offset += 1
            continue
        end = offset + 1
        while end < len(chunk) and not covered[end]:
            end += 1
        yield offset, chunk[offset:end]
        offset = end


def _source_recovery(prompt: str, start: int, chunk: str, prior: list[dict[str, Any]], index: Any):
    """Keep actual unquoted request text in the ledger if decomposition stalls."""
    for span_start, source in _uncovered_spans(chunk, prior):
        for offset, text in _request_chunks(source, limit=192):
            trimmed = text.strip()
            if not any(char.isalnum() for char in trimmed):
                continue
            source_start = start + span_start + offset + len(text) - len(text.lstrip())
            row = {"statement": trimmed, "search_query": trimmed[:96],
                   "source_quote": trimmed, "implementation_role": "gameplay"}
            record = _feature_record(prompt, source_start, trimmed, row, index)
            record["source_recovery"] = True
            yield record


def author_request_scope(router: Any, prompt: str, *, budget: Any = None) -> dict[str, Any]:
    """Identify requested features without making extraction a production gate."""
    try:
        index = _template_rag()
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        index = None
        print(f"REQUEST_RAG: unavailable ({type(exc).__name__}); preserving request", flush=True)
    features: list[dict[str, Any]] = []
    for start, chunk in _request_chunks(prompt):
        context = (
            "Original request (scope authority):\n" + chunk
            + "\n\nLocal task-template RAG (guidance, no target API authority):\n"
            + json.dumps(_references(index, chunk), ensure_ascii=False, separators=(",", ":"))
        )
        prior: list[dict[str, Any]] = []
        identities: set[tuple[str, str]] = set()
        repeats = 0
        repair_note = ""
        # One ordered response loop handles both progress and recovery. A finite
        # source page cannot cause endless paraphrases; exhaustion keeps its text.
        for _ in range(max(3, len(chunk) * 3)):
            if budget is not None:
                budget.consume("request_scope.feature")
            try:
                row = generate_fixed_template_value(
                    router, "planner", _feature_messages(context, chunk, prior, repair_note),
                    response_schema=_FEATURE_SCHEMA, enable_tools=False,
                )
            except StructuredOutputValidationError:
                features.extend(_source_recovery(prompt, start, chunk, prior, index))
                break
            if row["status"] == "complete":
                break
            if not _new_feature(row, chunk, identities):
                repeats += 1
                if repeats < 3:
                    repair_note = (
                        "\nThe previous feature was already covered, empty, or not quoted "
                        "from this source. Extract a different feature from the remaining "
                        "source, or return status=complete if none remains."
                    )
                    continue
                features.extend(_source_recovery(prompt, start, chunk, prior, index))
                break
            identities.add(_identity(row))
            prior.append(dict(row))
            features.append(_feature_record(prompt, start, chunk, row, index))
            repeats = 0
            repair_note = ""
        else:
            features.extend(_source_recovery(prompt, start, chunk, prior, index))
    recovered = sum(bool(row.get("source_recovery")) for row in features)
    print(f"REQUEST_SCOPE: features={len(features)} source_recovery={recovered}; original request preserved", flush=True)
    scope = {"schema_version": "mmm/request-scope-v1", "requested_prompt": prompt,
             "requirements": features}
    # Planning is user-facing design authoring, not production construction.
    # Generate observable behavior tasks later from these preserved source spans.
    return scope


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
