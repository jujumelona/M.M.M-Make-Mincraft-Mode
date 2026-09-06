from __future__ import annotations

"""Canonical research-evidence handoff for coding and repair.

This contract keeps the existing research/reuse engines as the single owners of
retrieval.  It only makes their authority explicit at the coder boundary and ensures
that a novel repair diagnostic triggers a fresh, narrow official-API lookup instead
of relying exclusively on stale planning evidence.
"""

import copy
import hashlib
import heapq
import json
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

_MARKER = "_mmm_research_evidence_handoff_v1"
_DEFAULT_RESEARCH_CONTEXT_BYTES = 8 * 1024
_MAX_REPAIR_HITS = 6
_MAX_REPAIR_EXCERPT_CHARS = 1200


def _sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _full_reusable_evidence(context: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    """Persist enough provenance/quality to reuse research without lossy summaries."""

    values = getattr(context, "evidence", {})
    if not isinstance(values, Mapping):
        return []
    ranked = heapq.nsmallest(
        max(1, int(limit)),
        values.values(),
        key=lambda item: (
            -float(getattr(item, "bestfit_score", 0.0) or 0.0),
            str(getattr(item, "path", "")),
            int(getattr(item, "start_line", 1) or 1),
        ),
    )
    reusable: list[dict[str, Any]] = []
    for item in ranked:
        quality = getattr(item, "quality", None)
        to_dict = getattr(quality, "to_dict", None)
        quality_payload = to_dict() if callable(to_dict) else None
        raw_metrics = getattr(item, "metrics", {})
        metrics = (
            {
                str(key): round(float(value), 6)
                for key, value in raw_metrics.items()
                if isinstance(value, (int, float))
            }
            if isinstance(raw_metrics, Mapping)
            else {}
        )
        reusable.append(
            {
                "evidence_id": str(getattr(item, "evidence_id", "")),
                "source_type": str(getattr(item, "source_type", "")),
                "path": str(getattr(item, "path", "")),
                "sha256": str(getattr(item, "sha256", "")),
                "start_line": int(getattr(item, "start_line", 1) or 1),
                "end_line": int(getattr(item, "end_line", 1) or 1),
                "symbols": list(getattr(item, "symbols", ())[:12]),
                "plan_steps": sorted(str(value) for value in getattr(item, "plan_steps", set())),
                "metrics": metrics,
                "quality": quality_payload,
                "bestfit_score": round(float(getattr(item, "bestfit_score", 0.0) or 0.0), 6),
                "graph_hop": getattr(item, "graph_hop", None),
                "algorithmic_plan": str(getattr(item, "algorithmic_plan", ""))[:1024],
                "snippet": str(getattr(item, "text", ""))[:640],
            }
        )
    return reusable


def _selected_fact_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for record in records:
        raw = record.get("fact_ids", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            continue
        for value in raw:
            fact_id = str(value)
            if fact_id and fact_id not in seen:
                seen.add(fact_id)
                result.append(fact_id)
    return result


def _reference_only_context(value: Mapping[str, Any], *, byte_budget: int) -> dict[str, Any]:
    """Mark research as reference evidence without granting source-reuse authority."""

    result = copy.deepcopy(dict(value))
    policy = dict(result.get("policy") or {})
    policy.update(
        {
            "reference_only": True,
            "source_reuse_authority": "approved_reuse_context_only",
            "pinned_license_provenance_required_for_source_reuse": True,
        }
    )
    result["policy"] = policy

    # The original selector guarantees a byte-bounded page.  Preserve that contract
    # after adding the authority marker by dropping only the lowest-ranked tail records.
    budget = max(1024, int(byte_budget))
    if _json_size(result) <= budget:
        return result
    shards = result.pop("ledger_shards", None)
    if shards is not None:
        result["ledger_shards_sha256"] = _sha(shards)
    records = result.get("records")
    if isinstance(records, list):
        while records and _json_size(result) > budget:
            records.pop()
            fact_ids = _selected_fact_ids(
                [item for item in records if isinstance(item, Mapping)]
            )
            result["selected_record_count"] = len(records)
            result["selected_fact_count"] = len(fact_ids)
            result["selected_facts_sha256"] = _sha(fact_ids)
            ledger_fact_count = result.get("ledger_fact_count")
            if isinstance(ledger_fact_count, int):
                result["omitted_fact_count"] = max(0, ledger_fact_count - len(fact_ids))
    if _json_size(result) > budget:
        raise RuntimeError("Coder research evidence contract exceeds its byte budget.")
    return result


def _install_research_selector(research_ledger_module: Any, custom_module_generator_module: Any) -> None:
    current = research_ledger_module.select_module_research_context
    if getattr(current, _MARKER, False):
        wrapped = current
    else:

        @wraps(current)
        def wrapped(*args: Any, **kwargs: Any):
            result = current(*args, **kwargs)
            if not isinstance(result, Mapping):
                return result
            budget = kwargs.get("byte_budget", _DEFAULT_RESEARCH_CONTEXT_BYTES)
            return _reference_only_context(result, byte_budget=int(budget))

        setattr(wrapped, _MARKER, True)
        wrapped.__wrapped__ = current  # type: ignore[attr-defined]
        research_ledger_module.select_module_research_context = wrapped

    # custom_module_generator imported the selector by value, so update that live alias.
    custom_module_generator_module.select_module_research_context = wrapped


def _diagnostic_queries(diagnostic: Mapping[str, Any]) -> tuple[str, ...]:
    symbols = [str(value) for value in diagnostic.get("symbols", ()) if str(value)]
    exceptions = [str(value) for value in diagnostic.get("exceptions", ()) if str(value)]
    files = [Path(str(value)).name for value in diagnostic.get("files", ()) if str(value)]
    tasks = [str(value) for value in diagnostic.get("tasks", ()) if str(value)]
    messages = [
        " ".join(str(value).split())[:1400]
        for value in diagnostic.get("messages", ())
        if str(value).strip()
    ]
    queries = [
        " ".join(
            [
                "exact Minecraft loader API signature compile repair",
                *symbols[:12],
                *exceptions[:6],
                *messages[-2:],
            ]
        ).strip(),
        " ".join(
            [
                "Minecraft build diagnostic dependency contract repair",
                *files[:8],
                *tasks[:6],
                *symbols[:8],
                *messages[-1:],
            ]
        ).strip(),
    ]
    normalized = [" ".join(query.split())[:3600] for query in queries]
    return tuple(dict.fromkeys(query for query in normalized if len(query) >= 8))


def _fresh_official_repair_evidence(
    root: Path,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    from .platform_catalog import adapter_from_project
    from .retrieval import retrieve_official_evidence

    queries = _diagnostic_queries(diagnostic)
    try:
        adapter = adapter_from_project(root)
    except Exception as exc:
        return {
            "schema_version": "mmm/fresh-repair-research-v1",
            "status": "TARGET_UNAVAILABLE",
            "queries_sha256": _sha(queries),
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}"[:512],
        }

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    coverage: list[float] = []
    for query in queries:
        try:
            receipt = retrieve_official_evidence(
                query,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                mappings=adapter.yarn_mappings,
                limit=4,
            )
            payload = receipt.to_dict()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:512])
            continue
        try:
            coverage.append(float(payload.get("coverage", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        for raw in payload.get("hits", ()):
            if not isinstance(raw, Mapping):
                continue
            evidence_id = str(raw.get("evidence_id") or raw.get("document_id") or "")
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            hits.append(
                {
                    "evidence_id": evidence_id,
                    "source_type": "official_documentation",
                    "document_id": str(raw.get("document_id") or ""),
                    "content_sha256": str(raw.get("content_sha256") or ""),
                    "score": float(raw.get("score", 0.0) or 0.0),
                    "excerpt": str(raw.get("excerpt") or "")[:_MAX_REPAIR_EXCERPT_CHARS],
                    "query_sha256": _sha(query),
                }
            )
            if len(hits) >= _MAX_REPAIR_HITS:
                break
        if len(hits) >= _MAX_REPAIR_HITS:
            break

    return {
        "schema_version": "mmm/fresh-repair-research-v1",
        "status": "EVIDENCE_FOUND" if hits else "NO_EVIDENCE",
        "target": {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        },
        "queries_sha256": _sha(queries),
        "query_count": len(queries),
        "hit_count": len(hits),
        "hit_ids_sha256": _sha([item["evidence_id"] for item in hits]),
        "coverage_max": round(max(coverage), 6) if coverage else 0.0,
        "hits": hits,
        "errors": errors,
        "policy": {
            "novel_diagnostic_requires_fresh_targeted_lookup": True,
            "official_api_evidence_precedes_guessing": True,
            "full_project_reresearch": False,
        },
    }


def _install_repair_retrieval(repair_module: Any, diagnostic_payload_fn: Any) -> None:
    cls = repair_module.RepairEngine
    current = cls._context
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def context(self: Any, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        normalized_root = root.expanduser().resolve()
        base = dict(current(self, normalized_root, evidence))
        diagnostic = diagnostic_payload_fn(evidence)
        cache_key = f"{normalized_root}:{_sha(diagnostic)}"
        cache = getattr(self, "_mmm_fresh_repair_research_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._mmm_fresh_repair_research_cache = cache
        fresh = cache.get(cache_key)
        if not isinstance(fresh, Mapping):
            fresh = _fresh_official_repair_evidence(normalized_root, diagnostic)
            cache[cache_key] = copy.deepcopy(fresh)
            while len(cache) > 16:
                cache.pop(next(iter(cache)))
        base["fresh_repair_research"] = copy.deepcopy(dict(fresh))
        retrieval_policy = dict(base.get("retrieval_policy") or {})
        retrieval_policy.update(
            {
                "fresh_targeted_official_retrieval": True,
                "diagnostic_specific_retrieval": True,
                "research_before_repair_guess": True,
            }
        )
        base["retrieval_policy"] = retrieval_policy
        return base

    setattr(context, _MARKER, True)
    context.__wrapped__ = current  # type: ignore[attr-defined]
    cls._context = context


def install(
    *,
    research_ledger_module: Any,
    custom_module_generator_module: Any,
    repair_module: Any,
) -> None:
    """Install one evidence handoff contract over the existing retrieval owners."""

    from . import research_coder_repair_reuse as reuse_hardener

    # The existing hardener owns dependency-neighborhood retrieval and prior-evidence
    # reuse.  Replace only its lossy receipt projection before installing it.
    reuse_hardener._reusable_evidence = _full_reusable_evidence
    reuse_hardener.harden()

    _install_research_selector(research_ledger_module, custom_module_generator_module)
    _install_repair_retrieval(
        repair_module,
        reuse_hardener._diagnostic_signature_payload,
    )


__all__ = [
    "_diagnostic_queries",
    "_full_reusable_evidence",
    "_reference_only_context",
    "install",
]
