from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .model_adapters import ModelConfigurationError
from .project_index import ProjectIndex
from .repository_explorer import RepositoryExplorer, exploration_fingerprint


def build_repository_observation_ledger(
    router: Any,
    index: ProjectIndex,
    *,
    query: str,
    byte_budget: int,
    diagnostic_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Build exact, line-ranked host grounding from the task-adaptive explorer."""
    if type(byte_budget) is not int or byte_budget < 1024:
        raise ValueError("repository grounding byte_budget must be an integer >= 1024")
    query = str(query or "").strip()
    if not query:
        raise ValueError("repository grounding query must be non-empty")

    explorer = RepositoryExplorer(index, router=router)
    line_budget = max(8, byte_budget // 192)
    degraded: list[str] = []
    try:
        exploration = explorer.explore(
            query,
            diagnostic_paths=diagnostic_paths,
            line_budget=line_budget,
        )
    except ModelConfigurationError as exc:
        degraded.append(f"{type(exc).__name__}: {exc}")
        exploration = explorer.explore(
            query,
            diagnostic_paths=diagnostic_paths,
            line_budget=line_budget,
            semantic=False,
            rerank=False,
        )

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for region in exploration.regions:
        record = _exact_region_record(index, region.to_dict())
        key = (
            record["path"],
            int(record["content_start_bytes"]),
            int(record["content_end_bytes"]),
        )
        if key in seen:
            continue
        candidate = {
            "schema_version": "mmm/source-observation-ledger-v2",
            "records": [*records, record],
        }
        if _json_size(candidate) > byte_budget:
            if records:
                break
            record = _clip_record(record, max(1, byte_budget // 2))
            if _json_size(
                {"schema_version": "mmm/source-observation-ledger-v2", "records": [record]}
            ) > byte_budget:
                raise ValueError("repository grounding budget is too small for one exact region")
        seen.add(key)
        records.append(record)

    if not records:
        raise ValueError("repository explorer produced no exact source region")

    observation_digest = hashlib.sha256()
    for record in records:
        _update_digest(observation_digest, record)
    query_sha256 = "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()
    project_sha256 = str(index.manifest_receipt()["sha256"])
    receipt = {
        "schema_version": "mmm/source-observation-receipt-v2",
        "project_sha256": project_sha256,
        "query_sha256": query_sha256,
        "observation_count": len(records),
        "observations_sha256": "sha256:" + observation_digest.hexdigest(),
        "exploration_sha256": exploration_fingerprint(exploration),
        "retrieval_route": exploration.route,
        "line_budget": exploration.line_budget,
        "lines_selected": sum(
            max(1, int(item["end_line"]) - int(item["start_line"]) + 1)
            for item in records
        ),
        "semantic_used": exploration.semantic_used,
        "rerank_used": exploration.rerank_used,
        "degraded_retrieval": degraded,
        "missing_terms": list(exploration.missing_terms),
        "policy": {
            "exact_source_quotes": True,
            "path_sha256_byte_range_bound": True,
            "task_adaptive_retrieval": True,
            "line_ranked_context": True,
            "generic_similar_code_not_authoritative": True,
        },
    }
    return {
        "schema_version": "mmm/source-observation-ledger-v2",
        "receipt": receipt,
        "exploration": exploration.to_dict(),
        "records": records,
    }


def build_repair_repository_context(
    router: Any,
    index: ProjectIndex,
    *,
    query: str,
    diagnostic_paths: Iterable[str] = (),
    byte_budget: int | None = None,
) -> dict[str, Any]:
    budget = index.policy.model_context_bytes if byte_budget is None else int(byte_budget)
    ledger = build_repository_observation_ledger(
        router,
        index,
        query=query,
        diagnostic_paths=diagnostic_paths,
        byte_budget=budget,
    )
    return {
        "schema_version": "mmm/repair-repository-context-v1",
        "manifest": index.manifest_receipt(),
        "retrieval_receipt": dict(ledger["receipt"]),
        "exploration": dict(ledger["exploration"]),
        "relevant": {
            "schema_version": "mmm/project-context-regions-v1",
            "selected_file_count": len({item["path"] for item in ledger["records"]}),
            "selected_region_count": len(ledger["records"]),
            "files": list(ledger["records"]),
        },
    }


def _exact_region_record(index: ProjectIndex, region: dict[str, Any]) -> dict[str, Any]:
    path_value = str(region["path"])
    item = index._by_path.get(path_value)
    if item is None:
        raise ValueError(f"repository explorer returned a path outside ProjectIndex: {path_value}")
    path = index.root / path_value
    raw = path.read_bytes()
    current_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    if current_sha != item.sha256:
        raise ValueError(f"Project source changed after its context index was built: {path_value}")

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    start_line = max(1, int(region["start_line"]))
    end_line = max(start_line, int(region["end_line"]))
    start_index = min(len(lines), start_line - 1)
    end_index = min(len(lines), end_line)
    prefix = "".join(lines[:start_index]).encode("utf-8")
    selected = "".join(lines[start_index:end_index]).encode("utf-8")
    core = {
        "path": path_value,
        "sha256": item.sha256,
        "start_line": start_line,
        "end_line": max(start_line, start_index + max(1, end_index - start_index)),
        "content_start_bytes": len(prefix),
        "content_end_bytes": len(prefix) + len(selected),
        "kind": "ranked_exact_source_region",
        "symbol": str(region.get("symbol", "")),
        "symbol_kind": str(region.get("kind", "")),
        "retrieval_scores": {
            key: float(region.get(key, 0.0) or 0.0)
            for key in (
                "score",
                "lexical_score",
                "graph_score",
                "api_score",
                "procedural_score",
                "semantic_score",
                "reranker_score",
                "diagnostic_score",
            )
        },
        "text": selected.decode("utf-8", errors="strict"),
    }
    core["observation_id"] = "obs_" + hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return core


def _clip_record(record: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    raw = str(record.get("text", "")).encode("utf-8")
    clipped = raw[:max_bytes].decode("utf-8", errors="ignore").encode("utf-8")
    value = dict(record)
    value["text"] = clipped.decode("utf-8")
    value["content_end_bytes"] = int(value["content_start_bytes"]) + len(clipped)
    value["end_line"] = int(value["start_line"]) + max(0, value["text"].count("\n"))
    value["observation_id"] = "obs_" + hashlib.sha256(
        json.dumps(
            {key: val for key, val in value.items() if key != "observation_id"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return value


def _json_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _update_digest(digest: Any, value: Any) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


__all__ = ["build_repair_repository_context", "build_repository_observation_ledger"]
