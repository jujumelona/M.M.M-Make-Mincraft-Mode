from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

from .model_adapters import ModelConfigurationError
from .procedural_retrieval import decompose_task_procedure, procedural_region_score
from .project_index import ProjectIndex
from .repository_explorer import CodeRegion, RepositoryExplorer, exploration_fingerprint

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,127}")
_BASELINE_ANCHOR_TERMS = frozenset(
    {
        "api",
        "contract",
        "dependency",
        "implements",
        "interface",
        "public",
        "register",
        "required",
        "schema",
    }
)
_HOST_ENTRY_PATHS = frozenset(
    {
        "build.gradle",
        "settings.gradle",
        "gradle.properties",
        "src/main/resources/fabric.mod.json",
    }
)


def build_repository_observation_ledger(
    router: Any,
    index: ProjectIndex,
    *,
    query: str,
    byte_budget: int,
    diagnostic_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Build bounded exact grounding from anchors, regions and procedural alignment."""
    if type(byte_budget) is not int or byte_budget < 1024:
        raise ValueError("repository grounding byte_budget must be an integer >= 1024")
    query = str(query or "").strip()
    if not query:
        raise ValueError("repository grounding query must be non-empty")

    diagnostics = tuple(str(value) for value in diagnostic_paths if str(value).strip())
    baseline_budget = max(512, byte_budget // 2)
    records, baseline_meta = _baseline_anchor_records(
        index,
        query=query,
        diagnostic_paths=diagnostics,
        byte_budget=baseline_budget,
    )
    seen = {
        (
            str(record["path"]),
            int(record["content_start_bytes"]),
            int(record["content_end_bytes"]),
        )
        for record in records
    }

    explorer = RepositoryExplorer(index, router=router)
    procedure_plan = decompose_task_procedure(query)
    line_budget = max(8, byte_budget // 192)
    degraded: list[str] = []
    exploration = _explore_with_degraded_fallback(
        explorer,
        query,
        diagnostics=diagnostics,
        line_budget=line_budget,
        degraded=degraded,
        lane="task",
    )
    procedure_exploration = None
    if index.files and procedure_plan.steps:
        procedure_query = " procedure-flow ".join(procedure_plan.steps)
        procedure_exploration = _explore_with_degraded_fallback(
            explorer,
            procedure_query,
            diagnostics=diagnostics,
            line_budget=line_budget,
            degraded=degraded,
            lane="procedure",
        )

    region_candidates = list(exploration.regions)
    if procedure_exploration is not None:
        region_candidates.extend(procedure_exploration.regions)
    ranked_regions = _procedurally_ranked_regions(region_candidates, procedure_plan.steps)

    procedure_hits = 0
    for region, alignment, observed_steps in ranked_regions:
        record = _exact_region_record(index, region.to_dict())
        record["retrieval_scores"]["procedure_alignment"] = round(alignment, 6)
        if alignment > 0:
            procedure_hits += 1
            record["procedure_trace"] = list(observed_steps[:24])
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
            remaining = max(
                0,
                byte_budget
                - _json_size(
                    {
                        "schema_version": "mmm/source-observation-ledger-v2",
                        "records": records,
                    }
                )
                - 256,
            )
            if remaining < 64:
                break
            clipped = _clip_record(record, remaining)
            if not clipped.get("text"):
                break
            candidate["records"][-1] = clipped
            if _json_size(candidate) > byte_budget:
                break
            record = clipped
            key = (
                record["path"],
                int(record["content_start_bytes"]),
                int(record["content_end_bytes"]),
            )
        seen.add(key)
        records.append(record)

    observation_digest = hashlib.sha256()
    for record in records:
        _update_digest(observation_digest, record)
    query_sha256 = "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()
    project_sha256 = str(index.manifest_receipt()["sha256"])
    procedural_receipt = {
        "plan": procedure_plan.to_dict(),
        "candidate_region_count": len(ranked_regions),
        "aligned_region_count": procedure_hits,
        "secondary_procedure_query_used": procedure_exploration is not None,
        "generic_semantic_similarity_is_not_procedural_authority": True,
    }
    receipt = {
        "schema_version": "mmm/source-observation-receipt-v2",
        "project_sha256": project_sha256,
        "query_sha256": query_sha256,
        "source_page_count": baseline_meta["source_partition_count"],
        "observation_count": len(records),
        "observations_sha256": "sha256:" + observation_digest.hexdigest(),
        "exploration_sha256": exploration_fingerprint(exploration),
        "procedural_retrieval_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(
                procedural_receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "retrieval_route": exploration.route,
        "baseline_anchor_count": baseline_meta["anchor_count"],
        "baseline_candidate_count": baseline_meta["candidate_count"],
        "line_budget": exploration.line_budget,
        "lines_selected": sum(_record_line_count(item) for item in records),
        "semantic_used": exploration.semantic_used
        or bool(procedure_exploration and procedure_exploration.semantic_used),
        "rerank_used": exploration.rerank_used
        or bool(procedure_exploration and procedure_exploration.rerank_used),
        "procedure_decomposition_used": True,
        "procedure_step_count": len(procedure_plan.steps),
        "procedural_similarity_used": bool(ranked_regions),
        "procedural_aligned_region_count": procedure_hits,
        "degraded_retrieval": degraded,
        "missing_terms": list(exploration.missing_terms),
        "policy": {
            "exact_source_quotes": True,
            "path_sha256_byte_range_bound": True,
            "global_contract_anchors_before_ranked_regions": True,
            "task_adaptive_retrieval": True,
            "line_ranked_context": True,
            "ordered_procedure_alignment": True,
            "greenfield_zero_source_is_valid": True,
            "generic_similar_code_not_authoritative": True,
        },
    }
    ledger = {
        "schema_version": "mmm/source-observation-ledger-v2",
        "receipt": receipt,
        "exploration": exploration.to_dict(),
        "procedural_retrieval": procedural_receipt,
        "records": records,
    }
    while records and _json_size(ledger) > byte_budget:
        records.pop()
    return ledger


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
        "procedural_retrieval": dict(ledger["procedural_retrieval"]),
        "relevant": {
            "schema_version": "mmm/project-context-regions-v1",
            "selected_file_count": len({item["path"] for item in ledger["records"]}),
            "selected_region_count": len(ledger["records"]),
            "files": list(ledger["records"]),
        },
    }


def _explore_with_degraded_fallback(
    explorer: RepositoryExplorer,
    query: str,
    *,
    diagnostics: tuple[str, ...],
    line_budget: int,
    degraded: list[str],
    lane: str,
):
    try:
        return explorer.explore(
            query,
            diagnostic_paths=diagnostics,
            line_budget=line_budget,
        )
    except ModelConfigurationError as exc:
        degraded.append(f"{lane}:{type(exc).__name__}: {exc}")
        return explorer.explore(
            query,
            diagnostic_paths=diagnostics,
            line_budget=line_budget,
            semantic=False,
            rerank=False,
        )


def _procedurally_ranked_regions(
    regions: Iterable[CodeRegion],
    plan_steps: tuple[str, ...],
) -> list[tuple[CodeRegion, float, tuple[str, ...]]]:
    unique: dict[tuple[str, int, int], CodeRegion] = {}
    for region in regions:
        key = (region.path, region.start_line, region.end_line)
        current = unique.get(key)
        if current is None or region.score > current.score:
            unique[key] = region
    ranked: list[tuple[float, float, int, str, int, CodeRegion, tuple[str, ...]]] = []
    for region in unique.values():
        alignment, observed = procedural_region_score(
            decompose_task_procedure(" -> ".join(plan_steps)),
            region.text,
        )
        combined = region.score + 3.0 * alignment
        ranked.append(
            (
                -combined,
                -alignment,
                region.line_count,
                region.path,
                region.start_line,
                region,
                observed,
            )
        )
    ranked.sort(key=lambda item: item[:5])
    return [(item[5], -item[1], item[6]) for item in ranked]


def _baseline_anchor_records(
    index: ProjectIndex,
    *,
    query: str,
    diagnostic_paths: Iterable[str],
    byte_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    query_terms = {token.casefold() for token in _TOKEN.findall(query)}
    explicit = {
        index._normalize_path(value)
        for value in diagnostic_paths
        if str(value).strip()
    }
    ranked: list[tuple[int, int, str, Any]] = []
    eligible_bytes = 0
    for item in index.files:
        if item.size_bytes > index.policy.max_single_file_bytes:
            continue
        eligible_bytes += max(1, int(item.size_bytes))
        path_terms = {token.casefold() for token in _TOKEN.findall(item.path)}
        item_terms = {str(token).casefold() for token in item.tokens}
        score = (
            (1_000_000 if item.path in explicit else 0)
            + 60 * len(query_terms & path_terms)
            + 8 * len(query_terms & item_terms)
            + 20 * len(_BASELINE_ANCHOR_TERMS & item_terms)
            + (100 if item.path in _HOST_ENTRY_PATHS else 0)
            + (80 if item.path.endswith(("Mod.java", "Client.java")) else 0)
        )
        ranked.append((-score, int(item.size_bytes), item.path, item))
    ranked.sort()

    records: list[dict[str, Any]] = []
    for _negative_score, _size, _path_value, item in ranked:
        path = index.root / item.path
        raw = path.read_bytes()
        current_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
        if current_sha != item.sha256:
            raise ValueError(f"Project source changed after its context index was built: {item.path}")
        normalized = raw.decode("utf-8", errors="replace").encode("utf-8")
        if not normalized and records:
            continue
        record = _baseline_record(item.path, item.sha256, normalized)
        candidate = {
            "schema_version": "mmm/source-observation-ledger-v2",
            "records": [*records, record],
        }
        if _json_size(candidate) > byte_budget:
            if records:
                continue
            clipped = _clip_record(record, max(1, byte_budget // 2))
            candidate["records"][-1] = clipped
            if _json_size(candidate) > byte_budget:
                continue
            record = clipped
        records.append(record)

    partition_bytes = max(1024, byte_budget)
    source_partition_count = 0 if not ranked else max(1, math.ceil(eligible_bytes / partition_bytes))
    return records, {
        "anchor_count": len(records),
        "candidate_count": len(ranked),
        "source_partition_count": source_partition_count,
    }


def _baseline_record(path: str, sha256: str, content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8", errors="strict")
    core = {
        "path": path,
        "sha256": sha256,
        "start_line": 1,
        "end_line": max(1, text.count("\n") + (0 if text.endswith("\n") else 1)),
        "content_start_bytes": 0,
        "content_end_bytes": len(content),
        "kind": "global_exact_source_anchor",
        "symbol": "",
        "symbol_kind": "file",
        "retrieval_scores": {},
        "text": text,
    }
    core["observation_id"] = "obs_" + hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return core


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


def _record_line_count(record: dict[str, Any]) -> int:
    if not record.get("text"):
        return 0
    return max(1, int(record.get("end_line", 1)) - int(record.get("start_line", 1)) + 1)


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
