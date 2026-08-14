from __future__ import annotations

"""Verifier-qualified trajectory memory for inference-time temporary skills.

Only structural task/action/verifier facts are stored. Model completion claims,
source bodies and arbitrary tool payloads are not reusable memory evidence.
"""

import hashlib
import json
import re
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from .procedural_memory_hierarchy import build_hierarchy, compact_hierarchy
from .trajectory_record_integrity import (
    derive_levels,
    record_memory_eligible,
    record_strong_skill_eligible,
    validate_trajectory_record,
)
from .trajectory_verification import TRAJECTORY_SCHEMA_VERSION, classify_verification

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:$<>/-]{1,127}|[가-힣]{2,}")
_LOCK = threading.RLock()
_ALLOWED_FACT_KEYS = frozenset(
    {
        "status",
        "stage",
        "kind",
        "tool",
        "operation",
        "action",
        "code",
        "severity",
        "jdt_status",
        "jdt_error_count",
        "build_status",
        "result_count",
        "coverage_score",
        "relevance_score",
        "relation_expansions",
        "candidate_count",
        "winner_score",
        "overall_status",
        "assertion_count",
        "interaction_count",
        "active_build_status",
    }
)


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _memory_dir(base: str | Path) -> Path:
    root = Path(base).expanduser().resolve()
    return root / ".minecraft_ai" / "trajectory-memory"


def memory_path(base: str | Path) -> Path:
    return _memory_dir(base) / "verified-trajectories.jsonl"


def remote_cache_path(base: str | Path, task_class: str) -> Path:
    safe = re.sub(r"[^a-z0-9_-]+", "-", task_class.casefold()).strip("-") or "general"
    return _memory_dir(base) / "remote-cache" / f"{safe}.jsonl"


def task_class_for_stage(stage: str) -> str:
    value = stage.casefold()
    if "repair" in value:
        return "repair"
    if "generate" in value:
        return "generation"
    if "build" in value or "gradle" in value:
        return "build"
    if "runtime" in value or "playtest" in value:
        return "runtime"
    if "quality" in value or "validate" in value:
        return "quality"
    if "research" in value:
        return "research"
    if "plan" in value:
        return "planning"
    if "package" in value or "release" in value:
        return "release"
    return "general"


def _structural_facts(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 5 or not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        lowered = key.casefold()
        if lowered in _ALLOWED_FACT_KEYS:
            if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
                result[key] = raw_value
            continue
        if isinstance(raw_value, Mapping):
            nested = _structural_facts(raw_value, depth=depth + 1)
            if nested:
                result[key] = nested
    return result


def _task_shape(task: Mapping[str, Any]) -> dict[str, Any]:
    payload = task.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    members = payload.get("members")
    member_ids: list[str] = []
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        for item in members[:32]:
            if not isinstance(item, Mapping):
                continue
            for key in ("module_id", "asset_id", "sound_id", "id"):
                value = str(item.get(key, "")).strip()
                if value:
                    member_ids.append(value)
                    break
    return {
        "node_id": str(task.get("node_id", ""))[:160],
        "stage": str(task.get("stage", ""))[:160],
        "kind": str(payload.get("kind", ""))[:160],
        "generation_stage": str(payload.get("generation_stage", ""))[:160],
        "member_ids": sorted(set(member_ids))[:32],
    }


def build_work_trajectory(
    task: Mapping[str, Any],
    *,
    outcome: str,
    receipt: Mapping[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    shape = _task_shape(task)
    stage = shape["stage"]
    task_class = task_class_for_stage(stage)
    normalized_outcome = "SUCCESS" if outcome.upper() == "SUCCESS" else "FAIL"
    verification = classify_verification(
        task_class=task_class,
        outcome=normalized_outcome,
        receipt=receipt,
        error=error,
    )
    body: dict[str, Any] = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "record_type": "verified_trajectory",
        "storage_format": "jsonl",
        "task_class": task_class,
        "stage": stage,
        "task_shape": shape,
        "outcome": normalized_outcome,
        "verification": verification,
        "verified_facts": _structural_facts(receipt or {}),
        "error_signature": " ".join(str(error).split())[:1200],
    }
    identity_source = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    body["trajectory_id"] = "sha256:" + hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
    return body


def append_trajectory(base: str | Path, row: Mapping[str, Any]) -> bool:
    if not record_memory_eligible(row):
        return False
    path = memory_path(base)
    identity = str(row.get("trajectory_id", ""))
    if not identity:
        raise ValueError("trajectory_id is required")
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        recent: deque[str] = deque(maxlen=512)
        if path.is_file() and not path.is_symlink():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, Mapping):
                            recent.append(str(value.get("trajectory_id", "")))
            except OSError:
                return False
        if identity in recent:
            return False
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return True


def _load_rows(path: Path, *, max_rows: int = 1024) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and record_memory_eligible(value):
                    rows.append(value)
    except OSError:
        return []
    return list(rows)


def _verification_weight(row: Mapping[str, Any]) -> float:
    if not validate_trajectory_record(row):
        return 0.0
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        return 0.0
    try:
        level = int(verification.get("level_index", 0) or 0)
    except (TypeError, ValueError):
        level = 0
    try:
        confidence = float(verification.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reproduced = verification.get("reproduced") is True
    return 0.08 * min(level, 5) + 0.25 * max(0.0, min(1.0, confidence)) + (0.08 if reproduced else 0.0)


def relevant_trajectories(
    base: str | Path,
    query: str,
    *,
    task_class: str,
    router: Any | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    rows = _load_rows(memory_path(base)) + _load_rows(remote_cache_path(base, task_class))
    target = _tokens(query + " " + task_class)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        if str(row.get("task_class", "")) not in {task_class, "general"}:
            continue
        rendered = json.dumps(row, ensure_ascii=False, sort_keys=True)
        values = _tokens(rendered)
        lexical = len(target & values) / max(1, len(target | values)) if target and values else 0.0
        class_bonus = 0.35 if row.get("task_class") == task_class else 0.0
        success_bonus = 0.08 if record_strong_skill_eligible(row) else 0.0
        scored.append((lexical + class_bonus + success_bonus + _verification_weight(row), str(row.get("trajectory_id", "")), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    shortlist = scored[: min(24, max(limit * 4, limit))]
    if router is not None and query.strip() and shortlist:
        try:
            docs = [json.dumps(item[2], ensure_ascii=False, sort_keys=True) for item in shortlist]
            reranked = router.rerank(query, docs)
            if len(reranked) == len(shortlist):
                shortlist = [
                    (score + 1.5 * float(rank), identity, row)
                    for (score, identity, row), rank in zip(shortlist, reranked, strict=True)
                ]
                shortlist.sort(key=lambda item: (-item[0], item[1]))
        except Exception:
            pass
    return [row for score, _identity, row in shortlist[:limit] if score > 0.0]


def _verified_failure(row: Mapping[str, Any]) -> bool:
    derived = derive_levels(row)
    return bool(
        derived
        and validate_trajectory_record(row)
        and derived.get("verified_failure") is True
    )


def synthesize_temporary_skill(
    query: str,
    records: Sequence[Mapping[str, Any]],
    *,
    task_class: str,
) -> dict[str, Any] | None:
    qualified = [row for row in records[:8] if record_strong_skill_eligible(row) or _verified_failure(row)]
    if not qualified:
        return None
    success_actions: Counter[str] = Counter()
    failure_signatures: Counter[str] = Counter()
    verifier_facts: Counter[str] = Counter()
    examples: list[str] = []
    source_levels: dict[str, str] = {}
    for row in qualified:
        shape = row.get("task_shape") if isinstance(row.get("task_shape"), Mapping) else {}
        stage = str(shape.get("stage", ""))
        kind = str(shape.get("kind", ""))
        label = ":".join(part for part in (stage, kind) if part)
        if label:
            if record_strong_skill_eligible(row):
                success_actions[label] += 1
            elif _verified_failure(row):
                failure_signatures[label] += 1
        error = str(row.get("error_signature", "")).strip()
        if error and _verified_failure(row):
            failure_signatures[error[:240]] += 1
        facts = row.get("verified_facts")
        if isinstance(facts, Mapping):
            for token in _tokens(json.dumps(facts, ensure_ascii=False, sort_keys=True)):
                if token in {"pass", "success", "fail", "jdt_status", "jdt_error_count", "overall_status"}:
                    verifier_facts[token] += 1
        identity = str(row.get("trajectory_id", ""))
        examples.append(identity)
        verification = row.get("verification")
        if identity and isinstance(verification, Mapping):
            source_levels[identity] = str(verification.get("level", "L0"))
    hierarchy = compact_hierarchy(build_hierarchy(qualified), max_items=18)
    if not success_actions and not failure_signatures:
        return None
    return {
        "schema_version": "mmm/temporary-skill-v3",
        "ephemeral": True,
        "task_class": task_class,
        "current_query_terms": sorted(_tokens(query))[:48],
        "procedural_hierarchy": hierarchy,
        "proven_patterns": [item for item, _count in success_actions.most_common(6)],
        "avoid_patterns": [item for item, _count in failure_signatures.most_common(6)],
        "verifier_hints": [item for item, _count in verifier_facts.most_common(8)],
        "source_trajectory_ids": examples[:8],
        "source_verification_levels": source_levels,
        "rule": (
            "Treat only L3+ successful trajectories as proven procedure. Verified failures are negative evidence. "
            "Current exact evidence, compiler diagnostics, executable tests and acceptance contracts remain authoritative."
        ),
    }


__all__ = [
    "append_trajectory",
    "build_work_trajectory",
    "memory_path",
    "relevant_trajectories",
    "remote_cache_path",
    "synthesize_temporary_skill",
    "task_class_for_stage",
]
