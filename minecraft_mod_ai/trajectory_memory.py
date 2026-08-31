from __future__ import annotations

"""Verifier-qualified trajectory memory for inference-time temporary skills.

JSONL is the durable, auditable source of truth. A rebuildable SQLite/FTS side index
is the canonical query and deduplication hot path, so normal operation never rescans
the full log before each append or retrieval.
"""

import hashlib
import json
import os
import re
import sqlite3
import threading
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .procedural_memory_hierarchy import build_hierarchy, compact_hierarchy
from .procedure_trace import extract_procedure
from .research_perf_common import env_int, table_exists
from .trajectory_record_integrity import (
    derive_levels,
    record_memory_eligible,
    record_strong_skill_eligible,
    validate_trajectory_record,
)
from .trajectory_verification import TRAJECTORY_SCHEMA_VERSION, classify_verification

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:$<>/-]{1,127}|[가-힣]{2,}")
_LOCK = threading.RLock()
_INDEX_LOCK = threading.RLock()
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
_EXECUTION_CONTEXT_ALIASES = {
    "minecraft_version": "minecraft_version",
    "target_version": "minecraft_version",
    "mc_version": "minecraft_version",
    "loader": "loader",
    "loader_id": "loader",
    "mod_loader": "loader",
    "loader_version": "loader_version",
    "mappings": "mappings",
    "mapping": "mappings",
    "mapping_id": "mappings",
    "mappings_version": "mappings_version",
    "mapping_version": "mappings_version",
    "java_version": "java_version",
    "jdk_version": "java_version",
    "platform": "platform",
    "platform_id": "platform",
    "provider": "provider",
    "provider_id": "provider",
    "source_api_family": "source_api_family",
    "model_profile": "model_profile",
    "source_commit": "source_commit",
    "git_commit": "source_commit",
    "revision": "source_commit",
    "manifest_sha256": "manifest_sha256",
    "platform_lock_sha256": "platform_lock_sha256",
    "error_code": "error_code",
    "diagnostic_code": "error_code",
    "failure_code": "error_code",
    "error_type": "error_type",
}
_STRICT_EXECUTION_CONTEXT_KEYS = frozenset(
    {
        "minecraft_version",
        "loader",
        "loader_version",
        "mappings",
        "mappings_version",
        "java_version",
        "source_commit",
        "manifest_sha256",
        "platform_lock_sha256",
        "error_code",
        "error_type",
    }
)
_CONTEXT_VALUE_TYPES = (str, int, float, bool)


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _state_path(
    base: str | Path,
    *parts: str,
    leaf_file: bool = False,
) -> Path:
    """Build a state path without allowing any component to redirect outside root."""

    root = Path(base).expanduser().resolve()
    current = root
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        if current.is_symlink():
            raise RuntimeError("Trajectory memory state must not traverse symbolic links.")
        if current.exists():
            expect_file = bool(leaf_file and is_leaf)
            valid_type = current.is_file() if expect_file else current.is_dir()
            if not valid_type:
                kind = "file" if expect_file else "directory"
                raise RuntimeError(f"Trajectory memory state is not a {kind}: {current}")
        try:
            current.resolve(strict=False).relative_to(root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Trajectory memory state escaped the configured project root."
            ) from exc
    return current


def _memory_dir(base: str | Path) -> Path:
    return _state_path(base, ".minecraft_ai", "trajectory-memory")


def memory_path(base: str | Path) -> Path:
    return _state_path(
        base,
        ".minecraft_ai",
        "trajectory-memory",
        "verified-trajectories.jsonl",
        leaf_file=True,
    )


def remote_cache_path(base: str | Path, task_class: str) -> Path:
    safe = re.sub(r"[^a-z0-9_-]+", "-", task_class.casefold()).strip("-") or "general"
    return _state_path(
        base,
        ".minecraft_ai",
        "trajectory-memory",
        "remote-cache",
        f"{safe}.jsonl",
        leaf_file=True,
    )


def _index_path(base: str | Path) -> Path:
    return _state_path(
        base,
        ".minecraft_ai",
        "trajectory-memory",
        "trajectory-index.sqlite3",
        leaf_file=True,
    )


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


def _normalize_context_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if not isinstance(value, _CONTEXT_VALUE_TYPES):
        return None
    if isinstance(value, str):
        normalized = " ".join(value.split())[:240]
        return normalized if normalized else None
    return value


def _normalize_execution_context_value(
    key: str,
    value: Any,
) -> str | int | float | bool | None:
    normalized = _normalize_context_value(value)
    if key != "java_version" or normalized is None or isinstance(normalized, bool):
        return normalized
    if isinstance(normalized, str) and normalized.isdecimal():
        return int(normalized)
    if isinstance(normalized, float) and normalized.is_integer():
        return int(normalized)
    return normalized


def _collect_execution_context(
    value: Any,
    output: dict[str, Any],
    *,
    depth: int = 0,
) -> None:
    if depth > 6:
        return
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = _EXECUTION_CONTEXT_ALIASES.get(str(raw_key).casefold())
            if key is not None:
                normalized = _normalize_execution_context_value(key, raw_value)
                if normalized is not None:
                    prior = output.get(key)
                    if prior is None:
                        output[key] = normalized
                    elif prior != normalized:
                        digest = hashlib.sha256(
                            json.dumps(
                                [prior, normalized],
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ).encode("utf-8")
                        ).hexdigest()[:16]
                        output[key] = f"__conflict__:{digest}"
            _collect_execution_context(raw_value, output, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value[:64]:
            _collect_execution_context(item, output, depth=depth + 1)


def execution_context_from_values(*values: Any) -> dict[str, Any]:
    """Extract only structured host/runtime identity fields, never model prose."""

    result: dict[str, Any] = {}
    for value in values:
        _collect_execution_context(value, result)
    return dict(sorted(result.items()))


def execution_context_from_messages(
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract reusable-state identity only from structured message payloads."""

    structured: list[Any] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, Mapping):
            structured.append(content)
            continue
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            structured.append(content)
            continue
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text or text[0] not in "[{":
            continue
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, (Mapping, list)):
            structured.append(decoded)
    return execution_context_from_values(*structured)


def _execution_context_compatible(
    row: Mapping[str, Any],
    current_context: Mapping[str, Any] | None,
) -> bool:
    current = execution_context_from_values(current_context or {})
    if not current:
        return True
    stored_raw = row.get("execution_context")
    stored = (
        execution_context_from_values(stored_raw)
        if isinstance(stored_raw, Mapping)
        else {}
    )
    for key, current_value in current.items():
        if key in _STRICT_EXECUTION_CONTEXT_KEYS and key not in stored:
            return False
        if key in stored and stored[key] != current_value:
            return False
    return True


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
    execution_context = execution_context_from_values(task, receipt or {})
    body: dict[str, Any] = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "record_type": "verified_trajectory",
        "storage_format": "jsonl",
        "task_class": task_class,
        "stage": stage,
        "task_shape": shape,
        "outcome": normalized_outcome,
        "verification": verification,
        "procedure": extract_procedure(receipt),
        "verified_facts": _structural_facts(receipt or {}),
        "execution_context": execution_context,
        "error_signature": " ".join(str(error).split())[:1200],
    }
    identity_source = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    body["trajectory_id"] = (
        "sha256:" + hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
    )
    return body


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
    return (
        0.08 * min(level, 5)
        + 0.25 * max(0.0, min(1.0, confidence))
        + (0.08 if reproduced else 0.0)
    )


def _open_index(base: str | Path) -> sqlite3.Connection:
    path = _index_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE IF NOT EXISTS trajectories (
            trajectory_id TEXT PRIMARY KEY,
            task_class TEXT NOT NULL,
            payload TEXT NOT NULL,
            token_text TEXT NOT NULL,
            verification_weight REAL NOT NULL,
            strong_skill INTEGER NOT NULL,
            source_path TEXT NOT NULL,
            source_order INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS trajectories_class_order
            ON trajectories(task_class, source_order DESC);
        CREATE TABLE IF NOT EXISTS trajectory_sources (
            source_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL,
            source_kind TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trajectory_meta (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        );
        """
    )
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS trajectories_fts USING fts5(
                trajectory_id UNINDEXED,
                token_text,
                tokenize = 'unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        pass
    return connection


def _last_source_order(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM trajectory_meta WHERE key = 'last_source_order'"
    ).fetchone()
    if row is not None:
        return int(row[0])
    value = int(
        connection.execute(
            "SELECT COALESCE(MAX(source_order), 0) FROM trajectories"
        ).fetchone()[0]
    )
    connection.execute(
        "INSERT INTO trajectory_meta(key, value) VALUES ('last_source_order', ?)",
        (value,),
    )
    return value


def _set_last_source_order(connection: sqlite3.Connection, value: int) -> None:
    connection.execute(
        """
        INSERT INTO trajectory_meta(key, value) VALUES ('last_source_order', ?)
        ON CONFLICT(key) DO UPDATE SET value = MAX(value, excluded.value)
        """,
        (int(value),),
    )


def _index_row(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    *,
    source: Path,
    order: int,
) -> bool:
    if not record_memory_eligible(row):
        return False
    identity = str(row.get("trajectory_id", ""))
    if not identity:
        return False
    payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
    token_text = " ".join(sorted(_tokens(payload)))
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO trajectories(
            trajectory_id, task_class, payload, token_text,
            verification_weight, strong_skill, source_path, source_order
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            identity,
            str(row.get("task_class", "general")),
            payload,
            token_text,
            float(_verification_weight(row)),
            1 if record_strong_skill_eligible(row) else 0,
            str(source),
            int(order),
        ),
    )
    if cursor.rowcount and table_exists(connection, "trajectories_fts"):
        connection.execute(
            "INSERT INTO trajectories_fts(trajectory_id, token_text) VALUES (?, ?)",
            (identity, token_text),
        )
    return bool(cursor.rowcount)


def _delete_source(connection: sqlite3.Connection, source: Path) -> None:
    identities = [
        str(row[0])
        for row in connection.execute(
            "SELECT trajectory_id FROM trajectories WHERE source_path = ?",
            (str(source),),
        )
    ]
    if identities and table_exists(connection, "trajectories_fts"):
        connection.executemany(
            "DELETE FROM trajectories_fts WHERE trajectory_id = ?",
            [(identity,) for identity in identities],
        )
    connection.execute(
        "DELETE FROM trajectories WHERE source_path = ?",
        (str(source),),
    )
    connection.execute(
        "DELETE FROM trajectory_sources WHERE source_path = ?",
        (str(source),),
    )


def _sync_source(
    connection: sqlite3.Connection,
    source: Path,
    *,
    kind: str,
) -> None:
    source = source.expanduser()
    if source.is_symlink():
        raise RuntimeError("Trajectory source must not be a symbolic link.")
    source = source.resolve()
    previous = connection.execute(
        """
        SELECT size_bytes, modified_ns, byte_offset, source_kind
        FROM trajectory_sources WHERE source_path = ?
        """,
        (str(source),),
    ).fetchone()
    if not source.is_file() or source.is_symlink():
        if previous is not None:
            _delete_source(connection, source)
        return

    stat = source.stat()
    size = int(stat.st_size)
    modified_ns = int(stat.st_mtime_ns)
    if previous is not None and int(previous[0]) == size and int(previous[1]) == modified_ns:
        return

    append_from = 0
    if (
        previous is not None
        and kind == "local"
        and size >= int(previous[2])
        and str(previous[3]) == "local"
    ):
        append_from = int(previous[2])
    else:
        _delete_source(connection, source)

    order = _last_source_order(connection)
    with source.open("rb") as handle:
        if append_from:
            handle.seek(append_from)
        for raw_bytes in handle:
            try:
                value = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping) and record_memory_eligible(value):
                candidate_order = order + 1
                if _index_row(
                    connection,
                    value,
                    source=source,
                    order=candidate_order,
                ):
                    order = candidate_order
        offset = handle.tell()

    _set_last_source_order(connection, order)
    connection.execute(
        """
        INSERT INTO trajectory_sources(
            source_path, size_bytes, modified_ns, byte_offset, source_kind
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            size_bytes = excluded.size_bytes,
            modified_ns = excluded.modified_ns,
            byte_offset = excluded.byte_offset,
            source_kind = excluded.source_kind
        """,
        (str(source), size, modified_ns, int(offset), kind),
    )


def _append_jsonl_line(path: Path, rendered: str) -> None:
    """Append without following a leaf symlink when the platform supports it."""

    if path.is_symlink():
        raise RuntimeError("Trajectory memory file must not be a symbolic link.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as handle:
            fd = -1
            handle.write(rendered)
    finally:
        if fd >= 0:
            os.close(fd)


def _append_jsonl_fallback(base: str | Path, row: Mapping[str, Any]) -> bool:
    path = memory_path(base)
    identity = str(row.get("trajectory_id", ""))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl_line(
        path,
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n",
    )
    return True


def append_trajectory(base: str | Path, row: Mapping[str, Any]) -> bool:
    """Append once using indexed O(log n) dedupe; fall back only if SQLite fails."""

    if not record_memory_eligible(row):
        return False
    identity = str(row.get("trajectory_id", ""))
    if not identity:
        raise ValueError("trajectory_id is required")
    path = memory_path(base)
    with _INDEX_LOCK, _LOCK:
        connection: sqlite3.Connection | None = None
        try:
            connection = _open_index(base)
            _sync_source(connection, path, kind="local")
            if connection.execute(
                "SELECT 1 FROM trajectories WHERE trajectory_id = ? LIMIT 1",
                (identity,),
            ).fetchone() is not None:
                return False

            path.parent.mkdir(parents=True, exist_ok=True)
            rendered = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            _append_jsonl_line(path, rendered)

            order = _last_source_order(connection) + 1
            inserted = _index_row(connection, row, source=path, order=order)
            if inserted:
                _set_last_source_order(connection, order)
            stat = path.stat()
            connection.execute(
                """
                INSERT INTO trajectory_sources(
                    source_path, size_bytes, modified_ns, byte_offset, source_kind
                ) VALUES (?, ?, ?, ?, 'local')
                ON CONFLICT(source_path) DO UPDATE SET
                    size_bytes = excluded.size_bytes,
                    modified_ns = excluded.modified_ns,
                    byte_offset = excluded.byte_offset,
                    source_kind = 'local'
                """,
                (
                    str(path.resolve()),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    int(stat.st_size),
                ),
            )
            connection.commit()
            return inserted
        except sqlite3.DatabaseError:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            return _append_jsonl_fallback(base, row)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass


def _load_rows(path: Path, *, max_rows: int = 1024) -> list[dict[str, Any]]:
    """Bounded JSONL fallback used only when the side index is unavailable."""

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


def _rank_relevant_rows(
    rows: Sequence[Mapping[str, Any]],
    query: str,
    *,
    task_class: str,
    router: Any | None,
    limit: int,
    current_context: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    target = _tokens(query + " " + task_class)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for raw_row in rows:
        row = dict(raw_row)
        if str(row.get("task_class", "")) not in {task_class, "general"}:
            continue
        if not _execution_context_compatible(row, current_context):
            continue
        rendered = json.dumps(row, ensure_ascii=False, sort_keys=True)
        values = _tokens(rendered)
        lexical = (
            len(target & values) / max(1, len(target | values))
            if target and values
            else 0.0
        )
        class_bonus = 0.35 if row.get("task_class") == task_class else 0.0
        success_bonus = 0.08 if record_strong_skill_eligible(row) else 0.0
        scored.append(
            (
                lexical + class_bonus + success_bonus + _verification_weight(row),
                str(row.get("trajectory_id", "")),
                row,
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    shortlist = scored[: min(24, max(limit * 4, limit))]
    if router is not None and query.strip() and shortlist:
        try:
            docs = [
                json.dumps(item[2], ensure_ascii=False, sort_keys=True)
                for item in shortlist
            ]
            reranked = router.rerank(query, docs)
            if len(reranked) == len(shortlist):
                shortlist = [
                    (score + 1.5 * float(rank), identity, row)
                    for (score, identity, row), rank in zip(
                        shortlist, reranked, strict=True
                    )
                ]
                shortlist.sort(key=lambda item: (-item[0], item[1]))
        except Exception:
            pass
    return [row for score, _identity, row in shortlist[:limit] if score > 0.0]


def _fts_query_terms(tokens: Iterable[str]) -> str:
    values: list[str] = []
    for token in sorted(set(tokens))[:32]:
        escaped = str(token).replace('"', '""')
        values.append(f'"{escaped}"')
    return " OR ".join(values)


def _indexed_candidate_rows(
    connection: sqlite3.Connection,
    query: str,
    task_class: str,
    *,
    cap: int,
) -> list[dict[str, Any]]:
    identities: list[str] = []
    terms = _tokens(query + " " + task_class)
    if terms and table_exists(connection, "trajectories_fts"):
        try:
            identities.extend(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT t.trajectory_id
                    FROM trajectories_fts
                    JOIN trajectories AS t
                      ON t.trajectory_id = trajectories_fts.trajectory_id
                    WHERE trajectories_fts MATCH ?
                      AND t.task_class IN (?, 'general')
                    ORDER BY bm25(trajectories_fts), t.source_order DESC
                    LIMIT ?
                    """,
                    (_fts_query_terms(terms), task_class, cap),
                )
            )
        except sqlite3.OperationalError:
            pass
    identities.extend(
        str(row[0])
        for row in connection.execute(
            """
            SELECT trajectory_id FROM trajectories
            WHERE task_class IN (?, 'general')
            ORDER BY source_order DESC
            LIMIT ?
            """,
            (task_class, min(cap, 32)),
        )
    )
    unique = list(dict.fromkeys(identities))[:cap]
    if not unique:
        return []
    placeholders = ",".join("?" for _ in unique)
    rows = connection.execute(
        f"SELECT trajectory_id, payload FROM trajectories "
        f"WHERE trajectory_id IN ({placeholders})",
        unique,
    ).fetchall()
    by_id = {str(row[0]): str(row[1]) for row in rows}
    result: list[dict[str, Any]] = []
    for identity in unique:
        raw = by_id.get(identity)
        if raw is None:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and record_memory_eligible(value):
            result.append(value)
    return result


def _indexed_rows_for_class(
    base: str | Path,
    query: str,
    task_class: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    with _INDEX_LOCK:
        connection = _open_index(base)
        try:
            _sync_source(connection, memory_path(base), kind="local")
            _sync_source(
                connection,
                remote_cache_path(base, task_class),
                kind="remote",
            )
            connection.commit()
            cap = env_int(
                "MMM_TRAJECTORY_CANDIDATE_CAP",
                max(64, limit * 8),
                minimum=max(24, limit),
                maximum=512,
            )
            return _indexed_candidate_rows(
                connection,
                query,
                task_class,
                cap=cap,
            )
        finally:
            connection.close()


def relevant_trajectories(
    base: str | Path,
    query: str,
    *,
    task_class: str,
    router: Any | None = None,
    limit: int = 6,
    current_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        rows = _indexed_rows_for_class(
            base,
            query,
            task_class,
            limit=limit,
        )
    except sqlite3.DatabaseError:
        rows = _load_rows(memory_path(base)) + _load_rows(
            remote_cache_path(base, task_class)
        )
    return _rank_relevant_rows(
        rows,
        query,
        task_class=task_class,
        router=router,
        limit=limit,
        current_context=current_context,
    )


def relevant_trajectories_many(
    base: str | Path,
    query: str,
    *,
    task_classes: Sequence[str],
    router: Any | None = None,
    limit: int = 6,
    current_context: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Rank several classes while syncing local durable memory only once."""

    classes = tuple(
        dict.fromkeys(
            task_class
            for raw in task_classes
            if (task_class := str(raw).strip())
        )
    )
    if not classes:
        return {}

    rows_by_class: dict[str, list[dict[str, Any]]] = {}
    try:
        with _INDEX_LOCK:
            connection = _open_index(base)
            try:
                _sync_source(connection, memory_path(base), kind="local")
                for task_class in classes:
                    _sync_source(
                        connection,
                        remote_cache_path(base, task_class),
                        kind="remote",
                    )
                connection.commit()
                cap = env_int(
                    "MMM_TRAJECTORY_CANDIDATE_CAP",
                    max(64, limit * 8),
                    minimum=max(24, limit),
                    maximum=512,
                )
                for task_class in classes:
                    rows_by_class[task_class] = _indexed_candidate_rows(
                        connection,
                        query,
                        task_class,
                        cap=cap,
                    )
            finally:
                connection.close()
    except sqlite3.DatabaseError:
        local_rows = _load_rows(memory_path(base))
        for task_class in classes:
            rows_by_class[task_class] = [
                *local_rows,
                *_load_rows(remote_cache_path(base, task_class)),
            ]

    return {
        task_class: _rank_relevant_rows(
            rows_by_class.get(task_class, []),
            query,
            task_class=task_class,
            router=router,
            limit=limit,
            current_context=current_context,
        )
        for task_class in classes
    }


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
    qualified = [
        row
        for row in records[:8]
        if record_strong_skill_eligible(row) or _verified_failure(row)
    ]
    if not qualified:
        return None
    success_actions: Counter[str] = Counter()
    failure_signatures: Counter[str] = Counter()
    verifier_facts: Counter[str] = Counter()
    examples: list[str] = []
    source_levels: dict[str, str] = {}
    for row in qualified:
        shape = (
            row.get("task_shape")
            if isinstance(row.get("task_shape"), Mapping)
            else {}
        )
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
            for token in _tokens(
                json.dumps(facts, ensure_ascii=False, sort_keys=True)
            ):
                if token in {
                    "pass",
                    "success",
                    "fail",
                    "jdt_status",
                    "jdt_error_count",
                    "overall_status",
                }:
                    verifier_facts[token] += 1
        identity = str(row.get("trajectory_id", ""))
        examples.append(identity)
        verification = row.get("verification")
        if identity and isinstance(verification, Mapping):
            source_levels[identity] = str(verification.get("level", "L0"))
    hierarchy = compact_hierarchy(build_hierarchy(qualified), max_items=18)
    if not success_actions and not failure_signatures and not hierarchy:
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
            "Treat only L3+ successful trajectories as proven procedure. Verified "
            "failures are negative evidence. Follow ordered workflow/subtask/function "
            "procedure motifs only when their current preconditions still hold. Current "
            "exact evidence, compiler diagnostics, executable tests and acceptance "
            "contracts remain authoritative."
        ),
    }


append_trajectory._mmm_research_memory_performance_v1 = True  # type: ignore[attr-defined]
relevant_trajectories._mmm_research_memory_performance_v1 = True  # type: ignore[attr-defined]


__all__ = [
    "append_trajectory",
    "build_work_trajectory",
    "execution_context_from_messages",
    "execution_context_from_values",
    "memory_path",
    "relevant_trajectories",
    "relevant_trajectories_many",
    "remote_cache_path",
    "synthesize_temporary_skill",
    "task_class_for_stage",
]
