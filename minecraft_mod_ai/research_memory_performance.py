from __future__ import annotations

"""Indexed verifier-qualified trajectory memory.

JSONL remains the durable/auditable log. A rebuildable SQLite+FTS side index makes
append dedupe O(log n), indexes task class and lexical terms, and incrementally tails
local append-only history instead of rescanning the whole log before every generation.
"""

import json
import sqlite3
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .research_perf_common import env_int, table_exists

_MARKER = "_mmm_research_memory_performance_v1"
_TRAJECTORY_LOCK = threading.RLock()

def _trajectory_db_path(tm: Any, base: str | Path) -> Path:
    return tm._memory_dir(base) / "trajectory-index.sqlite3"


def _open_trajectory_db(tm: Any, base: str | Path) -> sqlite3.Connection:
    path = _trajectory_db_path(tm, base)
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


def _trajectory_insert(tm: Any, connection: sqlite3.Connection, row: Mapping[str, Any], *, source: Path, order: int) -> bool:
    if not tm.record_memory_eligible(row):
        return False
    identity = str(row.get("trajectory_id", ""))
    if not identity:
        return False
    payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
    token_text = " ".join(sorted(tm._tokens(payload)))
    weight = float(tm._verification_weight(row))
    strong = 1 if tm.record_strong_skill_eligible(row) else 0
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
            weight,
            strong,
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


def _delete_trajectory_source(connection: sqlite3.Connection, source: Path) -> None:
    ids = [
        str(row[0])
        for row in connection.execute(
            "SELECT trajectory_id FROM trajectories WHERE source_path = ?",
            (str(source),),
        )
    ]
    if ids and table_exists(connection, "trajectories_fts"):
        connection.executemany(
            "DELETE FROM trajectories_fts WHERE trajectory_id = ?",
            [(identity,) for identity in ids],
        )
    connection.execute("DELETE FROM trajectories WHERE source_path = ?", (str(source),))
    connection.execute("DELETE FROM trajectory_sources WHERE source_path = ?", (str(source),))


def _sync_trajectory_source(tm: Any, connection: sqlite3.Connection, source: Path, *, kind: str) -> None:
    source = source.expanduser().resolve()
    previous = connection.execute(
        "SELECT size_bytes, modified_ns, byte_offset, source_kind FROM trajectory_sources WHERE source_path = ?",
        (str(source),),
    ).fetchone()
    if not source.is_file() or source.is_symlink():
        if previous is not None:
            _delete_trajectory_source(connection, source)
        return
    stat = source.stat()
    size = int(stat.st_size)
    mtime = int(stat.st_mtime_ns)
    if previous is not None and int(previous[0]) == size and int(previous[1]) == mtime:
        return

    # Local storage is append-only by contract, so only new bytes are parsed. Remote
    # caches may be atomically replaced by hydration and are therefore reindexed as a
    # source-local unit whenever they change.
    append_from = 0
    if (
        previous is not None
        and kind == "local"
        and size >= int(previous[2])
        and str(previous[3]) == "local"
    ):
        append_from = int(previous[2])
    else:
        _delete_trajectory_source(connection, source)

    order = _last_source_order(connection)
    with source.open("rb") as handle:
        if append_from:
            handle.seek(append_from)
        for raw_bytes in handle:
            try:
                raw = raw_bytes.decode("utf-8", errors="replace")
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping) and tm.record_memory_eligible(value):
                candidate_order = order + 1
                if _trajectory_insert(
                    tm, connection, value, source=source, order=candidate_order
                ):
                    order = candidate_order
        offset = handle.tell()
    _set_last_source_order(connection, order)
    connection.execute(
        """
        INSERT INTO trajectory_sources(source_path, size_bytes, modified_ns, byte_offset, source_kind)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            size_bytes = excluded.size_bytes,
            modified_ns = excluded.modified_ns,
            byte_offset = excluded.byte_offset,
            source_kind = excluded.source_kind
        """,
        (str(source), size, mtime, int(offset), kind),
    )


def _indexed_append_factory(tm: Any, original: Callable[..., bool]):
    @wraps(original)
    def append(base: str | Path, row: Mapping[str, Any]) -> bool:
        if not tm.record_memory_eligible(row):
            return False
        identity = str(row.get("trajectory_id", ""))
        if not identity:
            raise ValueError("trajectory_id is required")
        path = tm.memory_path(base)
        with _TRAJECTORY_LOCK, tm._LOCK:
            connection = _open_trajectory_db(tm, base)
            try:
                _sync_trajectory_source(tm, connection, path, kind="local")
                exists = connection.execute(
                    "SELECT 1 FROM trajectories WHERE trajectory_id = ? LIMIT 1",
                    (identity,),
                ).fetchone()
                if exists is not None:
                    return False
                path.parent.mkdir(parents=True, exist_ok=True)
                rendered = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(rendered)
                order = _last_source_order(connection) + 1
                inserted = _trajectory_insert(tm, connection, row, source=path, order=order)
                if inserted:
                    _set_last_source_order(connection, order)
                stat = path.stat()
                connection.execute(
                    """
                    INSERT INTO trajectory_sources(source_path, size_bytes, modified_ns, byte_offset, source_kind)
                    VALUES (?, ?, ?, ?, 'local')
                    ON CONFLICT(source_path) DO UPDATE SET
                        size_bytes = excluded.size_bytes,
                        modified_ns = excluded.modified_ns,
                        byte_offset = excluded.byte_offset,
                        source_kind = 'local'
                    """,
                    (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_size)),
                )
                connection.commit()
                return inserted
            except sqlite3.DatabaseError:
                connection.close()
                return original(base, row)
            finally:
                try:
                    connection.close()
                except Exception:
                    pass

    setattr(append, _MARKER, True)
    return append


def _fts_query_terms(tokens: Iterable[str]) -> str:
    values = []
    for token in sorted(set(tokens))[:32]:
        escaped = str(token).replace('"', '""')
        values.append(f'"{escaped}"')
    return " OR ".join(values)


def _trajectory_candidate_rows(tm: Any, connection: sqlite3.Connection, query: str, task_class: str, *, cap: int) -> list[dict[str, Any]]:
    identities: list[str] = []
    terms = tm._tokens(query + " " + task_class)
    if terms and table_exists(connection, "trajectories_fts"):
        try:
            match = _fts_query_terms(terms)
            identities.extend(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT t.trajectory_id
                    FROM trajectories_fts
                    JOIN trajectories AS t ON t.trajectory_id = trajectories_fts.trajectory_id
                    WHERE trajectories_fts MATCH ?
                      AND t.task_class IN (?, 'general')
                    ORDER BY bm25(trajectories_fts), t.source_order DESC
                    LIMIT ?
                    """,
                    (match, task_class, cap),
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
        f"SELECT trajectory_id, payload FROM trajectories WHERE trajectory_id IN ({placeholders})",
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
        if isinstance(value, dict) and tm.record_memory_eligible(value):
            result.append(value)
    return result


def _indexed_relevant_factory(tm: Any, original: Callable[..., list[dict[str, Any]]]):
    @wraps(original)
    def relevant(
        base: str | Path,
        query: str,
        *,
        task_class: str,
        router: Any | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        with _TRAJECTORY_LOCK:
            connection = _open_trajectory_db(tm, base)
            try:
                _sync_trajectory_source(tm, connection, tm.memory_path(base), kind="local")
                _sync_trajectory_source(
                    tm,
                    connection,
                    tm.remote_cache_path(base, task_class),
                    kind="remote",
                )
                connection.commit()
                cap = env_int(
                    "MMM_TRAJECTORY_CANDIDATE_CAP",
                    max(64, limit * 8),
                    minimum=max(24, limit),
                    maximum=512,
                )
                rows = _trajectory_candidate_rows(tm, connection, query, task_class, cap=cap)
            except sqlite3.DatabaseError:
                connection.close()
                return original(base, query, task_class=task_class, router=router, limit=limit)
            finally:
                try:
                    connection.close()
                except Exception:
                    pass

        target = tm._tokens(query + " " + task_class)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in rows:
            if str(row.get("task_class", "")) not in {task_class, "general"}:
                continue
            rendered = json.dumps(row, ensure_ascii=False, sort_keys=True)
            values = tm._tokens(rendered)
            lexical = len(target & values) / max(1, len(target | values)) if target and values else 0.0
            class_bonus = 0.35 if row.get("task_class") == task_class else 0.0
            success_bonus = 0.08 if tm.record_strong_skill_eligible(row) else 0.0
            scored.append(
                (
                    lexical + class_bonus + success_bonus + tm._verification_weight(row),
                    str(row.get("trajectory_id", "")),
                    row,
                )
            )
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

    setattr(relevant, _MARKER, True)
    return relevant



def harden(trajectory_memory_module: Any) -> tuple[Callable[..., Any], Callable[..., Any]]:
    current_append = trajectory_memory_module.append_trajectory
    current_relevant = trajectory_memory_module.relevant_trajectories
    if not getattr(current_append, _MARKER, False):
        indexed_append = _indexed_append_factory(trajectory_memory_module, current_append)
        trajectory_memory_module.append_trajectory = indexed_append
    else:
        indexed_append = current_append
    if not getattr(current_relevant, _MARKER, False):
        indexed_relevant = _indexed_relevant_factory(trajectory_memory_module, current_relevant)
        trajectory_memory_module.relevant_trajectories = indexed_relevant
    else:
        indexed_relevant = current_relevant
    return indexed_append, indexed_relevant


__all__ = ["harden"]
