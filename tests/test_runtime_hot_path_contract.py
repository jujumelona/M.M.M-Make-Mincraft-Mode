from __future__ import annotations

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from minecraft_mod_ai import research_rag_performance as rag_perf
from minecraft_mod_ai.external_mcp_router import ExternalMCPRouter


def test_external_mcp_provider_calls_are_not_globally_serialized(monkeypatch) -> None:
    router = ExternalMCPRouter(timeout_seconds=2.0)
    rendezvous = threading.Barrier(2)

    async def fake_call_provider_async(server_name, entry, *, tool, arguments):
        await asyncio.to_thread(rendezvous.wait, 1.0)
        return {
            "server_info": {"name": server_name},
            "result": {"tool": tool, "arguments": dict(arguments)},
        }

    monkeypatch.setattr(router, "_call_provider_async", fake_call_provider_async)

    def call(server_name: str) -> dict:
        return router._call_provider(
            server_name,
            {},
            tool="lookup",
            arguments={"query": server_name},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(call, ("provider-a", "provider-b")))

    assert {row["server_info"]["name"] for row in results} == {
        "provider-a",
        "provider-b",
    }


def test_ready_lsh_query_does_not_reconcile_side_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "semantic.sqlite"
    connection = sqlite3.connect(str(target))
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE chunks(
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE,
            source_path TEXT,
            text TEXT,
            start_line INTEGER,
            end_line INTEGER,
            sha256 TEXT,
            embedding TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO chunks(
            chunk_id, source_path, text, start_line, end_line, sha256, embedding
        ) VALUES ('exact', 'src/A.java', 'exact', 1, 1, 'sha256:x', '[1.0, 0.0]')
        """
    )
    connection.commit()
    rag_perf._ensure_semantic_lsh(connection)

    def forbidden_reconcile(_connection):
        raise AssertionError("query path must not reconcile semantic LSH")

    monkeypatch.setattr(rag_perf, "_ensure_semantic_lsh", forbidden_reconcile)
    rows = rag_perf._lsh_candidate_rows(connection, [1.0, 0.0], target=1, cap=8)
    connection.close()

    assert [str(row["chunk_id"]) for row in rows] == ["exact"]


def test_unready_lsh_fails_before_querying_candidates(tmp_path: Path) -> None:
    target = tmp_path / "semantic.sqlite"
    connection = sqlite3.connect(str(target))
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE chunks(
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE,
            source_path TEXT,
            text TEXT,
            start_line INTEGER,
            end_line INTEGER,
            sha256 TEXT,
            embedding TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE mmm_semantic_lsh(
            chunk_id TEXT PRIMARY KEY,
            sig_a INTEGER NOT NULL,
            sig_b INTEGER NOT NULL
        )
        """
    )
    connection.commit()

    try:
        rag_perf._lsh_candidate_rows(connection, [1.0, 0.0], target=1, cap=8)
    except RuntimeError as exc:
        assert "not ready" in str(exc)
    else:  # pragma: no cover - contract regression
        raise AssertionError("unready semantic LSH must fail closed")
    finally:
        connection.close()
