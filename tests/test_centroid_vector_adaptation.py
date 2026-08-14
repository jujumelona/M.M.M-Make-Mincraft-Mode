from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from minecraft_mod_ai.centroid_vector_rag import direct_centroid_vector_search
from minecraft_mod_ai.retrieval_adaptation import adapt_query_vector


class _EmbeddingRouter:
    def embed(self, text: str):
        if text == "original query":
            return [1.0, 0.0]
        if text in {"first local hit", "second local hit"}:
            return [0.0, 1.0]
        raise AssertionError(text)

    def rerank(self, _query: str, documents: list[str]):
        return [0.0 for _ in documents]


def _index(path: Path, q1: list[float]) -> None:
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            CREATE TABLE index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                source_path TEXT NOT NULL,
                normalized_path TEXT NOT NULL,
                text TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                embedding TEXT
            );
            """
        )
        connection.executemany(
            "INSERT INTO index_meta(key, value) VALUES (?, ?)",
            [
                ("semantic_embeddings", "1"),
                ("metadata", json.dumps({"minecraft_version": "1.20.1"})),
            ],
        )
        connection.executemany(
            """
            INSERT INTO chunks(
                chunk_id, source_path, normalized_path, text,
                start_line, end_line, sha256, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "q1-match",
                    "/project/Q1.java",
                    "/project/Q1.java",
                    "original query implementation q1 match",
                    1,
                    5,
                    "a" * 64,
                    json.dumps(q1),
                ),
                (
                    "centroid-only",
                    "/project/Centroid.java",
                    "/project/Centroid.java",
                    "local neighbor only",
                    1,
                    5,
                    "b" * 64,
                    json.dumps([0.0, 1.0]),
                ),
                (
                    "opposite",
                    "/project/Opposite.java",
                    "/project/Opposite.java",
                    "unrelated opposite vector",
                    1,
                    5,
                    "c" * 64,
                    json.dumps([-1.0, 0.0]),
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_q0_topk_centroid_q1_is_used_directly_for_vector_retrieval(tmp_path: Path) -> None:
    router = _EmbeddingRouter()
    q1 = adapt_query_vector(
        router,
        "original query",
        ["first local hit", "second local hit"],
    )
    assert q1
    assert q1 != [1.0, 0.0]
    assert q1 != [0.0, 1.0]

    index = tmp_path / "project-index.sqlite3"
    _index(index, q1)
    result = direct_centroid_vector_search(
        index,
        query="original query",
        q1_vector=q1,
        router=router,
        limit=2,
        required_metadata={"minecraft_version": "1.20.1"},
    )
    assert result is not None
    assert result["centroid_vector_direct"] is True
    assert result["retrieval_mode"] == "centroid-q1-vector+rerank"
    assert result["receipt"]["route"] == "centroid_vector"
    assert result["receipt"]["adaptation"] == "q0_topk_centroid_q1_direct_vector"
    assert result["hits"][0]["chunk_id"] == "q1-match"


def test_direct_q1_search_fails_closed_when_semantic_vectors_are_not_declared(tmp_path: Path) -> None:
    path = tmp_path / "no-semantic.sqlite3"
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            CREATE TABLE index_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                source_path TEXT NOT NULL,
                normalized_path TEXT NOT NULL,
                text TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                embedding TEXT
            );
            INSERT INTO index_meta(key, value) VALUES ('semantic_embeddings', '0');
            INSERT INTO index_meta(key, value) VALUES ('metadata', '{}');
            """
        )
        connection.commit()
    finally:
        connection.close()
    assert direct_centroid_vector_search(
        path,
        query="original query",
        q1_vector=[0.8, 0.6],
        router=_EmbeddingRouter(),
        limit=2,
    ) is None
