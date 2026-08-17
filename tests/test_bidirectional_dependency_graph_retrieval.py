from __future__ import annotations

import sqlite3
from pathlib import Path

from minecraft_mod_ai import rag_index
from minecraft_mod_ai.rag_index import ProjectRAGIndex, RAGHit
from minecraft_mod_ai.small_model_retrieval_efficiency_contract import install


def test_dependency_graph_can_retrieve_incoming_callers(tmp_path: Path) -> None:
    install()
    project = tmp_path / "project"
    source = project / "src/main/java/demo"
    source.mkdir(parents=True)
    target = source / "Target.java"
    caller = source / "Caller.java"
    target.write_text("package demo; final class Target {}\n", encoding="utf-8")
    caller.write_text(
        "package demo; final class Caller { Target target; }\n",
        encoding="utf-8",
    )
    metadata = {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mapping_namespace": "yarn",
        "java_version": "17",
        "license": "project-local",
        "source_commit": "snapshot-a",
        "relations": [
            {
                "source": str(caller.resolve()),
                "target": str(target.resolve()),
                "kind": "java_type",
            }
        ],
    }
    index_path = tmp_path / "project-rag.sqlite"
    ProjectRAGIndex(index_path).build(
        [project],
        metadata=metadata,
        router=None,
        semantic=False,
    )

    with sqlite3.connect(str(index_path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT chunk_id, source_path, text, start_line, end_line,
                   sha256, embedding
            FROM chunks
            WHERE source_path = ?
            ORDER BY start_line
            LIMIT 1
            """,
            (str(target.resolve()),),
        ).fetchone()
        assert row is not None
        chunk = rag_index._chunk_from_row(row, metadata)
        seed = RAGHit(
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            score=1.0,
            lexical_score=1.0,
            semantic_score=0.0,
            reranker_score=0.0,
            text=chunk.text,
            metadata=metadata,
            relation_score=0.0,
        )
        related = rag_index._expand_sqlite_relationships(
            connection,
            "Target dependency callers",
            [seed],
            metadata=metadata,
            budget=4,
        )

    caller_hit = next(
        hit for hit in related if hit.source_path == str(caller.resolve())
    )
    relation = caller_hit.metadata["_rag_relation"]
    assert relation["kind"] == "java_type"
    assert relation["direction"] == "incoming"
    assert getattr(
        rag_index._expand_sqlite_relationships,
        "_mmm_bidirectional_dependency_graph",
        False,
    )
