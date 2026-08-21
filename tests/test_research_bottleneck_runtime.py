from __future__ import annotations

import json
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from minecraft_mod_ai import centroid_vector_rag, rag_index, trajectory_memory
from minecraft_mod_ai import research_cpu_retrieval_performance as cpu_perf
from minecraft_mod_ai import research_gradle_performance as gradle_perf
from minecraft_mod_ai import research_rag_performance as rag_perf


def _metadata(source_commit: str = "sha256:test") -> dict[str, str]:
    return {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mapping_namespace": "yarn",
        "java_version": "17",
        "license": "project-local",
        "source_commit": source_commit,
    }


def test_research_runtime_is_installed_by_package_bootstrap() -> None:
    assert hasattr(rag_index.ProjectRAGIndex, "_full_rebuild")
    assert hasattr(rag_index, "_full_sqlite_search_pass")
    assert hasattr(centroid_vector_rag, "_full_direct_centroid_vector_search")
    rag_perf.harden(rag_index, centroid_vector_rag)
    assert getattr(
        trajectory_memory.append_trajectory,
        "_mmm_research_memory_performance_v1",
        False,
    )
    assert getattr(
        trajectory_memory.relevant_trajectories,
        "_mmm_research_memory_performance_v1",
        False,
    )


def test_project_rag_reuses_unchanged_content_and_reindexes_only_changed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first_file = root / "A.java"
    second_file = root / "B.java"
    first_file.write_text("public final class A {}\n", encoding="utf-8")
    second_file.write_text("public final class B {}\n", encoding="utf-8")
    index = rag_index.ProjectRAGIndex(tmp_path / "project-index.sqlite")

    initial = index.build([root], metadata=_metadata("sha256:one"), semantic=False)
    assert initial["files_indexed"] == 2

    hash_calls: list[Path] = []
    original_hash = rag_perf._file_sha256

    def tracked_hash(path: Path) -> str:
        hash_calls.append(path)
        return original_hash(path)

    monkeypatch.setattr(rag_perf, "_file_sha256", tracked_hash)
    reused = index.build([root], metadata=_metadata("sha256:two"), semantic=False)
    assert reused["incremental"] is True
    assert reused["changed_files"] == 0
    assert reused["removed_files"] == 0
    assert reused["reused_files"] == 2
    assert hash_calls == []

    # A content mutation invalidates only A.java; B.java remains indexed in place.
    first_file.write_text("public final class A { int value = 1; }\n", encoding="utf-8")
    changed = index.build([root], metadata=_metadata("sha256:three"), semantic=False)
    assert changed["incremental"] is True
    assert changed["changed_files"] == 1
    assert changed["removed_files"] == 0
    assert changed["reused_files"] == 1
    assert hash_calls == [first_file]


def test_multi_probe_lsh_bounds_dense_candidates_and_keeps_exact_neighbor(
    tmp_path: Path,
) -> None:
    target = tmp_path / "semantic.sqlite"
    connection = sqlite3.connect(str(target))
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE index_meta(key TEXT PRIMARY KEY, value TEXT)")
    connection.execute(
        "INSERT INTO index_meta(key, value) VALUES ('chunks_indexed', '1000')"
    )
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
    width = 16
    for index in range(1000):
        vector = [0.0] * width
        vector[index % width] = 1.0
        connection.execute(
            """
            INSERT INTO chunks(chunk_id, source_path, text, start_line, end_line, sha256, embedding)
            VALUES (?, ?, ?, 1, 1, 'sha256:x', ?)
            """,
            (
                str(index),
                f"src/{index}.java",
                f"chunk {index}",
                json.dumps(vector),
            ),
        )
    connection.commit()
    # Candidate lookup is query-only now. Reconciliation and the readiness marker
    # belong to build/update setup, so establish that state before measuring lookup.
    rag_perf._ensure_semantic_lsh(connection)

    query = [1.0] + [0.0] * (width - 1)
    rows = rag_perf._lsh_candidate_rows(connection, query, target=16, cap=128)
    connection.close()

    assert 0 < len(rows) <= 128
    assert len(rows) < 1000
    assert any(str(row["chunk_id"]) == "0" for row in rows)

    connection = sqlite3.connect(str(target))
    trace: list[str] = []
    connection.set_trace_callback(trace.append)
    rag_perf._ensure_semantic_lsh(connection)
    connection.set_trace_callback(None)
    connection.close()
    assert not any(statement.startswith("DELETE FROM mmm_semantic_lsh") for statement in trace)


def test_trajectory_memory_uses_indexed_append_and_query_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trajectory_memory, "record_memory_eligible", lambda row: True)
    monkeypatch.setattr(
        trajectory_memory,
        "record_strong_skill_eligible",
        lambda row: row.get("outcome") == "SUCCESS",
    )
    monkeypatch.setattr(trajectory_memory, "_verification_weight", lambda row: 0.5)

    for index in range(80):
        assert trajectory_memory.append_trajectory(
            tmp_path,
            {
                "trajectory_id": f"trajectory-{index}",
                "task_class": "repair" if index % 2 == 0 else "general",
                "outcome": "SUCCESS",
                "message": f"fabric repair symbol {index}",
            },
        )
    assert not trajectory_memory.append_trajectory(
        tmp_path,
        {
            "trajectory_id": "trajectory-2",
            "task_class": "repair",
            "outcome": "SUCCESS",
            "message": "duplicate",
        },
    )

    rows = trajectory_memory.relevant_trajectories(
        tmp_path,
        "fabric repair symbol",
        task_class="repair",
        router=None,
        limit=6,
    )
    db = (
        tmp_path
        / ".minecraft_ai"
        / "trajectory-memory"
        / "trajectory-index.sqlite3"
    )
    assert db.is_file()
    connection = sqlite3.connect(str(db))
    last_order = connection.execute(
        "SELECT value FROM trajectory_meta WHERE key = 'last_source_order'"
    ).fetchone()
    connection.close()
    assert last_order == (80,)
    assert len(rows) == 6
    assert all(row["task_class"] in {"repair", "general"} for row in rows)


def test_cpu_coalescer_merges_concurrent_requests_into_one_batch() -> None:
    observed: list[int] = []

    def process(group: list[cpu_perf._BatchRequest]) -> None:
        observed.append(len(group))
        for item in group:
            item.future.set_result(int(item.payload) * 2)

    batcher = cpu_perf._CoalescingBatcher(process, name="mmm-test-coalescer")

    def submit(value: int) -> int:
        return int(
            batcher.submit(
                cpu_perf._BatchRequest(
                    key=("same-model",),
                    adapter=None,
                    payload=value,
                    future=Future(),
                )
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(submit, range(8)))

    assert values == [value * 2 for value in range(8)]
    assert sum(observed) == 8
    assert max(observed) > 1


def test_gradle_hot_path_uses_daemon_build_cache_and_optional_configuration_cache(
) -> None:
    values = gradle_perf._optimized_gradle_arguments(
        ("--no-daemon", "build", "--build-cache", "--stacktrace"),
        enable_configuration_cache=True,
    )
    assert "--no-daemon" not in values
    assert "--daemon" in values
    assert values.count("--build-cache") == 1
    assert "--configuration-cache" in values
