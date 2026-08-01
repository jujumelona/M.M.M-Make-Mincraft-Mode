import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from minecraft_mod_ai.rag_index import ProjectRAGIndex


def _metadata() -> dict:
    return {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mapping_namespace": "yarn",
        "java_version": 17,
        "license": "Apache-2.0",
        "source_commit": "abc123",
    }


def test_lexical_project_rag_is_version_and_license_aware(tmp_path: Path) -> None:
    source = tmp_path / "project" / "src" / "Example.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "Registry.register(Registries.ITEM, new Identifier(MOD_ID, \"crystal\"), item);",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    result = ProjectRAGIndex(index_path).build(
        [source.parent.parent],
        metadata=_metadata(),
    )
    assert result["files_indexed"] == 1
    hits = ProjectRAGIndex(index_path).search(
        "Fabric Registry.register item crystal",
        required_metadata={"minecraft_version": "1.20.1", "loader": "fabric"},
    )
    assert hits
    assert hits[0].source_path.endswith("Example.java")
    assert hits[0].metadata["license"] == "Apache-2.0"


def test_new_index_is_streamed_sqlite_even_for_legacy_json_path(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "project"
    sources.mkdir()
    (sources / "First.java").write_text(
        "public final class First { void registerCrystal() {} }",
        encoding="utf-8",
    )
    (sources / "Second.md").write_text(
        "Fabric item registry notes for the crystal.",
        encoding="utf-8",
    )
    index_path = tmp_path / "project-index.json"

    result = ProjectRAGIndex(index_path).build(
        [sources],
        metadata=_metadata(),
    )

    assert index_path.read_bytes()[:16] == b"SQLite format 3\x00"
    assert result["index_backend"] == "sqlite"
    assert result["files_indexed"] == 2
    assert (
        inspect.signature(ProjectRAGIndex.build)
        .parameters["max_files"]
        .default
        is None
    )


def test_explicit_file_cap_remains_an_opt_in_host_policy(tmp_path: Path) -> None:
    sources = tmp_path / "project"
    sources.mkdir()
    (sources / "One.java").write_text("class One {}", encoding="utf-8")
    (sources / "Two.java").write_text("class Two {}", encoding="utf-8")

    with pytest.raises(ValueError, match="RAG file limit exceeded"):
        ProjectRAGIndex(tmp_path / "limited.sqlite").build(
            [sources],
            metadata=_metadata(),
            max_files=1,
        )


def test_default_build_has_no_former_5000_entry_cap(tmp_path: Path) -> None:
    source = tmp_path / "Repeated.java"
    source.write_text("class Repeated {}", encoding="utf-8")

    result = ProjectRAGIndex(tmp_path / "uncapped.sqlite").build(
        [source] * 5001,
        metadata=_metadata(),
    )

    # Repeated roots are deduplicated durably by SQLite, but all 5,001
    # traversal entries are accepted. The former default stopped at 5,000.
    assert result["files_indexed"] == 1
    assert result["chunks_indexed"] == 1


def test_large_single_line_source_is_streamed_instead_of_skipped(
    tmp_path: Path,
) -> None:
    source = tmp_path / "GeneratedRegistry.json"
    marker = "needle_large_registry_value"
    source.write_text(
        '{"payload":"' + ("x" * (2 * 1024 * 1024)) + marker + '"}',
        encoding="utf-8",
    )
    index = ProjectRAGIndex(tmp_path / "large-source.sqlite")

    build = index.build([source], metadata=_metadata())
    result = index.search_with_receipt(marker)

    assert build["files_indexed"] == 1
    assert build["chunks_indexed"] >= 1
    assert any(marker in hit.text for hit in result.hits)


def test_adaptive_multi_hop_search_expands_declared_relationships(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "project"
    sources.mkdir()
    (sources / "CentralRegistry.java").write_text(
        "CentralRegistry installs the moon gateway.",
        encoding="utf-8",
    )
    (sources / "TerrainShard.java").write_text(
        "TerrainShard emits deterministic region pieces.",
        encoding="utf-8",
    )
    metadata = {
        **_metadata(),
        "relations": {
            "CentralRegistry.java": [
                {"target": "TerrainShard.java", "kind": "calls"}
            ]
        },
    }
    index = ProjectRAGIndex(tmp_path / "relations.sqlite")
    index.build([sources], metadata=metadata)

    result = index.search_with_receipt(
        "Which dependency does CentralRegistry use?",
        required_metadata={
            "minecraft_version": "1.20.1",
            "source_commit": "abc123",
        },
    )

    assert result.receipt.route == "multi_hop"
    assert result.receipt.relation_expansions >= 1
    assert any(hit.source_path.endswith("TerrainShard.java") for hit in result.hits)
    related = next(
        hit for hit in result.hits if hit.source_path.endswith("TerrainShard.java")
    )
    assert related.relation_score > 0
    assert related.metadata["_rag_relation"]["kind"] == "calls"


def test_receipt_records_one_corrective_pass_and_remaining_coverage_gap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project" / "Registry.java"
    source.parent.mkdir()
    source.write_text(
        "Crystal Registry data generator for Fabric resources.",
        encoding="utf-8",
    )
    index = ProjectRAGIndex(tmp_path / "correction.db")
    index.build([source], metadata=_metadata())

    result = index.search_with_receipt("register crystal datagen")

    assert result.hits
    assert result.receipt.correction_applied is True
    assert result.receipt.corrected_query is not None
    assert "registry" in result.receipt.corrected_query
    assert result.receipt.candidates_considered >= 2
    assert result.receipt.coverage_score < 1.0
    assert "coverage_below_route_threshold" in result.receipt.warnings


def test_required_source_commit_gate_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "Example.java"
    source.write_text("class ExactSource {}", encoding="utf-8")
    index = ProjectRAGIndex(tmp_path / "gated.sqlite")
    index.build([source], metadata=_metadata())

    result = index.search_with_receipt(
        "ExactSource",
        required_metadata={
            "minecraft_version": "1.20.1",
            "loader": "fabric",
            "license": "Apache-2.0",
            "source_commit": "different-commit",
        },
    )

    assert result.hits == ()
    assert result.receipt.candidates_considered == 0
    assert result.receipt.warnings == ("required_metadata_mismatch",)


class _RecordingRouter:
    def __init__(self) -> None:
        self.embedding_batch_sizes: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedding_batch_sizes.append(len(texts))
        return [
            [float((len(text) % 17) + 1), 1.0]
            for text in texts
        ]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.5 for _ in documents]


def test_semantic_ingestion_is_batched_and_availability_is_explicit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.md"
    source.write_text(
        "\n".join(
            f"line {number} deterministic crystal registry"
            for number in range(7000)
        ),
        encoding="utf-8",
    )
    router = _RecordingRouter()
    index = ProjectRAGIndex(tmp_path / "semantic.sqlite")
    build = index.build(
        [source],
        metadata=_metadata(),
        router=router,  # type: ignore[arg-type]
        semantic=True,
    )

    assert build["chunks_indexed"] > 64
    assert max(router.embedding_batch_sizes) <= 64
    result = index.search_with_receipt(
        "deterministic crystal",
        router=router,  # type: ignore[arg-type]
        semantic=True,
        rerank=True,
    )
    assert result.receipt.semantic_used is True
    assert result.receipt.rerank_used is True

    plain_index = ProjectRAGIndex(tmp_path / "plain.sqlite")
    plain_index.build([source], metadata=_metadata())
    with pytest.raises(ValueError, match="no semantic embeddings"):
        plain_index.search(
            "crystal",
            router=router,  # type: ignore[arg-type]
            semantic=True,
        )


def test_deterministic_lexical_fallback_when_fts_is_unavailable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Fallback.java"
    source.write_text(
        "class Fallback { void registerFallbackCrystal() {} }",
        encoding="utf-8",
    )
    index_path = tmp_path / "fallback.sqlite"
    index = ProjectRAGIndex(index_path)
    index.build([source], metadata=_metadata())
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE index_meta SET value = '0' WHERE key = 'fts5'"
        )

    result = index.search_with_receipt("registerFallbackCrystal")

    assert result.hits
    assert result.receipt.lexical_backend == "deterministic_scan"


def test_existing_v1_json_index_remains_searchable(tmp_path: Path) -> None:
    index_path = tmp_path / "legacy.json"
    metadata = _metadata()
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "mmm/project-rag-index-v1",
                "chunks": [
                    {
                        "chunk_id": "sha256:legacy",
                        "source_path": "Legacy.java",
                        "text": "LegacyRegistry registers a legacy crystal.",
                        "start_line": 1,
                        "end_line": 1,
                        "sha256": "sha256:text",
                        "metadata": metadata,
                        "embedding": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ProjectRAGIndex(index_path).search_with_receipt(
        "legacy crystal",
        required_metadata={"source_commit": "abc123"},
    )

    assert result.hits
    assert result.hits[0].chunk_id == "sha256:legacy"
    assert result.receipt.lexical_backend == "legacy_scan"
