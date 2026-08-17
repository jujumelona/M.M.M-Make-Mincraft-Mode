from __future__ import annotations

from pathlib import Path

import minecraft_mod_ai.small_model_hybrid_search_contract as hybrid
import minecraft_mod_ai.small_model_relation_index_contract as relation_contract
from minecraft_mod_ai.production_tools import ProductionToolService


def _metadata(source_commit: str) -> dict[str, str]:
    return {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mapping_namespace": "yarn",
        "java_version": "17",
        "license": "project-local",
        "source_commit": source_commit,
    }


def test_exact_project_snapshot_skips_relation_scan_and_index_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    source = project / "src/main/java/demo/Test.java"
    source.parent.mkdir(parents=True)
    source.write_text("package demo; final class Test {}\n", encoding="utf-8")

    calls = 0
    real_derive = relation_contract.derive_relations

    def counted(roots):
        nonlocal calls
        calls += 1
        return real_derive(roots)

    monkeypatch.setattr(relation_contract, "derive_relations", counted)
    service = ProductionToolService(workspace_root=workspace, profile="t4_local")

    first = service.index_project_rag(["project"], metadata=_metadata("snapshot-a"))
    second = service.index_project_rag(["project"], metadata=_metadata("snapshot-a"))

    assert first["chunks_indexed"] >= 1
    assert calls == 1
    assert second["reused"] is True
    assert second["reuse_reason"] == "exact_project_snapshot"

    service.index_project_rag(["project"], metadata=_metadata("snapshot-b"))
    assert calls == 2


def test_search_cache_is_bound_to_exact_index_file_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "rag/project-index.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"first-index")
    service = ProductionToolService(workspace_root=workspace, profile="t4_local")

    first_key = hybrid._search_cache_key(
        service,
        query="Target dependency",
        index_path="rag/project-index.json",
        limit=8,
        semantic=False,
        rerank=False,
        required_metadata={"source_commit": "snapshot-a"},
    )
    hybrid._search_cache_put(first_key, {"hits": [{"source_path": "Target.java"}]})
    cached = hybrid._search_cache_get(first_key)
    assert cached is not None
    assert cached["search_reused"] is True

    target.write_bytes(b"second-index-with-different-identity")
    second_key = hybrid._search_cache_key(
        service,
        query="Target dependency",
        index_path="rag/project-index.json",
        limit=8,
        semantic=False,
        rerank=False,
        required_metadata={"source_commit": "snapshot-a"},
    )
    assert second_key != first_key
    assert hybrid._search_cache_get(second_key) is None


def test_runtime_search_exposes_snapshot_reuse_contract() -> None:
    assert getattr(
        ProductionToolService.search_code_rag,
        "_mmm_snapshot_search_reuse",
        False,
    )
