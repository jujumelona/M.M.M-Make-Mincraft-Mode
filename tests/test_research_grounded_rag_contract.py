from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import research_grounded_rag_contract as grounded


def test_removed_remote_retrieval_is_not_importable() -> None:
    """The deleted GitHub module must not be restored or used by planning."""
    assert not hasattr(grounded, "_github_adaptive_search")
    assert not hasattr(grounded, "_github_targeted_search")
    assert not hasattr(grounded, "_modrinth_search")
    assert not hasattr(grounded, "_external_retrieval")
    source = (Path(grounded.__file__)).read_text(encoding="utf-8")
    assert "github_adaptive_retrieval" not in source
    assert "api.modrinth.com" not in source
    assert "api.github.com" not in source


def test_plan_wrapper_uses_existing_curseforge_evidence_without_second_search(monkeypatch) -> None:
    """Existing results are retained; the wrapper only attaches local index."""
    queried = []
    payload = {
        "schema_version": "mmm/pre-design-grounded-rag-v6",
        "versions": ["1.21.1"],
        "external_source_count": 1,
        "domains": [{"queries": [{"external_rag": {
            "sources": [{"source_id": "curseforge:123", "provider": "curseforge"}],
            "providers": {"curseforge": {"status": "available"}},
        }}]}],
    }
    monkeypatch.setattr(
        grounded, "_ensure_local_index",
        lambda agentic, router: {"status": "available", "index_path": "/tmp/index.db"},
    )
    agentic = SimpleNamespace(
        _forced_rag_bundle=lambda router, brief: (queried.append(brief), payload)[1],
        _sha256=lambda value: "sha256:local",
    )
    grounded.install(agentic)
    result = agentic._forced_rag_bundle(object(), {"domains": []})
    assert len(queried) == 1
    assert result["schema_version"] == "mmm/pre-design-grounded-rag-v6"
    assert result["domains"] == payload["domains"]
    assert result["external_source_count"] == 1
    assert result["code_index_status"] == "available"
    assert result["research_sha256"] == "sha256:local"


def test_local_rag_index_is_built_when_missing(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Example.java").write_text("class Example {}", encoding="utf-8")
    monkeypatch.setenv("MMM_WORKSPACE", str(workspace))
    monkeypatch.delenv("MMM_PROJECT_RAG_INDEX", raising=False)

    class FakeIndex:
        def __init__(self, index_path):
            self.index_path = Path(index_path)

        def build(self, roots, **kwargs):
            assert roots == [workspace.resolve()]
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.write_bytes(b"fake-index")
            return {
                "schema_version": "mmm/rag-build-result-v1",
                "files_indexed": 1,
                "chunks_indexed": 1,
            }

    monkeypatch.setattr(grounded, "ProjectRAGIndex", FakeIndex)
    agentic = SimpleNamespace(_existing_code_index=lambda: None)

    receipt = grounded._ensure_local_index(agentic, router=None)

    assert receipt["status"] == "available"
    assert receipt["built"] is True
    assert Path(receipt["index_path"]).is_file()
    assert Path(receipt["index_path"]).name == "project-index.db"
    assert Path(receipt["index_path"]).parent.name == ".minecraft_ai"


def test_local_rag_never_indexes_process_cwd_without_project_scope(
    monkeypatch, tmp_path: Path
) -> None:
    engine_checkout = tmp_path / "mmm-engine"
    engine_checkout.mkdir()
    (engine_checkout / "SKILL.md").write_text("internal policy", encoding="utf-8")
    monkeypatch.chdir(engine_checkout)
    monkeypatch.delenv("MMM_WORKSPACE", raising=False)
    monkeypatch.delenv("MMM_PROJECT_RAG_INDEX", raising=False)

    class ForbiddenIndex:
        def __init__(self, _index_path):
            raise AssertionError("process CWD must not be indexed")

    monkeypatch.setattr(grounded, "ProjectRAGIndex", ForbiddenIndex)
    agentic = SimpleNamespace(_existing_code_index=lambda: None)

    receipt = grounded._ensure_local_index(agentic, router=None)

    assert receipt["status"] == "workspace_unconfigured"
    assert receipt["built"] is False


def test_router_attached_workspace_has_priority_over_process_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    engine_checkout = tmp_path / "mmm-engine"
    target_workspace = tmp_path / "generated-mod"
    engine_checkout.mkdir()
    target_workspace.mkdir()
    monkeypatch.chdir(engine_checkout)
    monkeypatch.delenv("MMM_WORKSPACE", raising=False)
    stale_index = engine_checkout / "rag" / "project-index.db"
    stale_index.parent.mkdir()
    stale_index.write_bytes(b"stale")
    monkeypatch.setenv("MMM_PROJECT_RAG_INDEX", str(stale_index))
    built_roots = []

    class FakeIndex:
        def __init__(self, index_path):
            self.index_path = Path(index_path)

        def build(self, roots, **_kwargs):
            built_roots.extend(roots)
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.write_bytes(b"index")
            return {"files_indexed": 0, "chunks_indexed": 0}

    monkeypatch.setattr(grounded, "ProjectRAGIndex", FakeIndex)
    router = SimpleNamespace(_mmm_workspace_root=str(target_workspace))
    agentic = SimpleNamespace(_existing_code_index=lambda: stale_index)

    receipt = grounded._ensure_local_index(agentic, router)

    assert receipt["status"] == "available"
    assert built_roots == [target_workspace.resolve()]
    assert Path(receipt["index_path"]).is_relative_to(target_workspace)
    assert stale_index.read_bytes() == b"stale"


def test_bundle_augmentation_preserves_curseforge_sources() -> None:
    agentic = SimpleNamespace(_sha256=lambda value: "sha256:test")
    payload = {
        "schema_version": "mmm/pre-design-grounded-rag-v6",
        "external_source_count": 1,
        "domains": [{"queries": [{"external_rag": {
            "sources": [{"source_id": "curseforge:123", "provider": "curseforge"}]
        }}]}],
    }
    result = grounded._augment_bundle(
        agentic, payload,
        local_index={"status": "available", "index_path": "/tmp/index.db"},
    )
    assert result["domains"] == payload["domains"]
    assert result["external_source_count"] == 1
    assert result["code_index_status"] == "available"
    assert result["research_sha256"] == "sha256:test"
