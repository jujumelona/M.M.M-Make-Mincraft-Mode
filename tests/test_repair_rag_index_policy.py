from __future__ import annotations

from minecraft_mod_ai import production_tools


def _project(tmp_path):
    project = tmp_path / "project"
    source = project / "src/main/java/example/Example.java"
    source.parent.mkdir(parents=True)
    source.write_text("final class Example {}\n", encoding="utf-8")
    return project


def test_repair_index_stays_lexical_even_if_legacy_wrapper_requests_semantic(monkeypatch, tmp_path) -> None:
    project = _project(tmp_path)
    service = production_tools.ProductionToolService(workspace_root=tmp_path, profile="test")
    captured = {}

    class ForbiddenRouter:
        def __init__(self, *args, **kwargs):
            raise AssertionError("repair indexing must not construct the embedding router")

    def build(self, roots, *, metadata, router=None, semantic=False, max_files=None):
        del self, roots, metadata, max_files
        captured["router"] = router
        captured["semantic"] = semantic
        return {"semantic_embeddings": semantic}

    monkeypatch.delenv("MMM_RAG_EAGER_REPAIR_SEMANTIC", raising=False)
    monkeypatch.setattr(production_tools, "ModelRouter", ForbiddenRouter)
    monkeypatch.setattr(production_tools.ProjectRAGIndex, "build", build)

    result = service.index_project_rag(
        [str(project.relative_to(tmp_path))],
        metadata={"source_commit": "sha256:test", "license": "project-local"},
        semantic=True,
    )

    assert result == {"semantic_embeddings": False}
    assert captured == {"router": None, "semantic": False}


def test_eager_repair_semantic_index_is_explicit_operator_opt_in(monkeypatch, tmp_path) -> None:
    project = _project(tmp_path)
    service = production_tools.ProductionToolService(workspace_root=tmp_path, profile="test")
    sentinel = object()
    captured = {}

    monkeypatch.setenv("MMM_RAG_EAGER_REPAIR_SEMANTIC", "1")
    monkeypatch.setattr(production_tools, "ModelRouter", lambda **_kwargs: sentinel)

    def build(self, roots, *, metadata, router=None, semantic=False, max_files=None):
        del self, roots, metadata, max_files
        captured["router"] = router
        captured["semantic"] = semantic
        return {"semantic_embeddings": semantic}

    monkeypatch.setattr(production_tools.ProjectRAGIndex, "build", build)

    result = service.index_project_rag(
        [str(project.relative_to(tmp_path))],
        metadata={"source_commit": "sha256:test", "license": "project-local"},
        semantic=True,
    )

    assert result == {"semantic_embeddings": True}
    assert captured == {"router": sentinel, "semantic": True}
