from __future__ import annotations

from minecraft_mod_ai import production_tools


def test_semantic_rag_calls_reuse_one_model_router(tmp_path, monkeypatch) -> None:
    created = []

    class FakeRouter:
        def __init__(self, *, profile: str):
            self.profile = profile
            created.append(self)

    seen = []

    class FakeIndex:
        def __init__(self, path):
            self.path = path

        def build(self, roots, *, metadata, router, semantic, max_files=None):
            seen.append(router)
            return {"ok": True}

    monkeypatch.setattr(production_tools, "ModelRouter", FakeRouter)
    monkeypatch.setattr(production_tools, "ProjectRAGIndex", FakeIndex)
    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    service = production_tools.ProductionToolService(
        workspace_root=tmp_path,
        profile="test-profile",
    )

    for index in range(2):
        root = tmp_path / f"source-{index}"
        root.mkdir()
        service.index_project_rag(
            [str(root)],
            metadata={"kind": "project"},
            semantic=True,
        )

    assert len(created) == 1
    assert created[0].profile == "test-profile"
    assert seen == [created[0], created[0]]
