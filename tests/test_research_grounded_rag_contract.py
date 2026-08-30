from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import research_grounded_rag_contract as grounded


def test_discovery_query_variants_preserve_original_without_forced_source_suffix() -> None:
    variants = grounded._query_variants("maplestory.mode.spawn.mob")

    assert variants == ("maplestory.mode.spawn.mob", "maplestory mode spawn mob")
    assert all("source implementation" not in item for item in variants)
    assert len(variants) == len(set(variants))


def test_external_retrieval_needs_no_api_key_and_follows_source_repository(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)

    readme = base64.b64encode(b"Mob spawning implementation and registry notes").decode()

    def fake_json(url: str, *, github: bool = False):
        del github
        if "api.modrinth.com/v2/search" in url:
            return {
                "hits": [
                    {
                        "project_id": "abc123",
                        "slug": "example-mob-mod",
                        "title": "Example Mob Mod",
                        "description": "Adds configurable mob spawning",
                        "author": "example",
                        "versions": ["1.21.1"],
                        "downloads": 100,
                        "license": "MIT",
                    }
                ]
            }
        if "api.modrinth.com/v2/project/abc123" in url:
            return {
                "source_url": "https://github.com/example/example-mob-mod",
                "body": "Open source Fabric mob spawning implementation",
            }
        if url == "https://api.github.com/repos/example/example-mob-mod":
            return {
                "default_branch": "main",
                "html_url": "https://github.com/example/example-mob-mod",
                "license": {"spdx_id": "MIT"},
            }
        if "/readme?ref=main" in url:
            return {
                "content": readme,
                "html_url": "https://github.com/example/example-mob-mod/blob/main/README.md",
            }
        if "/git/trees/main?recursive=1" in url:
            return {
                "tree": [
                    {
                        "type": "blob",
                        "path": "src/main/java/example/MobSpawnRegistry.java",
                    }
                ]
            }
        raise AssertionError(url)

    monkeypatch.setattr(grounded, "_http_json", fake_json)
    monkeypatch.setattr(
        grounded,
        "_http_text",
        lambda url, **kwargs: "class MobSpawnRegistry { void registerMobSpawn() {} }",
    )

    result = grounded._external_retrieval("mob spawn registry", ("1.21.1",))

    assert result["credentials_required"] is False
    assert result["status"] == "available"
    assert result["corrective_search_used"] is False
    assert result["actual_source_document_count"] >= 1
    source_types = {item["source_type"] for item in result["documents"]}
    assert "github_source" in source_types


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


def test_bundle_augmentation_keeps_external_source_content(monkeypatch) -> None:
    def fake_external(query, versions, *, mode):
        del versions
        assert mode == grounded._PLANNING_DISCOVERY
        return {
            "schema_version": "mmm/external-grounded-rag-v1",
            "status": "available",
            "query": query,
            "actual_source_document_count": 1,
            "document_count": 1,
            "documents": [
                {
                    "source_id": "github:a/b:Example.java",
                    "source_type": "github_source",
                    "url": "https://github.com/a/b/blob/main/Example.java",
                    "content": "class Example {}",
                }
            ],
        }

    monkeypatch.setattr(grounded, "_external_retrieval", fake_external)
    agentic = SimpleNamespace(_sha256=lambda value: "sha256:test")
    payload = {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "versions": [],
        "domains": [
            {
                "domain_id": "mob",
                "queries": [
                    {
                        "query": "mob spawn",
                        "project_rag": {"sources": []},
                        "code_rag": {"hits": []},
                    }
                ],
            }
        ],
    }

    result = grounded._augment_bundle(
        agentic,
        payload,
        versions=(),
        local_index={"status": "available", "index_path": "/tmp/index.db"},
        external_queries=("mob spawn",),
    )

    query = result["domains"][0]["queries"][0]
    assert query["external_rag"]["documents"][0]["content"] == "class Example {}"
    assert result["external_source_count"] == 1
    assert result["code_index_status"] == "available"
