from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from minecraft_mod_ai import github_adaptive_retrieval as gh
from minecraft_mod_ai import research_grounded_rag_contract as rg


def _doc(**kwargs):
    return dict(kwargs)


def test_repository_discovery_follows_duplicate_only_page_until_empty():
    calls = []

    def fake_json(url: str):
        calls.append(url)
        page = int(parse_qs(urlparse(url).query)["page"][0])
        # Every query-variant has a duplicate-only middle page and a new later page.
        if page == 1:
            return {"items": [{"full_name": "a/one"}], "incomplete_results": False}
        if page == 2:
            return {"items": [{"full_name": "a/one"}], "incomplete_results": False}
        if page == 3:
            return {"items": [{"full_name": "b/two"}], "incomplete_results": False}
        return {"items": [], "incomplete_results": False}

    result = gh.discover_repositories("space travel", http_json=fake_json)
    assert ("b", "two") in result.repositories
    assert result.saturation_reason == "frontier_exhausted"
    assert any("page=4" in url for url in calls)


def test_repository_discovery_distinguishes_provider_limit_from_exhaustion():
    def fake_json(_url: str):
        raise RuntimeError("HTTP Error 422: search limit reached")

    result = gh.discover_repositories("space travel", http_json=fake_json)
    assert result.saturation_reason == "provider_limit"
    assert result.errors


def test_repository_document_retrieval_ignores_legacy_cardinality_budgets():
    tree = {
        "truncated": False,
        "tree": [
            {"type": "blob", "path": "src/main/java/A.java", "size": 999999},
            {"type": "blob", "path": "src/test/java/ATest.java", "size": 999999},
            {"type": "blob", "path": "src/main/resources/x.json", "size": 999999},
        ],
    }

    def fake_json(url: str):
        if "/git/trees/" in url:
            return tree
        return {"default_branch": "main", "html_url": "https://github.com/a/b", "license": {"spdx_id": "MIT"}}

    result = gh.retrieve_repository_documents(
        "a", "b", "space travel", http_json=fake_json,
        http_text=lambda url: url,
        source_document=_doc,
        request_budget=0,
        byte_budget=0,
        coverage_target=0.0,
    )
    assert len(result.documents) == len(tree["tree"])
    assert result.saturation_reason == "frontier_exhausted"


def test_recursive_tree_truncation_falls_back_to_subtree_frontier():
    root_sha = "rootsha"
    child_sha = "childsha"

    def fake_json(url: str):
        if "?recursive=1" in url:
            return {"truncated": True, "tree": []}
        if url.endswith("/git/trees/main"):
            return {"sha": root_sha, "tree": [{"type": "tree", "path": "src", "sha": child_sha}]}
        if url.endswith(f"/git/trees/{child_sha}"):
            return {"sha": child_sha, "tree": [{"type": "blob", "path": "A.java", "size": 1}]}
        return {"default_branch": "main", "html_url": "https://github.com/a/b"}

    result = gh.retrieve_repository_documents(
        "a", "b", "space", http_json=fake_json,
        http_text=lambda _url: "class A {}", source_document=_doc,
    )
    assert any(str(item.get("source_id", "")).endswith("src/A.java") for item in result.documents)
    assert result.saturation_reason == "frontier_exhausted"


def test_source_document_keeps_complete_content():
    text = "complete source body whose terminal marker must remain: TAIL_MARKER"
    doc = rg._source_document(
        source_id="x",
        title="x",
        url="https://example.invalid/x",
        content=text,
        source_type="test",
    )
    assert doc["content"] == text
    assert doc["content"].endswith("TAIL_MARKER")



def test_modrinth_paginates_to_reported_total(monkeypatch):
    offsets = []

    def fake_json(url: str, *, github: bool = False):
        del github
        if "/v2/project/" in url:
            return {}
        qs = parse_qs(urlparse(url).query)
        offset = int(qs.get("offset", ["0"])[0])
        offsets.append(offset)
        if offset == 0:
            return {"hits": [{"project_id": "p1", "slug": "p1"}], "total_hits": 2}
        return {"hits": [{"project_id": "p2", "slug": "p2"}], "total_hits": 2}

    monkeypatch.setattr(rg, "_http_json", fake_json)
    projects, errors = rg._modrinth_search("space", ())
    assert not errors
    assert [p["project_id"] for p in projects] == ["p1", "p2"]
    assert len(offsets) == len(projects)


def test_query_variants_are_not_top_n_sliced():
    variants = rg._query_variants("space.travel")
    assert variants == ("space.travel", "space travel", "space travel minecraft fabric mod source implementation")
