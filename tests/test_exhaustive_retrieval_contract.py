from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from minecraft_mod_ai import github_adaptive_retrieval as gh
from minecraft_mod_ai import research_grounded_rag_contract as rg


def _doc(**kwargs):
    return dict(kwargs)


def test_repository_discovery_stops_at_explicit_work_budget(monkeypatch):
    calls: list[str] = []

    def fake_json(url: str):
        calls.append(url)
        return {
            "items": [
                {"full_name": f"owner/repo-{index}"}
                for index in range(50)
            ],
            "incomplete_results": False,
        }

    monkeypatch.setenv("MMM_GITHUB_SEARCH_REQUEST_BUDGET", "2")
    result = gh.discover_repositories("space travel", http_json=fake_json)

    assert result.search_requests == 2
    assert len(calls) == 2
    assert result.repositories
    assert result.saturation_reason == "search_request_budget_exhausted"


def test_repository_discovery_stops_immediately_on_provider_limit(monkeypatch):
    calls = 0

    def fake_json(_url: str):
        nonlocal calls
        calls += 1
        raise RuntimeError("HTTP Error 422: search limit reached")

    monkeypatch.setenv("MMM_GITHUB_SEARCH_REQUEST_BUDGET", "8")
    result = gh.discover_repositories("space travel", http_json=fake_json)

    assert calls == 1
    assert result.saturation_reason == "provider_limited"
    assert result.errors


def test_repository_document_retrieval_honors_work_budgets():
    result = gh.retrieve_repository_documents(
        "a",
        "b",
        "space travel",
        http_json=lambda _url: {},
        http_text=lambda url: url,
        source_document=_doc,
        request_budget=0,
        byte_budget=0,
        coverage_target=0.0,
    )

    assert result.documents == ()
    assert result.requests_used == 0
    assert result.saturation_reason == "source_request_budget_exhausted"


def test_recursive_tree_truncation_uses_bounded_subtree_walk():
    root_sha = "rootsha"
    child_sha = "childsha"

    def fake_json(url: str):
        if url.endswith("/repos/a/b"):
            return {"default_branch": "main", "html_url": "https://github.com/a/b"}
        if "?recursive=1" in url:
            return {"truncated": True, "tree": []}
        if url.endswith("/git/trees/main"):
            return {
                "sha": root_sha,
                "tree": [{"type": "tree", "path": "src", "sha": child_sha}],
            }
        if url.endswith(f"/git/trees/{child_sha}"):
            return {
                "sha": child_sha,
                "tree": [{"type": "blob", "path": "Space.java", "size": 16}],
            }
        raise AssertionError(url)

    result = gh.retrieve_repository_documents(
        "a",
        "b",
        "space",
        http_json=fake_json,
        http_text=lambda _url: "class Space {}",
        source_document=_doc,
        request_budget=8,
    )

    assert any(
        str(item.get("source_id", "")).endswith("src/Space.java")
        for item in result.documents
    )
    assert result.requests_used <= 8
    assert result.tree_truncated is True


def test_source_document_keeps_complete_content_within_source_limit():
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


def test_modrinth_fetches_one_relevance_page_not_the_entire_catalog(monkeypatch):
    search_offsets: list[int] = []

    def fake_json(url: str, *, github: bool = False):
        del github
        if "/v2/project/" in url:
            return {}
        qs = parse_qs(urlparse(url).query)
        search_offsets.append(int(qs.get("offset", ["0"])[0]))
        return {
            "hits": [
                {"project_id": f"p{index}", "slug": f"p{index}"}
                for index in range(4)
            ],
            "total_hits": 10_000,
        }

    monkeypatch.setattr(rg, "_http_json", fake_json)
    projects, errors = rg._modrinth_search("space", ())

    assert not errors
    assert len(projects) == 4
    assert search_offsets == [0]


def test_discovery_query_variants_are_deterministic_and_bounded():
    variants = rg._query_variants("space.travel")

    assert variants == ("space.travel", "space travel")
    assert all("source implementation" not in value for value in variants)
