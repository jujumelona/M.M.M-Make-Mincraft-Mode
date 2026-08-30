from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from minecraft_mod_ai import github_adaptive_retrieval as github_rag


def _document(**kwargs):
    return dict(kwargs)


def test_repository_query_ladder_broadens_obligation_to_recall_queries():
    ladder = github_rag.repository_query_ladder(
        "planet_colonization reusable implementation source code Players can colonize planets."
    )

    assert ladder[0].startswith("planet colonization")
    assert "planet minecraft" in ladder
    assert "planet fabric" in ladder
    assert all("reusable implementation source code" not in item.casefold() for item in ladder)


def test_repository_discovery_broadens_after_specific_query_returns_zero(monkeypatch):
    calls: list[str] = []

    def http_json(url: str):
        q = parse_qs(urlparse(url).query)["q"][0]
        calls.append(q)
        if q.startswith("planet minecraft "):
            return {"items": [{"full_name": "Mixinors/Astromine"}]}
        return {"items": []}

    monkeypatch.setenv("MMM_GITHUB_SEARCH_REQUEST_BUDGET", "8")
    result = github_rag.discover_repositories(
        "planet_colonization reusable implementation source code Players can colonize planets.",
        http_json=http_json,
    )

    assert ("Mixinors", "Astromine") in result.repositories
    assert any(call.startswith("planet colonization ") for call in calls)
    assert any(call.startswith("planet minecraft ") for call in calls)


def test_repository_discovery_does_not_truncate_candidates_to_old_repo_cap(monkeypatch):
    repositories = [
        {"full_name": f"owner/repo-{index}"}
        for index in range(9)
    ]

    def http_json(url: str):
        del url
        return {"items": repositories}

    monkeypatch.setenv("MMM_GITHUB_SEARCH_REQUEST_BUDGET", "1")
    result = github_rag.discover_repositories("spacecraft", http_json=http_json)

    assert len(result.repositories) == 9


def test_repository_inspection_can_reach_relevant_file_after_fourth(monkeypatch):
    raw_calls: list[str] = []
    paths = [f"src/main/java/demo/A{index}.java" for index in range(8)]

    def http_json(url: str):
        if url.endswith("/repos/owner/repo"):
            return {
                "default_branch": "main",
                "html_url": "https://github.com/owner/repo",
                "license": {"spdx_id": "MIT"},
            }
        if "/git/trees/" in url:
            return {
                "truncated": False,
                "tree": [
                    {"path": path, "type": "blob", "size": 80}
                    for path in paths
                ],
            }
        raise AssertionError(url)

    def http_text(url: str):
        raw_calls.append(url)
        if url.endswith("/A7.java"):
            return "public final class PlanetColonyManager { void colonizePlanet() {} }"
        return "public final class Helper { void tick() {} }"

    monkeypatch.setenv("MMM_GITHUB_SOURCE_REQUEST_BUDGET", "32")
    monkeypatch.setenv("MMM_GITHUB_SOURCE_BYTE_BUDGET", str(1024 * 1024))
    monkeypatch.setenv("MMM_GITHUB_EVIDENCE_COVERAGE_TARGET", "1.0")
    result = github_rag.retrieve_repository_documents(
        "owner",
        "repo",
        "planet colony",
        http_json=http_json,
        http_text=http_text,
        source_document=_document,
    )

    assert len(raw_calls) == 8
    assert any(str(item["source_id"]).endswith(":src/main/java/demo/A7.java") for item in result.documents)
    assert result.coverage_score == 1.0
    assert result.saturation_reason == "evidence_coverage_satisfied"


def test_source_inspection_is_query_specific_for_same_repository(monkeypatch):
    paths = [
        "src/main/java/demo/TradeLedger.java",
        "src/main/java/demo/PlanetColony.java",
    ]

    def http_json(url: str):
        if url.endswith("/repos/owner/repo"):
            return {"default_branch": "main", "html_url": "https://github.com/owner/repo"}
        if "/git/trees/" in url:
            return {
                "truncated": False,
                "tree": [{"path": path, "type": "blob", "size": 120} for path in paths],
            }
        raise AssertionError(url)

    def http_text(url: str):
        if url.endswith("/TradeLedger.java"):
            return "class TradeLedger { void tradeCurrency() {} }"
        if url.endswith("/PlanetColony.java"):
            return "class PlanetColony { void colonizePlanet() {} }"
        raise AssertionError(url)

    monkeypatch.setenv("MMM_GITHUB_SOURCE_REQUEST_BUDGET", "20")
    monkeypatch.setenv("MMM_GITHUB_EVIDENCE_COVERAGE_TARGET", "1.0")
    trade = github_rag.retrieve_repository_documents(
        "owner",
        "repo",
        "trade currency",
        http_json=http_json,
        http_text=http_text,
        source_document=_document,
    )
    planet = github_rag.retrieve_repository_documents(
        "owner",
        "repo",
        "planet colony",
        http_json=http_json,
        http_text=http_text,
        source_document=_document,
    )

    assert str(trade.documents[0]["source_id"]).endswith(":src/main/java/demo/TradeLedger.java")
    assert str(planet.documents[0]["source_id"]).endswith(":src/main/java/demo/PlanetColony.java")
