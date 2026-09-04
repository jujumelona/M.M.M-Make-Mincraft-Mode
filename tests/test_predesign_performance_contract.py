from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from minecraft_mod_ai import pre_design_grounded_rag as rag


def test_linked_github_readmes_are_fetched_concurrently(monkeypatch):
    barrier = threading.Barrier(2, timeout=2.0)

    def text(url: str, headers=None) -> str:
        barrier.wait()
        return f"implementation body from {url}"

    monkeypatch.setattr(rag, "_text", text)
    records = [
        {"metadata": {"source_url": f"https://github.com/example/repo-{index}"}}
        for index in range(4)
    ]

    found, receipt = rag._linked_github_sources(
        records, disabled=lambda: False, disable=lambda: None
    )

    assert [item["metadata"]["repository"] for item in found] == [
        f"example/repo-{index}" for index in range(4)
    ]
    assert receipt["source_requests"] == 4


def test_identical_evidence_write_skips_second_fsync(monkeypatch, tmp_path):
    target = tmp_path / "evidence.json"
    rag._write(target, "same payload")

    def explode(_fd: int) -> None:
        raise AssertionError("identical content must not be rewritten or fsynced")

    monkeypatch.setattr(rag.os, "fsync", explode)
    rag._write(target, "same payload")
    assert target.read_text(encoding="utf-8") == "same payload"


def test_predesign_code_rag_uses_lexical_hot_path(monkeypatch):
    calls: list[dict[str, object]] = []

    @dataclass(frozen=True)
    class Receipt:
        route: str = "generic"

    class Result:
        hits = ()
        receipt = Receipt()

    class FakeIndex:
        def __init__(self, path: Path) -> None:
            self.path = path

        def search_with_receipt(self, query: str, **kwargs):
            calls.append({"query": query, **kwargs})
            return Result()

    monkeypatch.setattr(rag, "ProjectRAGIndex", FakeIndex)
    result = rag._search_code_index(Path("unused.db"), "alien combat behavior")

    assert result["status"] == "searched"
    assert calls == [
        {
            "query": "alien combat behavior",
            "limit": 8,
            "semantic": False,
            "rerank": False,
        }
    ]


def test_github_fallback_uses_bounded_parallel_slots(monkeypatch):
    monkeypatch.delenv("CURSEFORGE_API_KEY", raising=False)
    barrier = threading.Barrier(2, timeout=2.0)
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def github(query: str, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait()
            return [], {
                "provider": "github",
                "status": "available",
                "result_count": 0,
                "search_requests": 1,
                "source_requests": 0,
            }
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(
        rag,
        "_search_modrinth",
        lambda query: ([], {"provider": "modrinth", "status": "available", "result_count": 0}),
    )
    monkeypatch.setattr(rag, "_search_github", github)
    monkeypatch.setattr(
        rag,
        "_search_authoritative_catalog",
        lambda query, versions: {"sources": [], "errors": []},
    )
    monkeypatch.setattr(rag, "_existing_code_index", lambda: None)
    monkeypatch.setattr(
        rag,
        "_search_code_index",
        lambda index, query: {"status": "not_indexed", "hits": []},
    )

    brief = {
        "domains": [
            {
                "domain_id": "request",
                "queries": ["query one", "query two"],
            }
        ]
    }
    bundle = rag._forced_rag_bundle(object(), brief)

    assert max_active == 2
    assert bundle["github_fallback_workers"] == rag._MAX_GITHUB_FALLBACK_WORKERS
