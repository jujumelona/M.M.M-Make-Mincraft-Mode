from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import central_research
from minecraft_mod_ai import platform_live_rag_contract as contract


def _fake_retrieval_module():
    class FakeIndex:
        original_calls = 0
        last_target = None

        def __init__(self, documents=()):
            self.documents = documents

        def retrieve(
            self,
            query: str,
            *,
            minecraft_version=None,
            loader=None,
            mappings=None,
            limit=6,
        ):
            type(self).original_calls += 1
            type(self).last_target = (minecraft_version, loader, mappings, limit)
            return [query]

    return SimpleNamespace(
        OfficialCorpusIndex=FakeIndex,
        BUILTIN_CORPUS=("doc",),
        retrieve_official_evidence=lambda *args, **kwargs: ["original-public"],
    )


def _install_without_central_side_effects(monkeypatch: pytest.MonkeyPatch, retrieval) -> None:
    original_central_retrieve = central_research.retrieve_official_evidence
    monkeypatch.setattr(
        central_research,
        "retrieve_official_evidence",
        original_central_retrieve,
    )
    monkeypatch.setattr(contract, "_replace_kwonly_default", lambda *args, **kwargs: None)
    contract.install(retrieval_module=retrieval)


def test_missing_target_skips_index_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieval = _fake_retrieval_module()
    _install_without_central_side_effects(monkeypatch, retrieval)

    index = retrieval.OfficialCorpusIndex()
    assert index.retrieve("query") == []
    assert retrieval.OfficialCorpusIndex.original_calls == 0


def test_missing_target_skips_thread_index_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieval = _fake_retrieval_module()
    _install_without_central_side_effects(monkeypatch, retrieval)

    def fail_if_indexed(_retrieval):
        raise AssertionError("Official RAG index must not be built without a concrete target")

    monkeypatch.setattr(contract, "_thread_index", fail_if_indexed)
    assert retrieval.retrieve_official_evidence("query") == []


def test_concrete_target_preserves_original_retrieval_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = _fake_retrieval_module()
    _install_without_central_side_effects(monkeypatch, retrieval)

    index = retrieval.OfficialCorpusIndex()
    assert index.retrieve(
        "query",
        minecraft_version="1.21.1",
        loader="Fabric",
        mappings="yarn",
        limit=4,
    ) == ["query"]
    assert retrieval.OfficialCorpusIndex.original_calls == 1
    assert retrieval.OfficialCorpusIndex.last_target == ("1.21.1", "fabric", "yarn", 4)
