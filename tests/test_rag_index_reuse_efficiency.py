from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import central_research, retrieval
from minecraft_mod_ai import platform_live_rag_contract as live_rag


def test_builtin_rag_index_is_constructed_once_per_thread(monkeypatch) -> None:
    created: list[int] = []

    class FakeIndex:
        def __init__(self, *, documents):
            created.append(id(documents))
            self.documents = documents

    fake_retrieval = SimpleNamespace(
        OfficialCorpusIndex=FakeIndex,
        BUILTIN_CORPUS=(object(),),
    )
    monkeypatch.setattr(live_rag._RAG_THREAD_STATE, "indexes", {}, raising=False)

    first = live_rag._thread_index(fake_retrieval)
    second = live_rag._thread_index(fake_retrieval)
    assert first is second
    assert len(created) == 1


def test_runtime_rag_entrypoints_use_shared_index_contract() -> None:
    assert getattr(
        retrieval.retrieve_official_evidence,
        "_mmm_thread_local_index_reuse",
        False,
    )
    assert getattr(
        central_research.retrieve_official_evidence,
        "_mmm_thread_local_index_reuse",
        False,
    )
