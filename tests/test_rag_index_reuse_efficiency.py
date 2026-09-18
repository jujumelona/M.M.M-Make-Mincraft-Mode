from __future__ import annotations

from minecraft_mod_ai import retrieval


def test_builtin_rag_index_is_constructed_once_per_thread(monkeypatch) -> None:
    created: list[int] = []
    original = retrieval.OfficialCorpusIndex

    class CountingIndex(original):
        def __init__(self, *args, **kwargs):
            created.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(retrieval, "OfficialCorpusIndex", CountingIndex)
    monkeypatch.delattr(retrieval._CORPUS_THREAD_STATE, "official_index", raising=False)

    first = retrieval.retrieve_official_evidence("Fabric build project")
    second = retrieval.retrieve_official_evidence("Fabric metadata project")

    assert first.schema_version == "minecraft-mod-ai/retrieval-receipt-v1"
    assert second.schema_version == "minecraft-mod-ai/retrieval-receipt-v1"
    assert len(created) == 1


def test_runtime_rag_entrypoint_owns_shared_index_contract_directly() -> None:
    assert hasattr(retrieval, "_CORPUS_THREAD_STATE")
    assert retrieval.retrieve_official_evidence.__module__ == "minecraft_mod_ai.retrieval"
    assert retrieval._retrieve_official_evidence_impl.__module__ == "minecraft_mod_ai.retrieval"
