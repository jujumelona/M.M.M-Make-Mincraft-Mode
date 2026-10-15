"""Active construction evidence must fit a small-model turn without forgetting sources."""
from __future__ import annotations

from minecraft_mod_ai.construction_memory import (
    ConstructionMemory,
    _MAX_ACTIVE_EVIDENCE_BYTES,
)


def _source_page():
    return {"source": [
        {"signature": f"void method{index}()", "code": "x" * 7000}
        for index in range(5)
    ]}


def test_large_evidence_is_recallable_after_active_view_eviction():
    memory = ConstructionMemory()
    page = _source_page()
    memory.remember("source", "src/main/java/Example.java", 0, page)

    assert len(memory.records) == 5
    assert memory._active_bytes() <= _MAX_ACTIVE_EVIDENCE_BYTES
    assert len(memory.active) < len(memory.records)

    oldest = next(key for key in memory.records if key not in memory.active)
    assert oldest in {row["evidence_id"] for row in memory.catalogue()["records"]}
    memory.recall(oldest)
    assert oldest in memory.active
    assert len(memory.records) == 5
    assert memory._active_bytes() <= _MAX_ACTIVE_EVIDENCE_BYTES


def test_resumed_checkpoint_retain_evicted_evidence_and_bounded_view():
    page = _source_page()
    first = ConstructionMemory()
    first.remember("source", "src/main/java/Example.java", 0, page)
    saved = first.export()
    restored = ConstructionMemory()
    restored.restore(saved, lambda kind, query, offset: page)
    assert set(restored.records) == set(first.records)
    assert restored._active_bytes() <= _MAX_ACTIVE_EVIDENCE_BYTES
    assert restored.export() == first.export()
