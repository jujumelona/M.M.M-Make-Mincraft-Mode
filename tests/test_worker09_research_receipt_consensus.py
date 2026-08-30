from __future__ import annotations

import pytest

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.research_ledger import (
    ResearchLedgerError,
    _sha256_json,
    select_module_research_context,
)


def _module(
    module_id: str,
    *,
    shard_index: int,
    shard_count: int,
    facts: list[dict[str, object]],
    corpus_sha256: str,
    fact_count: int,
) -> ProductionModule:
    return ProductionModule(
        module_id=module_id,
        kind="integration",
        config={
            "integration_type": "mmm_research_shard",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "facts": facts,
            "receipt": {
                "facts_sha256": corpus_sha256,
                "fact_count": fact_count,
                "shard_fact_count": len(facts),
                "shard_sha256": _sha256_json(facts),
            },
            "artifact": {
                "target_path": f".minecraft_ai/research/{module_id}.json",
            },
        },
    )


def test_research_shards_require_consensus_on_corpus_fact_count() -> None:
    first = {"fact_id": "fact:a"}
    second = {"fact_id": "fact:b"}
    corpus = [first, second]
    corpus_hash = _sha256_json(corpus)
    modules = [
        _module(
            "research_a",
            shard_index=0,
            shard_count=2,
            facts=[first],
            corpus_sha256=corpus_hash,
            fact_count=1,
        ),
        _module(
            "research_b",
            shard_index=1,
            shard_count=2,
            facts=[second],
            corpus_sha256=corpus_hash,
            fact_count=2,
        ),
    ]

    with pytest.raises(ResearchLedgerError, match="fact_count receipts disagree"):
        select_module_research_context(modules, query="anything")


def test_empty_research_corpus_still_requires_matching_corpus_hash() -> None:
    module = _module(
        "research_empty",
        shard_index=0,
        shard_count=1,
        facts=[],
        corpus_sha256="sha256:" + "0" * 64,
        fact_count=0,
    )

    with pytest.raises(ResearchLedgerError, match="approved corpus receipt"):
        select_module_research_context([module], query="anything")
