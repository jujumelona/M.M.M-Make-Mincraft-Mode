from __future__ import annotations

import inspect

from minecraft_mod_ai import (
    agentic_research_fusion,
    minecraft_mcp_evidence_contract,
    reuse_discovery,
)


def test_agentic_research_fanout_is_deadline_bound() -> None:
    source = inspect.getsource(agentic_research_fusion)
    assert "ThreadPoolExecutor" not in source
    assert "as_completed" not in source
    assert 'stage="agentic-rag-primary"' in source
    assert 'stage="agentic-rag-correction"' in source
    assert "iter_completed_with_deadlines" in source


def test_reuse_discovery_remote_fanouts_are_deadline_bound() -> None:
    source = inspect.getsource(reuse_discovery)
    assert "ThreadPoolExecutor" not in source
    assert "as_completed" not in source
    assert 'stage="reuse-modrinth-source-resolution"' in source
    assert 'stage=f"reuse-catalog-wave-{variant_index}"' in source


def test_external_mcp_batch_is_deadline_bound() -> None:
    source = inspect.getsource(minecraft_mcp_evidence_contract)
    assert "ThreadPoolExecutor" not in source
    assert "as_completed" not in source
    assert 'stage="external-mcp-batch"' in source
    assert "iter_completed_with_deadlines" in source
