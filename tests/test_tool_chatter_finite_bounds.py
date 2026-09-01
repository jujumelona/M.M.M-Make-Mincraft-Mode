from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    ModelConfigurationError,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    RetrievalDecision,
    RetrievalObservation,
    RetrievalProgress,
    generate_with_tools,
)


def test_retrieval_progress_stops_duplicate_queries() -> None:
    """Verify that RetrievalProgress rejects repeated queries as duplicate."""
    progress = RetrievalProgress()

    d1 = progress.begin("search_code_rag", {"query": "BlockRegistry"})
    assert d1 == RetrievalDecision.EXECUTE

    obs = progress.observe(
        "search_code_rag",
        {"query": "BlockRegistry"},
        {"results": ["BlockRegistry.java"]},
        usable=True,
    )
    assert obs == RetrievalObservation.FRESH

    d2 = progress.begin("search_code_rag", {"query": "BlockRegistry"})
    assert d2 == RetrievalDecision.DUPLICATE_QUERY


def test_retrieval_progress_detects_duplicate_evidence() -> None:
    """Verify that RetrievalProgress detects identical evidence returned from different queries."""
    progress = RetrievalProgress()

    progress.begin("search_code_rag", {"query": "BlockRegistry"})
    obs1 = progress.observe(
        "search_code_rag",
        {"query": "BlockRegistry"},
        {"results": ["BlockRegistry.java"]},
        usable=True,
    )
    assert obs1 == RetrievalObservation.FRESH

    progress.begin("search_code_rag", {"query": "FindBlocks"})
    obs2 = progress.observe(
        "search_code_rag",
        {"query": "FindBlocks"},
        {"results": ["BlockRegistry.java"]},
        usable=True,
    )
    assert obs2 == RetrievalObservation.DUPLICATE_EVIDENCE


def test_forced_rag_finite_attempts_cap() -> None:
    """Verify that host-forced RAG terminates after the bounded violating attempt."""
    from contextlib import nullcontext

    router = MagicMock()
    router._generation_scope.return_value = nullcontext()
    config = MagicMock()
    config.adapter = "llama_cpp"
    config.max_context = 16384
    config.max_input_tokens = 0
    config.max_new_tokens = 4096
    config.extra = {"runtime_contract": "qwen", "qwen_family": "qwen3.5"}

    adapter = MagicMock()
    # This regression exercises the forced-RAG loop, not exact llama context
    # accounting. Explicitly model an adapter without that optional capability;
    # MagicMock would otherwise fabricate a callable method and fake token fields.
    adapter.input_context_accounting = None
    adapter.generate_turn.return_value = MagicMock(
        content="I am not calling the tool.",
        tool_calls=(),
    )

    request = GenerationRequest(
        messages=(
            {
                "role": "user",
                "content": "Implement feature with mandatory RAG",
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
        ),
        metadata={"require_rag": True},
    )

    runtime = MagicMock()
    runtime.allowed_tools.return_value = frozenset({"search_code_rag"})

    with pytest.raises(
        ModelConfigurationError,
        match="Production coder did not honor host-forced RAG tool choice",
    ):
        generate_with_tools(
            router,
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )
