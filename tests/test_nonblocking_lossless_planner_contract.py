from __future__ import annotations

import inspect

from minecraft_mod_ai import (
    central_research,
    ecosystem_discovery,
    game_design,
    parallel_runtime_contract,
    small_model_agent_policy,
)


def _long_text() -> str:
    return ("networking persistence custom_java 한국어 요청 segment.\n" * 1800) + "TAIL_SENTINEL"


def test_planner_runtime_has_no_optional_future_overlap_owner() -> None:
    source = inspect.getsource(parallel_runtime_contract)
    assert "_PLANNER_AUX_EXECUTOR" not in source
    assert "_PLANNER_STATE" not in source
    assert "_install_planner_overlap" not in source
    assert "ecosystem_with_prefetch" not in source
    assert "implementation_evidence_with_overlap" not in source


def test_small_model_policy_renders_complete_request() -> None:
    request = _long_text()
    rendered = small_model_agent_policy._render(request)
    assert rendered == request
    assert rendered.endswith("TAIL_SENTINEL")
    assert len(rendered) > 24_000


def test_central_research_pages_reconstruct_long_query_exactly() -> None:
    query = _long_text()
    pages = central_research._lossless_query_pages(
        query,
        central_research._MAX_QUERY_BYTES,
    )
    assert len(pages) > 1
    assert "".join(pages) == query
    assert all(
        len(page.encode("utf-8")) <= central_research._MAX_QUERY_BYTES
        for page in pages
    )


def test_runtime_ecosystem_seed_query_keeps_full_tail() -> None:
    prompt = _long_text()
    query = ecosystem_discovery._seed_query(prompt, {})
    assert query == prompt
    assert query.endswith("TAIL_SENTINEL")
    assert len(query.encode("utf-8")) > 2_000


def test_game_design_request_pages_are_lossless() -> None:
    prompt = _long_text()
    pages = game_design._lossless_request_pages(
        prompt,
        max_json_text_bytes=4096,
    )
    assert len(pages) > 1
    assert "".join(pages) == prompt
    assert all(game_design._json_text_bytes(page) <= 4096 for page in pages)
