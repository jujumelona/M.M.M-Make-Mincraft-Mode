from __future__ import annotations

import minecraft_mod_ai.game_design as game_design


def test_retired_parallel_request_page_helpers_are_absent() -> None:
    assert not hasattr(game_design, "_request_page_worker_count")
    assert not hasattr(game_design, "_generate_sharded_design_pages")


def test_large_request_pages_round_trip_losslessly() -> None:
    prompt = "alpha beta gamma delta " * 1000
    pages = game_design._lossless_request_pages(
        prompt,
        max_json_text_bytes=1024,
    )

    assert len(pages) > 1
    assert "".join(pages) == prompt
    assert all(page for page in pages)
