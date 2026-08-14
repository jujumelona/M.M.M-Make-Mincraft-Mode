from __future__ import annotations

import threading

import minecraft_mod_ai.game_design as game_design


def _design(page_index: int) -> dict[str, object]:
    return {
        "title": f"page-{page_index}",
        "pitch": f"page {page_index}",
        "core_loop": [],
        "progression": [],
        "combat": {},
        "mod_context": {},
        "modules": [],
        "assets": [],
        "acceptance_tests": [],
    }


def test_large_request_pages_overlap_and_merge_in_page_order(monkeypatch) -> None:
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_generate(
        router,
        *,
        request_text,
        media_paths,
        page_index,
        page_count,
    ):
        del router, request_text, media_paths
        assert page_count == 2
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=2.0)
            return _design(page_index)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(game_design, "_generate_sharded_design_page", fake_generate)
    monkeypatch.setattr(game_design, "_request_page_worker_count", lambda router, width: 2)

    results = game_design._generate_sharded_design_pages(
        object(),
        page_jobs=((0, "page-zero", {}), (1, "page-one", {})),
        media_paths=(),
    )

    assert max_active == 2
    assert [result["title"] for result in results] == ["page-0", "page-1"]


def test_unknown_router_never_inherits_process_parallel_capacity(monkeypatch) -> None:
    class UnknownRouter:
        registry = None
        profile = "unknown"

    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "4")
    assert game_design._request_page_worker_count(UnknownRouter(), 8) == 1
