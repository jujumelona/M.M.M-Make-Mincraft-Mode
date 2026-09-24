from __future__ import annotations

from minecraft_mod_ai.small_model_atomic_coder_execution import (
    _bounded_initial_observations,
)



class _Index:
    def __init__(self) -> None:
        self.calls = []

    def select_page(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > 1:
            raise AssertionError("initial collector must not paginate through the project")
        return {
            "page_index": 0,
            "project_sha256": "sha256:" + "1" * 64,
            "query_sha256": "sha256:" + "2" * 64,
            "start_position": 0,
            "start_offset": 0,
            "next_cursor": "more-pages-exist",
            "complete": False,
            "files": [
                {
                    "path": "src/main/java/demo/Feature.java",
                    "sha256": "sha256:" + "3" * 64,
                    "content_start_bytes": 0,
                    "content_end_bytes": 16,
                    "content": "class Feature {}",
                }
            ],
        }


def test_atomic_initial_source_reads_only_one_ranked_page() -> None:
    index = _Index()

    ledger = _bounded_initial_observations(
        index,
        query="Feature authoritative state",
        byte_budget=32 * 1024,
    )

    assert len(index.calls) == 1
    assert index.calls[0]["cursor"] == ""
    assert index.calls[0]["byte_budget"] == 4 * 1024
    assert ledger["receipt"]["source_page_count"] == 1
    assert ledger["receipt"]["observation_count"] == 1
    assert ledger["receipt"]["policy"]["initial_page_only"] is True
    assert ledger["receipt"]["policy"]["source_page_complete"] is False
    assert ledger["records"][0]["path"] == "src/main/java/demo/Feature.java"
