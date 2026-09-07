from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from minecraft_mod_ai.small_model_atomic_coder_execution import (
    _bounded_initial_observations,
)


def _sha_json(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _update_digest(digest, value) -> None:
    digest.update(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _exact_observation(*, path, sha256, start, content, source_page):
    core = {
        "path": path,
        "sha256": sha256,
        "content_start_bytes": start,
        "content_end_bytes": start + len(content),
        "source_page_index": source_page,
        "kind": "exact_source_excerpt",
        "text": content.decode("utf-8"),
    }
    return {"observation_id": "obs_" + _sha_json(core).removeprefix("sha256:"), **core}


def _append_observation(records, keys, record) -> None:
    key = (
        record["path"],
        record["content_start_bytes"],
        record["content_end_bytes"],
    )
    if key not in keys:
        keys.add(key)
        records.append(record)


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
    generator_module = SimpleNamespace(
        _json_size=lambda value: len(json.dumps(value).encode("utf-8")),
        _update_digest=_update_digest,
        _append_observation=_append_observation,
        _exact_observation=_exact_observation,
        CustomModuleGenerationError=RuntimeError,
    )
    index = _Index()

    ledger = _bounded_initial_observations(
        generator_module,
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
