from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import platform_live_discovery as live
from minecraft_mod_ai import reuse_planner as reuse
from minecraft_mod_ai.evidence_first_planning import build_request_catalog


def _catalog(prompt: str) -> dict[str, object]:
    return build_request_catalog(prompt, {})


def test_pre_retrieval_plan_is_frozen_before_any_target_search() -> None:
    prompt = "Add persistent player trading."
    catalog = _catalog(prompt)
    design: dict[str, object] = {"_evidence_request_catalog": catalog}

    frozen = reuse.compile_pre_retrieval_plan(prompt, design)
    design["_pre_retrieval_plan"] = frozen
    graph = reuse.decompose_capability_graph(
        prompt,
        design=design,
        module_kinds=("unrelated.module",),
    )

    assert frozen["request_catalog_sha256"] == catalog["catalog_sha256"]
    assert frozen["planned_work"]
    assert graph.source_plan_sha256 == frozen["plan_sha256"]
    graph_payload = graph.to_dict()
    graph_payload.pop("source_plan_sha256")
    assert graph_payload == frozen["capability_graph"]


def test_frozen_semantic_plan_blocks_post_plan_redecomposition() -> None:
    prompt = "Add persistent player trading."
    design: dict[str, object] = {"_evidence_request_catalog": _catalog(prompt)}
    frozen = reuse.compile_pre_retrieval_plan(prompt, design)
    design["_pre_retrieval_plan"] = frozen

    class _ExplodingRouter:
        def generate_text(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("semantic decomposition ran again after plan approval")

    graph = reuse.decompose_capability_graph(
        prompt,
        design=design,
        module_kinds=("unrelated.module",),
        semantic_router=_ExplodingRouter(),
    )
    assert graph.source_plan_sha256 == frozen["plan_sha256"]


def test_tampered_pre_retrieval_plan_fails_before_search() -> None:
    prompt = "Add persistent player trading."
    design: dict[str, object] = {"_evidence_request_catalog": _catalog(prompt)}
    frozen = reuse.compile_pre_retrieval_plan(prompt, design)
    frozen["planned_work"][0]["objective"] = "silently changed after approval"
    design["_pre_retrieval_plan"] = frozen

    with pytest.raises(ValueError, match="hash mismatch"):
        reuse.decompose_capability_graph(prompt, design=design)


def test_platform_selection_has_no_semantic_candidate_window() -> None:
    source = (reuse.__file__ and open(reuse.__file__, encoding="utf-8").read()) or ""
    assert "MMM_PLATFORM_CANDIDATE_LIMIT" not in source
    assert "_target_candidate_limit" not in source


def test_pack_versions_use_mojang_primary_then_bounded_transport_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int | None]] = []
    payload = {
        "results": [
            {
                "title": "Minecraft Java Edition 26.2",
                "draft": False,
                "html_url": "https://feedback.minecraft.net/hc/en-us/articles/46690753273997",
                "body": (
                    "<p>The Data Pack version is now 107.1.</p>"
                    "<p>The Resource Pack version is now 88.0.</p>"
                ),
            }
        ]
    }

    def fetch(url: str, *, timeout: int = 20, retries: int | None = None) -> bytes:
        calls.append((url, timeout, retries))
        return json.dumps(payload).encode("utf-8")

    live._official_pack_versions.cache_clear()
    monkeypatch.setattr(live, "_fetch", fetch)
    try:
        assert live._official_pack_versions("26.2") == (
            "107.1",
            "88.0",
            "https://feedback.minecraft.net/hc/en-us/articles/46690753273997",
        )
    finally:
        live._official_pack_versions.cache_clear()

    assert len(calls) == 2
    assert calls[0][0].startswith("https://piston-meta.mojang.com/")
    assert calls[1][0].startswith(
        "https://feedback.minecraft.net/api/v2/help_center/articles/search.json?"
    )
    assert calls[1][1] <= 6
    assert calls[1][2] <= 2
