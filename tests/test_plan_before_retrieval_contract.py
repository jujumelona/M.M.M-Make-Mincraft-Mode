from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import evidence_request_guard as request_guard
from minecraft_mod_ai import game_design
from minecraft_mod_ai import platform_central_ai_contract as platform_contract
from minecraft_mod_ai import platform_live_discovery as live
from minecraft_mod_ai import reuse_planner as reuse
from minecraft_mod_ai.evidence_first_planning import build_request_catalog
from minecraft_mod_ai.game_design import GameDesignPlanner


def _design(catalog: dict[str, object]) -> dict[str, object]:
    raw_requirements = catalog.get("requirements", [])
    requirement_ids = [
        str(item.get("requirement_id") or "")
        for item in raw_requirements
        if isinstance(item, dict) and str(item.get("requirement_id") or "")
    ]
    return {
        "title": "Plan First",
        "pitch": "Build the authored gameplay requirement.",
        "core_loop": ["Exercise the authored gameplay behavior."],
        "progression": ["Preserve the authored requirement through implementation."],
        "combat": {},
        "mod_context": {},
        "modules": [
            {
                "plugin_id": "authored_behavior",
                "status": "custom",
                "reason": "Own the exact authored gameplay behavior.",
                "requirement_refs": requirement_ids,
                "implementation_obligations": [
                    "Implement the approved observable behavior without changing scope."
                ],
            }
        ],
        "assets": [],
        "acceptance_tests": ["The approved authored behavior is observable in Minecraft."],
    }


def _catalog(prompt: str) -> dict[str, object]:
    return build_request_catalog(prompt, {})


class _Router:
    pass


class _Selection:
    migration_requested = False
    optimization = SimpleNamespace(
        evidence=SimpleNamespace(deep_research={"status": "ok"})
    )

    def __init__(self, adapter: object) -> None:
        self.adapter = adapter

    def to_dict(self) -> dict[str, object]:
        return {
            "target": {
                "minecraft_version": "mmm-test-target",
                "loader": "fabric",
                "mappings": "mmm-test-target+test-mappings",
                "source_api_family": "fabric_reviewed_test_template",
            },
            "migration_requested": False,
        }


def test_installed_platform_search_consumes_authoritative_plan_first(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_platform_lock: object,
) -> None:
    prompt = "Add persistent player trading."
    authoritative = _catalog(prompt)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        request_guard,
        "build_authoritative_request_catalog",
        lambda supplied, router=None: (
            authoritative
            if supplied == prompt
            else pytest.fail("request guard received a different prompt")
        ),
    )
    monkeypatch.setattr(
        game_design,
        "_generate_game_design_once",
        lambda *_args, **_kwargs: _design(authoritative),
    )

    def resolve_platform(_prompt: str, *, design: dict[str, object], **_kwargs: object):
        observed["design"] = design
        observed["graph"] = reuse.decompose_capability_graph(prompt, design=design)
        adapter = SimpleNamespace(
            minecraft_version="mmm-test-target",
            loader="fabric",
            yarn_mappings="mmm-test-target+test-mappings",
        )
        return _Selection(adapter)

    monkeypatch.setattr(platform_contract, "resolve_platform", resolve_platform)
    monkeypatch.setattr(
        platform_contract,
        "retarget_proposal",
        lambda proposal, _selection: replace(
            proposal,
            spec=replace(proposal.spec, platform=synthetic_platform_lock),
            approval_hash="",
        ).with_hash(),
    )

    design, _proposal = GameDesignPlanner(_Router()).plan(prompt)

    search_input = observed["design"]
    assert isinstance(search_input, dict)
    assert search_input["_evidence_request_catalog"] == authoritative
    frozen = search_input["_pre_retrieval_plan"]
    assert isinstance(frozen, dict)
    assert frozen["request_catalog_sha256"] == authoritative["catalog_sha256"]
    assert frozen["planned_work"]
    search_graph = observed["graph"]
    assert isinstance(search_graph, reuse.CapabilityGraph)
    assert search_graph.source_plan_sha256 == frozen["plan_sha256"]
    graph_payload = search_graph.to_dict()
    graph_payload.pop("source_plan_sha256")
    assert graph_payload == frozen["capability_graph"]
    assert design["_pre_retrieval_plan"] == frozen


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
        module_kinds=("unrelated_module",),
        semantic_router=_ExplodingRouter(),
    )

    assert graph.source_plan_sha256 == frozen["plan_sha256"]
    graph_payload = graph.to_dict()
    graph_payload.pop("source_plan_sha256")
    assert graph_payload == frozen["capability_graph"]


def test_tampered_pre_retrieval_plan_fails_before_search() -> None:
    prompt = "Add persistent player trading."
    design: dict[str, object] = {"_evidence_request_catalog": _catalog(prompt)}
    frozen = reuse.compile_pre_retrieval_plan(prompt, design)
    frozen["planned_work"][0]["objective"] = "silently changed after approval"
    design["_pre_retrieval_plan"] = frozen

    with pytest.raises(ValueError, match="hash mismatch"):
        reuse.decompose_capability_graph(prompt, design=design)


def test_pack_versions_use_mojang_primary_then_bounded_official_fallback(
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


def test_live_target_resolution_has_a_bounded_default_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MMM_PLATFORM_CANDIDATE_LIMIT", raising=False)
    assert reuse._target_candidate_limit() == 8

    monkeypatch.setenv("MMM_PLATFORM_CANDIDATE_LIMIT", "999")
    assert reuse._target_candidate_limit() == 32
