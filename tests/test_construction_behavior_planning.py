"""Regression tests for request-source anchored behavior construction.

The motivating real-model failure normalized consecutive spaces in a user
quote. These tests assert provenance, not a gameplay-specific special case.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai.construction_planner import (
    author_behavior_requirements,
    bind_behavior_scope,
    plan_construction_tasks,
)
from minecraft_mod_ai.request_scope import _source_quote_bounds


def _page(*, intent="", quote="", status="behavior"):
    return {
        "status": status,
        "intent": intent,
        "source_quote": quote,
        "trigger": "Player action" if intent else "",
        "effects": "Requested state changes" if intent else "",
        "state_reads": [],
        "state_writes": [],
    }


class PageRouter:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = 0

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        assert role == "planner"
        assert data_name == "construction_behavior_next"
        assert "source_text" in messages[-1]["content"]
        self.calls += 1
        return next(self.pages)


def _scope(source):
    return {
        "schema_version": "mmm/request-scope-v1",
        "requested_prompt": source,
        "requirements": [{
            "requirement_id": "req-one",
            "statement": "Compact paraphrase of the original requirement",
            "source_span": {"text": source, "start": 0, "end": len(source)},
            "implementation_role": "gameplay",
            "source_recovery": True,
        }],
    }


def test_quote_resolves_exact_unicode_source_slice_even_when_model_collapses_whitespace():
    source = "부품을  구입하고\t추가 기능을 설치한다"
    quote = "부품을 구입하고 추가 기능을 설치한다"
    bounds = _source_quote_bounds(source, quote)
    assert bounds == (0, len(source))
    router = PageRouter([_page(intent="Purchase parts", quote=quote), _page(status="complete")])
    saved = bind_behavior_scope(_scope(source), router)
    assert saved["behavior_requirements"][0]["source_quote"] == source
    assert saved["requirements"][0]["source_span"]["text"] == source
    assert router.calls == 2


def test_production_scope_reuses_saved_behavior_without_more_model_calls():
    source = "Gather resources; trade for spacecraft parts."
    router = PageRouter([
        _page(intent="Gather materials", quote="Gather resources"),
        _page(intent="Buy parts", quote="trade for spacecraft parts"),
        _page(status="complete"),
    ])
    scoped = bind_behavior_scope(_scope(source), router)
    assert len(scoped["behavior_requirements"]) == 2
    assert len({b["behavior_id"] for b in scoped["behavior_requirements"]}) == 2
    assert router.calls == 3
    replay = bind_behavior_scope(deepcopy(scoped), None)
    assert replay == scoped
    tasks = plan_construction_tasks(replay, (), {}, None)
    ids = {task.task_id for task in tasks}
    for behavior in scoped["behavior_requirements"]:
        assert behavior["behavior_id"] + ".logic" in ids
        assert behavior["behavior_id"] + ".verify" in ids


def test_unanchored_model_answer_preserves_original_but_does_not_claim_grounded():
    scope = _scope("Collect minerals")
    router = PageRouter([_page(intent="Unrequested power", quote="Install a fictional device")])
    result = author_behavior_requirements(scope, router)
    assert len(result) == 1
    assert result[0].source_quote == "Collect minerals"
    assert result[0].source_status == "source_preserved_unresolved"
    assert result[0].trigger == ""


def test_repeated_english_intent_with_distinct_source_spans_is_valid():
    scope = _scope("Gather resources; gather more resources")
    router = PageRouter([
        _page(intent="Gather resources", quote="Gather resources"),
        _page(intent="Gather resources", quote="gather more resources"),
        _page(status="complete"),
    ])
    result = author_behavior_requirements(scope, router)
    assert len(result) == 2
    assert result[0].behavior_id != result[1].behavior_id
    assert all(row.source_status == "grounded" for row in result)


def test_duplicate_model_action_stops_without_requery_or_losing_remaining_original_text():
    source = "Collect minerals; trade minerals for fuel"
    router = PageRouter([
        _page(intent="Collect", quote="Collect minerals"),
        _page(intent="Collect", quote="Collect minerals"),
    ])
    result = author_behavior_requirements(_scope(source), router)
    assert router.calls == 2
    assert len([r for r in result if r.source_status == "grounded"]) == 1
    unhandled = [r for r in result if r.source_status == "source_preserved_unresolved"]
    assert unhandled
    assert any("trade minerals for fuel" in r.source_quote for r in unhandled)


def test_full_span_duplicate_is_a_single_action_not_an_exception():
    source = "Collect and trade minerals"
    router = PageRouter([
        _page(intent="Collect and trade minerals", quote=source),
        _page(intent="Collect and trade minerals", quote=source),
    ])
    result = author_behavior_requirements(_scope(source), router)
    assert router.calls == 2
    assert len(result) == 1
    assert result[0].source_status == "grounded"


def test_new_production_run_reuses_completed_behavior_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_CONSTRUCTION_BEHAVIOR_CACHE_ROOT", str(tmp_path))
    original = _scope("Collect minerals and build a ship")
    router = PageRouter([
        _page(intent="Build a ship after mining", quote=original["requested_prompt"]),
        _page(status="complete"),
    ])
    first = bind_behavior_scope(original, router)
    assert router.calls == 2
    no_more_model_calls = PageRouter([])
    second = bind_behavior_scope(original, no_more_model_calls)
    assert second == first
    assert no_more_model_calls.calls == 0


def test_unresolved_source_cannot_be_marked_runtime_verified_even_with_pass_receipt():
    from minecraft_mod_ai.construction_coverage import requirement_completion

    receipt = {
        "requirement_id": "req-one",
        "source_sha256": "source-digest",
        "behaviors": [
            {"behavior_id": "grounded", "source_status": "grounded"},
            {"behavior_id": "unresolved", "source_status": "source_preserved_unresolved"},
        ],
    }
    events = [
        {"behavior_id": bid, "source_sha256": "source-digest",
         "status": "PASS", "trigger_observed": True,
         "effect_observed": True, "state_observed": True}
        for bid in ("grounded", "unresolved")
    ]
    result = requirement_completion(receipt, events)
    assert result["status"] == "SOURCE_REVIEW_REQUIRED"
    assert result["source_review_required_ids"] == ["unresolved"]
