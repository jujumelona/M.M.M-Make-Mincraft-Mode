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


def test_identical_user_phrases_in_two_source_positions_are_distinct_anchors():
    original = "collect; collect"
    router = PageRouter([
        _page(intent="Collect", quote="collect"),
        _page(intent="Collect", quote="collect"),
        _page(status="complete"),
    ])
    result = author_behavior_requirements(_scope(original), router)
    assert [row.source_quote for row in result] == ["collect", "collect"]
    assert result[0].behavior_id != result[1].behavior_id
    assert all(row.source_status == "grounded" for row in result)


def test_shared_scope_writes_different_requirements_without_stale_map_conflicts(monkeypatch, tmp_path):
    from minecraft_mod_ai.construction_behavior_checkpoint import BehaviorPlanningCheckpoint
    from minecraft_mod_ai.construction_task import BehaviorRequirement

    monkeypatch.setenv("MMM_CONSTRUCTION_BEHAVIOR_CACHE_ROOT", str(tmp_path))
    base = _scope("Collect ore; trade ore")
    base["requirements"] = [
        {"requirement_id": "req-collect", "statement": "Collect ore"},
        {"requirement_id": "req-trade", "statement": "Trade ore"},
    ]
    first, second = base["requirements"]
    router = PageRouter([])
    a = BehaviorPlanningCheckpoint(base, router)
    b = BehaviorPlanningCheckpoint(base, router)  # Two snapshots precede either write.
    collected = BehaviorRequirement(
        "behavior-collect", "req-collect", "Collect ore", "Collect ore",
        "Player mines", "Ore gained",
    )
    traded = BehaviorRequirement(
        "behavior-trade", "req-trade", "Trade ore", "Trade ore",
        "Player trades", "Currency gained",
    )
    with a.claim():
        a.save(first, [collected])
    with b.claim():
        b.save(second, [traded])
    c = BehaviorPlanningCheckpoint(base, router)
    assert [x.behavior_id for x in c.load(first)] == ["behavior-collect"]
    assert [x.behavior_id for x in c.load(second)] == ["behavior-trade"]
    assert len(list((tmp_path / "v3").rglob("*.json"))) == 2


def test_v2_checkpoint_is_migrated_without_model_requests(monkeypatch, tmp_path):
    from minecraft_mod_ai.construction_behavior_checkpoint import (
        BehaviorPlanningCheckpoint, _digest,
    )
    from minecraft_mod_ai.construction_task import BehaviorRequirement
    import json

    monkeypatch.setenv("MMM_CONSTRUCTION_BEHAVIOR_CACHE_ROOT", str(tmp_path))
    scope = _scope("Collect ore")
    parent = scope["requirements"][0]["requirement_id"]
    behavior = BehaviorRequirement(
        "behavior-existing", parent, "Collect ore", "Collect ore",
        "Player mines", "Ore gained",
    )
    old_identity = _digest({
        "schema": "mmm/behavior-plan-checkpoint-v2",
        "prompt": scope["requested_prompt"],
        "requirements": scope["requirements"],
        "model_id": "",
    })
    rows = {parent: [behavior.to_dict()]}
    legacy = {
        "schema": "mmm/behavior-plan-checkpoint-v2",
        "identity": old_identity,
        "rows": rows,
        "digest": _digest(rows),
    }
    old_path = tmp_path / (old_identity + ".json")
    old_path.write_text(json.dumps(legacy), encoding="utf-8")
    router = PageRouter([])
    restored = bind_behavior_scope(scope, router)
    assert router.calls == 0
    assert restored["behavior_requirements"] == rows[parent]
    assert json.loads(old_path.read_text(encoding="utf-8")) == legacy
    v3 = BehaviorPlanningCheckpoint(scope, router)
    assert v3.load(scope["requirements"][0])[0].behavior_id == "behavior-existing"


def test_behavior_checkpoint_detects_real_accepted_record_conflict(monkeypatch, tmp_path):
    from minecraft_mod_ai.construction_behavior_checkpoint import BehaviorPlanningCheckpoint
    from minecraft_mod_ai.construction_task import BehaviorRequirement

    monkeypatch.setenv("MMM_CONSTRUCTION_BEHAVIOR_CACHE_ROOT", str(tmp_path))
    scope = _scope("Collect ore")
    requirement = scope["requirements"][0]
    checkpoint = BehaviorPlanningCheckpoint(scope, PageRouter([]))
    a = BehaviorRequirement("behavior-a", "req-one", "Mine ore", "Collect ore", "Mine", "Ore")
    b = BehaviorRequirement("behavior-b", "req-one", "Mine ore", "Collect ore", "Mine", "Ore")
    with checkpoint.claim():
        checkpoint.save(requirement, [a])
        checkpoint.save(requirement, [a])
        with pytest.raises(ValueError, match="DIFFERENT_ACCEPTED_RECORD"):
            checkpoint.save(requirement, [b])
    assert checkpoint.load(requirement)[0].behavior_id == "behavior-a"
