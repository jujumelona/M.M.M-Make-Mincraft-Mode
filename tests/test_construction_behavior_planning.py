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
    source = "Gather resources and trade for spacecraft parts."
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


def test_unanchored_behavior_is_rejected_not_silently_fabricated():
    scope = _scope("Collect minerals")
    router = PageRouter([_page(intent="Unrequested power", quote="Install a fictional device")])
    with pytest.raises(ValueError, match="CONSTRUCTION_BEHAVIOR_UNGROUNDED"):
        author_behavior_requirements(scope, router)


def test_duplicate_intent_is_not_silently_counted_as_distinct():
    scope = _scope("Gather resources and gather more resources")
    router = PageRouter([
        _page(intent="Gather resources", quote="Gather resources"),
        _page(intent="Gather resources", quote="gather more resources"),
    ])
    with pytest.raises(ValueError, match="CONSTRUCTION_BEHAVIOR_NO_PROGRESS"):
        author_behavior_requirements(scope, router)
