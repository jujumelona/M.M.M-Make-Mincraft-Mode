"""Regression: hallucinated control input must not survive planning caches."""
from __future__ import annotations

from minecraft_mod_ai.construction_planner import author_behavior_requirements
from minecraft_mod_ai.construction_behavior_checkpoint import BehaviorPlanningCheckpoint


SOURCE = "우주로 나가서 다른 행성을 탐험하고 광물을 채집한다"


def _row():
    return {
        "behavior_id": "behavior_001",
        "requirement_id": "request_001",
        "intent": "Explore new planets",
        "source_quote": SOURCE,
        "trigger": "Space key pressed",
        "effects": "Travel to planets and mine new minerals",
        "state_reads": [],
        "state_writes": [],
    }


def _scope():
    return {"requirements": [{
        "requirement_id": "request_001",
        "statement": SOURCE,
        "source_span": {"text": SOURCE},
    }]}


def test_saved_behavior_scope_rejects_invented_keyboard_mechanic():
    scope = _scope()
    scope["behavior_requirements"] = [_row()]
    behaviors = author_behavior_requirements(scope, router=None)
    assert len(behaviors) == 1
    assert behaviors[0].trigger == ""
    assert behaviors[0].effects == _row()["effects"]
    assert behaviors[0].source_quote == SOURCE


def test_existing_checkpoint_is_sanitized_without_changing_original_requirement():
    records = BehaviorPlanningCheckpoint._verify_rows(
        _scope()["requirements"][0], [_row()])
    assert records[0].trigger == ""
    assert records[0].source_quote == SOURCE
    assert records[0].effects == _row()["effects"]


def test_new_model_behavior_is_sanitized_before_receipt():
    class Model:
        calls = 0

        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert role == "planner"
            assert data_name == "construction_behavior_next"
            self.calls += 1
            if self.calls == 1:
                return {
                    "status": "behavior",
                    "intent": "Explore new planets",
                    "source_quote": SOURCE,
                    "trigger": "Space key pressed",
                    "effects": "Travel to planets and mine new minerals",
                    "state_reads": [], "state_writes": [],
                }
            return {
                "status": "complete", "intent": "", "source_quote": "",
                "trigger": "", "effects": "", "state_reads": [], "state_writes": [],
            }

    behaviors = author_behavior_requirements(_scope(), Model())
    assert len(behaviors) == 1
    assert behaviors[0].trigger == ""
    assert behaviors[0].intent == "Explore new planets"
