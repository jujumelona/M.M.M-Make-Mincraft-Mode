from __future__ import annotations

import json
from threading import RLock

from minecraft_mod_ai import planning_criterion_fragments as fragments
from minecraft_mod_ai import planning_state_adaptive_implementation as adaptive


class _BatchRouter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(
            {
                "criterion_fragments": [
                    {
                        "criterion_index": 0,
                        "section_updates": [
                            {
                                "section": "behavior_contract",
                                "implementation": "implement criterion zero",
                                "constraint": "reject invalid zero state",
                                "evidence_refs": [],
                            }
                        ],
                    },
                    {
                        "criterion_index": 1,
                        "section_updates": [
                            {
                                "section": "behavior_contract",
                                "implementation": "implement criterion one",
                                "constraint": "reject invalid one state",
                                "evidence_refs": [],
                            }
                        ],
                    },
                ]
            }
        )


def test_requirement_batch_generates_multiple_criteria_in_one_model_call():
    router = _BatchRouter()
    result = fragments.generate_criterion_fragments_batch(
        router,
        requirement={"statement": "Build a ship."},
        criteria={0: "Part is purchasable.", 1: "Part is persisted."},
        selected_sections=adaptive.WORKSHEET_SECTIONS,
        evidence=[],
        allowed_refs=set(),
    )
    assert router.calls == 1
    assert set(result) == {0, 1}


def test_compile_criterion_single_flights_sibling_criteria(monkeypatch):
    calls = 0

    def generated(_router, **kwargs):
        nonlocal calls
        calls += 1
        return {
            index: {
                "section_updates": [
                    {
                        "section": "behavior_contract",
                        "implementation": f"implementation {index}",
                        "constraint": "",
                        "evidence_refs": [],
                    }
                ]
            }
            for index in kwargs["criteria"]
        }

    monkeypatch.setattr(adaptive, "generate_criterion_fragments_batch", generated)
    cache = {}
    lock = RLock()
    common = dict(
        router=object(),
        requirement={"statement": "Build a ship."},
        requirement_ref="req_1",
        selected_sections=adaptive.WORKSHEET_SECTIONS,
        evidence=[],
        allowed_refs=set(),
        criteria=("criterion zero", "criterion one"),
        batch_pending_indices=(0, 1),
        batch_cache=cache,
        batch_lock=lock,
    )
    first = adaptive._compile_criterion(
        criterion_index=0,
        criterion="criterion zero",
        **common,
    )
    second = adaptive._compile_criterion(
        criterion_index=1,
        criterion="criterion one",
        **common,
    )
    assert calls == 1
    assert first != second
    assert set(cache) == {0, 1}
