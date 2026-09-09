from __future__ import annotations

from threading import RLock
from types import SimpleNamespace

from minecraft_mod_ai.model_output_atomicity_contract import install
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS
from minecraft_mod_ai import planning_criterion_fragments as fragments
from minecraft_mod_ai import planning_state_adaptive_implementation as adaptive


class _BatchRouter:
    def __init__(self) -> None:
        self.calls = 0
        self.tool_names: list[str] = []

    def generate_text(self, *_args, **_kwargs):
        raise AssertionError("planner batch must not use raw structured text generation")

    def generate_tool_decision(self, *_args, tool_name, **_kwargs):
        self.calls += 1
        self.tool_names.append(tool_name)
        return {
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


install(model_router_module=SimpleNamespace(ModelRouter=_BatchRouter))


def test_requirement_batch_generates_multiple_criteria_in_one_model_call():
    router = _BatchRouter()
    result = fragments.generate_criterion_fragments_batch(
        router,
        requirement={"statement": "Build a ship."},
        criteria={0: "Part is purchasable.", 1: "Part is persisted."},
        selected_sections=WORKSHEET_SECTIONS,
        evidence=[],
        allowed_refs=set(),
    )
    assert router.calls == 1
    assert router.tool_names == ["submit_criterion_fragments"]
    assert set(result) == {0, 1}


def test_compile_criterion_single_flights_sibling_criteria():
    router = _BatchRouter()
    cache = {}
    lock = RLock()
    common = dict(
        router=router,
        requirement={"statement": "Build a ship."},
        requirement_ref="req_1",
        selected_sections=WORKSHEET_SECTIONS,
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
    assert router.calls == 1
    assert router.tool_names == ["submit_criterion_fragments"]
    assert first != second
    assert set(cache) == {0, 1}
