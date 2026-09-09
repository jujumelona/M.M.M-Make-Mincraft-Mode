from __future__ import annotations

import pytest

from minecraft_mod_ai.planning_criterion_fragments import generate_criterion_fragment
from minecraft_mod_ai.planning_detail_template import CORE_WORKSHEET_SECTIONS


class _SequenceRouter:
    def __init__(self, outputs: list[object]) -> None:
        self._outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []

    def generate_text(self, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("criterion repair must not use raw structured text generation")

    def generate_tool_decision(
        self,
        _role: str,
        messages: list[dict[str, str]],
        **_kwargs: object,
    ) -> object:
        self.calls.append(messages)
        output = self._outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def _fragment(*, implementation: str = "", constraint: str = "") -> dict[str, object]:
    updates: list[dict[str, object]] = []
    for index, section in enumerate(CORE_WORKSHEET_SECTIONS):
        updates.append(
            {
                "section": section,
                "implementation": implementation if index == 0 else "",
                "constraint": constraint if index == 0 else "",
                "evidence_refs": [],
            }
        )
    return {"section_updates": updates}


def _generate(router: _SequenceRouter) -> dict[str, object]:
    return generate_criterion_fragment(
        router,
        requirement={"statement": "Collected resources enter the player inventory."},
        criterion="Collected items are added to the player's inventory.",
        selected_sections=CORE_WORKSHEET_SECTIONS,
        evidence=[],
        allowed_refs=set(),
    )


def test_all_empty_fragment_gets_one_corrective_call() -> None:
    router = _SequenceRouter(
        [
            _fragment(),
            _fragment(implementation="After a successful collection, the collected stack is present in inventory."),
        ]
    )

    result = _generate(router)

    assert len(router.calls) == 2
    assert result["section_updates"][0]["implementation"]
    second_user_prompt = router.calls[1][1]["content"]
    assert "Correction required" in second_user_prompt
    assert "at least one selected section concretely" in second_user_prompt


def test_second_all_empty_fragment_remains_terminal() -> None:
    router = _SequenceRouter([_fragment(), _fragment()])

    with pytest.raises(
        ValueError,
        match="DETAILED_PLAN_NO_PROGRESS: acceptance criterion produced no implementation content",
    ):
        _generate(router)

    assert len(router.calls) == 2


def test_non_no_progress_tool_contract_error_is_not_retried() -> None:
    router = _SequenceRouter([ValueError("forced template transport failed")])

    with pytest.raises(ValueError, match="forced template transport failed"):
        _generate(router)

    assert len(router.calls) == 1


def test_omitted_required_sections_are_not_inferred_inapplicable() -> None:
    from minecraft_mod_ai.planning_criterion_fragments import (
        MissingWorksheetSections, assemble_worksheet_from_fragments,
    )

    with pytest.raises(MissingWorksheetSections) as caught:
        assemble_worksheet_from_fragments(
            {"statement": "Mine resources and credit currency."},
            selected_sections=CORE_WORKSHEET_SECTIONS,
            criteria=("Mining credits currency.",),
            fragments={0: _fragment(implementation="Credit currency once after server-confirmed mining.")},
            allowed_refs=set(),
        )
    assert "reuse_assessment" in caught.value.sections
    assert "verification" in caught.value.sections
