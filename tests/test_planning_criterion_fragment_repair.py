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


def _generate(router: _SequenceRouter) -> dict[str, object]:
    return generate_criterion_fragment(
        router,
        requirement={"statement": "Collected resources enter the player inventory."},
        criterion="Collected items are added to the player's inventory.",
        selected_sections=CORE_WORKSHEET_SECTIONS,
        evidence=[],
        allowed_refs=set(),
    )


def test_legacy_prose_response_is_rejected_without_rewriting_or_retry() -> None:
    from minecraft_mod_ai.structured_output import StructuredOutputValidationError
    router = _SequenceRouter([{"section_updates": [{"section": "behavior_contract",
        "implementation": "Add the collected stack to inventory.", "constraint": "",
        "evidence_refs": []}]}])
    with pytest.raises(StructuredOutputValidationError):
        _generate(router)
    assert len(router.calls) == 1


def test_legacy_status_response_is_rejected_without_retry() -> None:
    from minecraft_mod_ai.structured_output import StructuredOutputValidationError
    router = _SequenceRouter([{"status": "done", "record": None, "reason": "", "evidence_refs": []}])
    with pytest.raises(StructuredOutputValidationError):
        _generate(router)
    assert len(router.calls) == 1


def test_non_no_progress_tool_contract_error_is_not_retried() -> None:
    router = _SequenceRouter([ValueError("forced template transport failed")])

    with pytest.raises(ValueError, match="forced template transport failed"):
        _generate(router)

    assert len(router.calls) == 1


def test_omitted_required_sections_are_not_inferred_inapplicable() -> None:
    from minecraft_mod_ai.planning_criterion_fragments import (
        MissingWorksheetSections, assemble_worksheet_from_fragments,
    )

    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
    section = "behavior_contract"
    specification = {concern: [{field: f"authored {field}" for field in fields.split()}]
                     for concern, fields in DETAIL_RECORDS[section].items()}
    specification["inapplicable_concerns"] = []
    fragment = {"section_updates": [{"section": section, "specification": specification,
                                     "constraint_evidence_refs": []}]}
    with pytest.raises(MissingWorksheetSections) as caught:
        assemble_worksheet_from_fragments(
            {"statement": "Mine resources and credit currency."},
            selected_sections=CORE_WORKSHEET_SECTIONS,
            criteria=("Mining credits currency.",),
            fragments={0: fragment},
            allowed_refs=set(),
        )
    assert "reuse_assessment" in caught.value.sections
    assert "verification" in caught.value.sections
