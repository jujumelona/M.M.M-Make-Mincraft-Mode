from minecraft_mod_ai import model_context_budget as context_budget
from minecraft_mod_ai import model_router as model_router_module
from minecraft_mod_ai import planning_semantic_research as semantic
from minecraft_mod_ai.semantic_tool_context_alignment_installation import (
    tool_decision_request_messages,
)


def test_semantic_admission_counts_router_mandatory_tool_context():
    requirement = "Players can trade resources to upgrade spacecraft systems."
    obligation = "Players can trade resources to upgrade spacecraft systems."
    source_id = "fixture:spacecraft"
    window = "Players trade resources to upgrade spacecraft systems. " * 48
    units = semantic._source_units(window)
    max_index = len(units) - 1

    raw_assessment = semantic._assessment_messages(
        requirement,
        obligation,
        source_id,
        units,
    )
    raw_verification = semantic._verification_messages(
        requirement,
        obligation,
        source_id,
        units,
        max_index,
        max_index,
    )
    forced_assessment = tool_decision_request_messages(
        model_router_module,
        raw_assessment,
        tool_name=semantic._ASSESSMENT_TOOL_NAME,
    )
    forced_verification = tool_decision_request_messages(
        model_router_module,
        raw_verification,
        tool_name=semantic._VERIFICATION_TOOL_NAME,
    )

    assessment_budget = context_budget._canonical_size(raw_assessment) + 1
    verification_budget = context_budget._canonical_size(raw_verification) + 1

    assert context_budget._canonical_size(raw_assessment) <= assessment_budget
    assert context_budget._canonical_size(raw_verification) <= verification_budget
    assert context_budget._canonical_size(forced_assessment) > assessment_budget
    assert context_budget._canonical_size(forced_verification) > verification_budget

    assert not semantic._semantic_window_fits(
        window,
        requirement_statement=requirement,
        obligation=obligation,
        source_id=source_id,
        assessment_budget=assessment_budget,
        verification_budget=verification_budget,
    )


def test_semantic_admission_accepts_when_full_router_envelope_fits():
    requirement = "Players can upgrade spacecraft."
    obligation = "Players can upgrade spacecraft."
    source_id = "fixture:small"
    window = "Players can upgrade spacecraft."
    units = semantic._source_units(window)
    max_index = len(units) - 1

    assessment_request = tool_decision_request_messages(
        model_router_module,
        semantic._assessment_messages(requirement, obligation, source_id, units),
        tool_name=semantic._ASSESSMENT_TOOL_NAME,
    )
    verification_request = tool_decision_request_messages(
        model_router_module,
        semantic._verification_messages(
            requirement,
            obligation,
            source_id,
            units,
            max_index,
            max_index,
        ),
        tool_name=semantic._VERIFICATION_TOOL_NAME,
    )

    assert semantic._semantic_window_fits(
        window,
        requirement_statement=requirement,
        obligation=obligation,
        source_id=source_id,
        assessment_budget=context_budget._canonical_size(assessment_request),
        verification_budget=context_budget._canonical_size(verification_request),
    )
