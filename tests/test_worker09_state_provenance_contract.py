from __future__ import annotations

from dataclasses import replace

import pytest

from minecraft_mod_ai import complete_spec, spec
from minecraft_mod_ai import design_resolution_provenance_contract as provenance
from minecraft_mod_ai.proposal_deserialization_contract import install


install(spec, complete_spec)


def _minimal_base_proposal() -> spec.Proposal:
    return spec.Proposal(
        schema_version="minecraft-mod-ai/proposal-v1",
        proposal_version=1,
        status=spec.ProposalStatus.AWAITING_APPROVAL,
        requested_prompt="test",
        spec=spec.ModSpec(
            mod_id="test_mod",
            mod_name="Test Mod",
            package_name="com.example.testmod",
            version="1.0.0",
            summary="test",
            contents=(),
        ),
        assumptions=(),
        exclusions=(),
        deferred_requests=(),
        acceptance_tests=("works",),
        evidence_sources=(),
        evidence_snapshot_hash="sha256:" + "1" * 64,
        capability_manifest_hash="sha256:" + "2" * 64,
        imported_source_snapshot_hash="",
        risk_approvals=(),
        approval_hash="",
    )


def test_base_approval_binds_supplied_integrity_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = _minimal_base_proposal()
    monkeypatch.setattr(spec.Proposal, "validate", lambda self: None)

    expected = proposal.calculate_hash()
    approved = proposal.approve(expected)

    assert approved.status is spec.ProposalStatus.APPROVED
    assert approved.approval_hash == expected


def test_complete_approval_binds_supplied_integrity_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _minimal_base_proposal()
    proposal = complete_spec.CompleteProposal(
        schema_version="mmm/complete-proposal-v1",
        proposal_version=1,
        status=complete_spec.CompleteProposalStatus.AWAITING_APPROVAL,
        requested_prompt="test",
        base_proposal=base,
        game_design={"goal": "test"},
        modules=(),
        assets=(),
        acceptance_tests=("works",),
        external_runtime_required=True,
        existing_input_sha256="",
        approval_hash="",
    )
    monkeypatch.setattr(
        complete_spec.CompleteProposal,
        "validate",
        lambda self, **kwargs: None,
    )

    expected = proposal.calculate_hash()
    approved = proposal.approve(expected)

    assert approved.status is complete_spec.CompleteProposalStatus.APPROVED
    assert approved.approval_hash == expected


def test_approved_state_without_receipt_fails_closed() -> None:
    proposal = replace(
        _minimal_base_proposal(),
        status=spec.ProposalStatus.APPROVED,
        approval_hash="",
    )

    with pytest.raises(
        spec.SpecValidationError,
        match="approved state requires its approval_hash",
    ):
        proposal.validate()


def test_resume_does_not_rebind_missing_provenance_to_current_runtime() -> None:
    payload = {
        "schema_version": "minecraft-mod-ai/proposal-v1",
        "proposal_version": 1,
        "status": "awaiting_user_approval",
        "requested_prompt": "test",
        "approval_hash": "",
    }

    with pytest.raises(
        spec.SpecValidationError,
        match="missing authoritative provenance receipts",
    ):
        spec.Proposal.from_dict(payload)


def _evaluation() -> dict:
    return {
        "candidates": [
            {"candidate_id": "a", "summary": "A"},
            {"candidate_id": "b", "summary": "B"},
        ],
        "selected_candidate_id": "b",
        "evidence_refs": ["evidence:official"],
        "requirement_refs": ["req_trade"],
    }


def test_design_selection_requires_explicit_content_bound_receipt() -> None:
    evaluation = _evaluation()
    plan = {"design_alternative_evaluations": [evaluation]}

    assert provenance._explicit_selected_alternatives(plan) == []

    evaluation["evaluation_sha256"] = "sha256:" + "0" * 64
    assert provenance._explicit_selected_alternatives(plan) == []

    evaluation["evaluation_sha256"] = provenance._evaluation_receipt_hash(evaluation)
    selected = provenance._explicit_selected_alternatives(plan)

    assert len(selected) == 1
    assert selected[0]["selection"] == "b"
    assert selected[0]["selection_receipt_sha256"] == evaluation["evaluation_sha256"]


def test_valid_hash_is_integrity_not_semantic_correctness() -> None:
    evaluation = _evaluation()
    evaluation["evidence_refs"] = []
    evaluation["evaluation_sha256"] = provenance._evaluation_receipt_hash(evaluation)

    assert provenance._explicit_selected_alternatives(
        {"design_alternative_evaluations": [evaluation]}
    ) == []
