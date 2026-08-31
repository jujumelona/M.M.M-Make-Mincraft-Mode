from __future__ import annotations

import pytest

from minecraft_mod_ai import complete_spec, spec
from minecraft_mod_ai.proposal_deserialization_contract import install


def _complete_payload() -> dict:
    return {
        "schema_version": "mmm/complete-proposal-v1",
        "proposal_version": 1,
        "status": "awaiting_user_approval",
        "requested_prompt": "test",
        "base_proposal": {},
        "game_design": {},
        "modules": [],
        "assets": [],
        "acceptance_tests": ["works"],
        "external_runtime_required": True,
        "existing_input_sha256": "",
        "approval_hash": "",
    }


def test_complete_proposal_does_not_stringify_non_string_acceptance_test() -> None:
    install(spec, complete_spec)
    payload = _complete_payload()
    payload["acceptance_tests"] = [123]
    with pytest.raises(spec.SpecValidationError, match=r"acceptance_tests\[0\]"):
        complete_spec.CompleteProposal.from_dict(payload)


def test_complete_proposal_does_not_normalize_saved_module_identity() -> None:
    install(spec, complete_spec)
    payload = _complete_payload()
    payload["modules"] = [
        {
            "module_id": "Boss System",
            "kind": "custom_java",
            "config": {},
            "depends_on": [],
            "required_gates": [],
        }
    ]
    with pytest.raises(spec.SpecValidationError, match="already be lowercase snake_case"):
        complete_spec.CompleteProposal.from_dict(payload)


def test_base_proposal_list_fields_do_not_accept_string_iterables() -> None:
    install(spec, complete_spec)
    digest = "sha256:" + "a" * 64
    payload = {
        "schema_version": "minecraft-mod-ai/proposal-v1",
        "proposal_version": 1,
        "status": "awaiting_user_approval",
        "requested_prompt": "test",
        "spec": {
            "mod_id": "test_mod",
            "mod_name": "Test Mod",
            "package_name": "com.example.testmod",
            "version": "1.0.0",
            "summary": "test",
            "platform": {},
            "contents": [],
            "boss": None,
        },
        "assumptions": "not-a-list",
        "exclusions": [],
        "deferred_requests": [],
        "acceptance_tests": ["works"],
        "evidence_sources": [],
        "evidence_snapshot_hash": digest,
        "capability_manifest_hash": digest,
        "imported_source_snapshot_hash": digest,
        "risk_approvals": [],
        "approval_hash": "",
    }
    with pytest.raises(spec.SpecValidationError, match="assumptions must be a JSON list"):
        spec.Proposal.from_dict(payload)
