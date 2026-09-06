from __future__ import annotations

from minecraft_mod_ai.task_execution_classification import claims_runtime


def test_registry_outcome_text_does_not_claim_runtime_capability() -> None:
    assert not claims_runtime(
        {
            "semantic_outcome": "Establish the registry identity for the authored block.",
            "provides": ["registry_id:demo:block"],
        }
    )


def test_capability_export_is_the_runtime_authority() -> None:
    assert claims_runtime(
        {
            "semantic_outcome": "Verify the complete semantic outcome.",
            "provides": ["capability:space_travel", "requirement_done:req_space"],
        }
    )
