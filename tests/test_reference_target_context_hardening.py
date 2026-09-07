from __future__ import annotations

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop


TARGET_PATH = "src/main/java/generated/example/mmmplan/TaskFeature.java"
TARGET_SYMBOL = "TaskFeature"


def _fresh_request(*, task_reuse_refs=(), binding_source_refs=()) -> dict:
    return {
        "phase": "implement_module",
        "module": {
            "module_id": "task_feature",
            "kind": "custom_java",
            "config": {
                "evidence_task": {
                    "task_id": "task_feature",
                    "semantic_outcome": "Implement the approved task feature.",
                    "reuse_refs": list(task_reuse_refs),
                    "production_bindings": [
                        {
                            "reuse_action": "fresh",
                            "source_refs": list(binding_source_refs),
                            "owned_anchors": [
                                {
                                    "kind": "symbol",
                                    "locator": f"{TARGET_PATH}#{TARGET_SYMBOL}",
                                }
                            ],
                        }
                    ],
                }
            },
        },
    }


def test_fresh_task_with_reuse_refs_fails_closed_to_localization() -> None:
    payload = _fresh_request(task_reuse_refs=("component:existing_trade_engine",))

    assert tool_loop._fresh_target_has_reuse_evidence(payload) is True
    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path is None
    assert context.is_new_file is False
    assert context.localization_stage == tool_loop.LocalizationStage.NEED_FILE
    assert context.evidence_source == "reuse_evidence_requires_localization"


def test_fresh_binding_with_source_refs_fails_closed_to_localization() -> None:
    payload = _fresh_request(
        binding_source_refs=("src/main/java/mod/TradeEngine.java",)
    )

    assert tool_loop._fresh_target_has_reuse_evidence(payload) is True
    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path is None
    assert context.is_new_file is False
    assert context.localization_stage == tool_loop.LocalizationStage.NEED_FILE
    assert context.evidence_source == "reuse_evidence_requires_localization"
