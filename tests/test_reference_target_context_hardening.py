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


def test_fresh_task_reuse_refs_do_not_erase_reserved_creation_target() -> None:
    payload = _fresh_request(task_reuse_refs=("component:existing_trade_engine",))

    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.target_symbol == TARGET_SYMBOL
    assert context.is_new_file is True
    assert context.localization_stage == tool_loop.LocalizationStage.READY


def test_fresh_binding_source_refs_do_not_replace_reserved_creation_target() -> None:
    payload = _fresh_request(
        binding_source_refs=("src/main/java/mod/TradeEngine.java",)
    )

    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.target_symbol == TARGET_SYMBOL
    assert context.is_new_file is True
    assert context.localization_stage == tool_loop.LocalizationStage.READY
