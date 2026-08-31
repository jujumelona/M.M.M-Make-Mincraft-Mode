from __future__ import annotations

import json

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop


TASK_ID = "task_alien_planet_interaction_semantic_im_47278ef7e7"
TARGET_PATH = (
    "src/main/java/generated/alienmod/mmmplan/"
    "TaskAlienPlanetInteractionSemanticIm47278ef7e7.java"
)
TARGET_SYMBOL = "TaskAlienPlanetInteractionSemanticIm47278ef7e7"


def _request(*, action: str = "fresh", include_incidental_source: bool = False) -> dict:
    task = {
        "task_id": TASK_ID,
        "owned_anchors": [
            {
                "kind": "symbol",
                "locator": f"{TARGET_PATH}#{TARGET_SYMBOL}",
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "main",
            },
            {
                "kind": "test",
                "locator": (
                    "src/test/java/generated/alienmod/mmmplan/"
                    f"{TARGET_SYMBOL}Test.java#{TARGET_SYMBOL}Test"
                ),
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "main",
            },
        ],
        "production_bindings": [
            {
                "task_ref": TASK_ID,
                "reuse_action": action,
                "owned_anchors": [
                    {
                        "kind": "symbol",
                        "locator": f"{TARGET_PATH}#{TARGET_SYMBOL}",
                        "ownership": "exclusive",
                        "status": "host_reserved",
                        "module_id": ":",
                        "source_set": "main",
                    }
                ],
            }
        ],
    }
    request = {
        "phase": "implement_module",
        "task": "Implement the approved Minecraft/Fabric mod feature in the current project.",
        "module": {
            "module_id": TASK_ID,
            "kind": "custom_java",
            "config": {"evidence_task": task},
        },
    }
    if include_incidental_source:
        request["initial_exact_source_context"] = {
            "records": [
                {
                    "path": "src/main/resources/fabric.mod.json",
                    "content": '{"schemaVersion": 1, "id": "alienmod"}',
                },
                {
                    "path": "src/main/java/generated/alienmod/Existing.java",
                    "content": "package generated.alienmod; public final class Existing {}",
                },
            ]
        }
    return request


def test_fresh_evidence_task_uses_host_reserved_symbol_as_new_file() -> None:
    context = tool_loop._extract_mutation_context_from_payload(_request())

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.target_symbol == TARGET_SYMBOL
    assert context.is_new_file is True
    assert context.localization_stage == tool_loop.LocalizationStage.READY
    assert context.evidence_source == "evidence_fresh_owned_anchor"


def test_active_extractor_prioritizes_fresh_anchor_over_incidental_project_context() -> None:
    context = tool_loop._extract_mutation_context_from_payload(
        _request(include_incidental_source=True)
    )

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.target_path != "src/main/resources/fabric.mod.json"
    assert context.is_new_file is True
    assert context.localization_stage == tool_loop.LocalizationStage.READY


def test_output_continuation_can_recover_fresh_target_from_module_receipt_alone() -> None:
    continuation = _request()
    continuation["task"] = (
        "Continue the approved module from the preserved staged workspace; "
        "do not restart completed work."
    )
    continuation["continuation"] = {
        "reason": "previous_tool_enabled_page_exhausted_output",
        "continuation_index": 1,
        "preserved_path_count": 0,
        "preserved_paths_preview": [],
    }
    state = tool_loop.HostRunState()
    messages = [{"role": "user", "content": json.dumps(continuation)}]

    assert tool_loop.is_mutation_ready(messages, state) is True
    assert state.mutation_context is not None
    assert state.mutation_context.target_path == TARGET_PATH
    assert state.mutation_context.localization_stage == tool_loop.LocalizationStage.READY


def test_adapt_task_still_requires_existing_source_localization() -> None:
    context = tool_loop._extract_mutation_context_from_payload(_request(action="adapt"))
    assert context is None or context.is_new_file is False
