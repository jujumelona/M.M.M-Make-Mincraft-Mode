from __future__ import annotations

import json
from types import SimpleNamespace

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


def test_task_local_envelope_prioritizes_java_anchor_over_incidental_resource() -> None:
    """Regression for the production trace that pinned a mixins JSON instead of Java."""

    request = _request(include_incidental_source=True)
    evidence_task = request["module"].pop("config")["evidence_task"]
    request["module"]["evidence_task"] = evidence_task
    request["initial_exact_source_context"] = {
        "records": [
            {
                "path": "src/main/resources/generated_mod.mixins.json",
                "content": '{"required":true,"mixins":[]}',
            }
        ]
    }

    context = tool_loop._extract_mutation_context_from_payload(request)

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.target_path != "src/main/resources/generated_mod.mixins.json"
    assert context.is_new_file is True


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


def test_task_local_output_continuation_recovers_fresh_target() -> None:
    continuation = _request()
    evidence_task = continuation["module"].pop("config")["evidence_task"]
    continuation["module"]["evidence_task"] = evidence_task
    continuation["continuation"] = {
        "reason": "previous_tool_enabled_page_exhausted_output",
        "continuation_index": 1,
        "preserved_path_count": 1,
        "preserved_paths_preview": ["src/main/resources/generated_mod.mixins.json"],
    }
    state = tool_loop.HostRunState()

    assert tool_loop.is_mutation_ready(
        [{"role": "user", "content": json.dumps(continuation)}], state
    ) is True
    assert state.mutation_context is not None
    assert state.mutation_context.target_path == TARGET_PATH
    assert state.mutation_context.target_pinned is True


def test_adapt_task_still_requires_existing_source_localization() -> None:
    context = tool_loop._extract_mutation_context_from_payload(_request(action="adapt"))
    assert context is None or context.is_new_file is False


def test_existing_pinned_target_refreshes_from_workspace(tmp_path) -> None:
    target = tmp_path / TARGET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "package generated.alienmod.mmmplan; public final class "
        + TARGET_SYMBOL
        + " {}\n",
        encoding="utf-8",
    )
    state = tool_loop.HostRunState()
    state.mutation_context = tool_loop.TargetMutationContext(
        target_path=TARGET_PATH,
        target_symbol=TARGET_SYMBOL,
        source_body="stale source",
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(TARGET_PATH,),
        target_pinned=True,
    )

    refreshed = tool_loop._reconcile_materialized_target_from_workspace(
        state, SimpleNamespace(workspace_root=str(tmp_path))
    )

    assert refreshed is not None
    assert refreshed.is_new_file is False
    assert refreshed.source_body == target.read_text(encoding="utf-8")
    assert state.mutation_context == refreshed


def test_exact_edit_precondition_failure_has_mutation_specific_code() -> None:
    code = tool_loop._runtime_failure_code(
        "apply_source_edit",
        "AgentToolRuntimeError: Exact source-edit precondition failed for "
        + TARGET_PATH
        + ": expected 1 matches, found 0",
    )
    assert code == "MUTATION_STALE_PRECONDITION"


def test_fixed_point_call_identity_ignores_volatile_source_payload() -> None:
    first = SimpleNamespace(
        name="apply_source_edit",
        arguments={
            "operation": "replace_exact",
            "path": TARGET_PATH,
            "old": "old source one",
            "new": "new source one",
        },
    )
    second = SimpleNamespace(
        name="apply_source_edit",
        arguments={
            "operation": "replace_exact",
            "path": TARGET_PATH,
            "old": "different stale source",
            "new": "different replacement",
        },
    )

    assert tool_loop._fixed_point_tool_calls((first,)) == tool_loop._fixed_point_tool_calls(
        (second,)
    )


def test_no_progress_streak_counts_consecutive_failures() -> None:
    state = tool_loop.HostRunState()
    value = {
        "phase": "ACT",
        "calls": [{"name": "apply_source_edit", "path": TARGET_PATH}],
        "results": [
            {
                "name": "apply_source_edit",
                "ok": False,
                "failure_code": "MUTATION_STALE_PRECONDITION",
            }
        ],
    }

    assert state.record_no_progress_result(value) is False
    assert state.no_progress_streak == 1
    assert state.record_no_progress_result(value) is True
    assert state.no_progress_streak == 2
