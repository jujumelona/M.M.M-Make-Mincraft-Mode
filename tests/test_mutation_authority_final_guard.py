from __future__ import annotations

import json

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.mutation_authority_final_guard import install

TASK_ID = "task_debug_token"
TARGET = "src/main/java/dev/mmm/DebugToken.java"
OTHER = "src/main/java/dev/mmm/MMMDebugFixture.java"


def _authority_payload() -> dict:
    primary = {
        "kind": "symbol",
        "locator": f"{TARGET}#DebugToken",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    return {
        "schema_version": "mmm/direct-task-mutation-authority-v1",
        "task_id": TASK_ID,
        "mutation_target": {
            "path": TARGET,
            "symbol": "DebugToken",
            "mode": "create_or_edit_exact_host_binding",
        },
        "module": {
            "module_id": TASK_ID,
            "kind": "custom_java",
            "config": {
                "evidence_task": {
                    "task_id": TASK_ID,
                    "owned_anchors": [primary],
                    "production_bindings": [
                        {
                            "task_ref": TASK_ID,
                            "reuse_action": "fresh",
                            "owned_anchors": [primary],
                        }
                    ],
                }
            },
        },
    }


def setup_module() -> None:
    install(tool_loop)


def test_host_authority_is_pinned_structurally_not_by_evidence_label() -> None:
    state = tool_loop.HostRunState()
    messages = [
        {"role": "developer", "content": json.dumps(_authority_payload())},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "target_file": OTHER,
                    "source": "package dev.mmm; public final class MMMDebugFixture {}",
                }
            ),
        },
    ]

    assert tool_loop.is_mutation_ready(messages, state) is True
    assert state.mutation_context is not None
    assert state.mutation_context.target_path == TARGET
    assert getattr(state.mutation_context, "target_pinned", False) is True
    assert TARGET in getattr(state.mutation_context, "writable_paths", ())


def test_retrieval_cannot_replace_pinned_target_or_expand_write_authority() -> None:
    state = tool_loop.HostRunState()
    messages = [{"role": "developer", "content": json.dumps(_authority_payload())}]
    assert tool_loop.is_mutation_ready(messages, state) is True
    before = state.mutation_context
    assert before is not None
    before_writable = tuple(getattr(before, "writable_paths", ()))
    before_creatable = tuple(getattr(before, "creatable_paths", ()))

    recorded = state.record_evidence(
        {
            "hits": [
                {
                    "path": OTHER,
                    "symbol": "MMMDebugFixture",
                    "text": "package dev.mmm; public final class MMMDebugFixture {}",
                }
            ]
        },
        usable=True,
    )

    assert recorded is True
    after = state.mutation_context
    assert after is not None
    assert after.target_path == TARGET
    assert getattr(after, "target_pinned", False) is True
    assert tuple(getattr(after, "writable_paths", ())) == before_writable
    assert tuple(getattr(after, "creatable_paths", ())) == before_creatable
    assert OTHER not in getattr(after, "writable_paths", ())


def test_authorized_target_stays_executable_and_unrelated_target_is_rejected() -> None:
    state = tool_loop.HostRunState()
    messages = [{"role": "developer", "content": json.dumps(_authority_payload())}]
    assert tool_loop.is_mutation_ready(messages, state) is True

    assert (
        tool_loop._mutation_target_error(
            "apply_source_edit",
            {"operation": "create_file", "path": TARGET},
            state.mutation_context,
        )
        is None
    )
    error = tool_loop._mutation_target_error(
        "apply_source_edit",
        {"operation": "replace_exact", "path": OTHER},
        state.mutation_context,
    )
    assert error is not None
    assert "MUTATION_TARGET_DRIFT" in error
