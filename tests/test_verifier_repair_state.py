from __future__ import annotations

import json

from minecraft_mod_ai.verifier_fail_closed_completion_installation import (
    _applied_created_paths,
)


def _assistant_call(call_id: str, arguments: dict[str, str]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def _applied_receipt(call_id: str, *, path: str, before: str | None, after: str) -> dict[str, str]:
    return {
        "role": "tool",
        "name": "apply_source_edit",
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "ok": True,
                "result": {
                    "_mmm_observation": {
                        "sanitized": True,
                        "truncated": False,
                        "trust": "untrusted_data_only",
                    },
                    "structured_content": {
                        "schema_version": "mmm/source-patch-receipt-v1",
                        "status": "APPLIED",
                        "changed_paths": [path],
                        "operations": [
                            {
                                "operation": "create" if before is None else "replace",
                                "path": path,
                                "before_sha256": before,
                                "after_sha256": after,
                            }
                        ],
                    },
                },
            }
        ),
    }


def test_applied_create_survives_reused_tool_call_id_after_later_edit():
    path = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    call_id = "host_mutation_stable_id"
    messages = [
        _assistant_call(
            call_id,
            {
                "operation": "create_file",
                "path": path,
                "content": "package dev.mmm.debugfixture; public class DebugToken {}",
            },
        ),
        _applied_receipt(
            call_id,
            path=path,
            before=None,
            after="sha256:created",
        ),
        _assistant_call(
            call_id,
            {
                "operation": "replace_exact",
                "path": path,
                "old": "DebugToken {}",
                "new": "DebugToken { public DebugToken() {} }",
            },
        ),
        _applied_receipt(
            call_id,
            path=path,
            before="sha256:created",
            after="sha256:edited",
        ),
    ]

    assert _applied_created_paths(messages) == frozenset({path})
