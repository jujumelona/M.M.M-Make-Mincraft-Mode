from __future__ import annotations

from minecraft_mod_ai.progress_aware_tool_loop import HostRunState


def _applied_payload(*, path: str, before: str | None, after: str) -> dict[str, object]:
    return {
        "ok": True,
        "result": {
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
            }
        },
    }


def test_applied_create_survives_later_edit_in_same_host_run_state():
    path = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    state = HostRunState()

    assert state.record_mutation(
        "apply_source_edit",
        {
            "operation": "create_file",
            "path": path,
            "content": "package dev.mmm.debugfixture; public class DebugToken {}",
        },
        _applied_payload(path=path, before=None, after="sha256:created"),
    )
    assert state.record_mutation(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": path,
            "old": "DebugToken {}",
            "new": "DebugToken { public DebugToken() {} }",
        },
        _applied_payload(
            path=path,
            before="sha256:created",
            after="sha256:edited",
        ),
    )

    assert state.created_paths == {path}
    assert state.workspace_changed is True
    assert state.validation_status == "PENDING"
