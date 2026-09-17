from __future__ import annotations

from minecraft_mod_ai.progress_aware_tool_loop import HostRunState, TargetMutationContext


TARGET = "src/main/java/dev/mmm/debugfixture/DebugToken.java"


def _fresh_context() -> TargetMutationContext:
    return TargetMutationContext(
        target_path=TARGET,
        target_symbol="DebugToken",
        is_new_file=True,
        evidence_source="host_task_authority",
        writable_paths=(TARGET,),
        creatable_paths=(TARGET,),
        target_pinned=True,
    )


def _applied_receipt() -> dict[str, object]:
    return {
        "ok": True,
        "result": {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "operations": [
                {
                    "operation": "create",
                    "path": TARGET,
                    "before_sha256": None,
                    "after_sha256": "sha256:created",
                }
            ],
        },
    }


def test_fresh_target_remains_creatable_before_first_mutation() -> None:
    context = _fresh_context()

    assert context.is_new_file is True
    assert context.creatable_paths == (TARGET,)
    assert context.is_mutation_ready is True


def test_successful_create_retires_creation_authority() -> None:
    state = HostRunState(mutation_context=_fresh_context())
    source = "package dev.mmm.debugfixture;\n\npublic final class DebugToken {}\n"

    assert state.record_mutation(
        "apply_source_edit",
        {"operation": "create_file", "path": TARGET, "content": source},
        _applied_receipt(),
    ) is True

    assert state.mutation_context is not None
    assert state.mutation_context.is_new_file is False
    assert TARGET not in state.mutation_context.creatable_paths
    assert state.mutation_context.evidence_source == "mutation_receipt"
    assert state.mutation_context.source_body == source


def test_static_fresh_authority_cannot_resurrect_materialized_target() -> None:
    state = HostRunState(mutation_context=_fresh_context())
    source = "package dev.mmm.debugfixture;\n\npublic final class DebugToken {}\n"
    state.record_mutation(
        "apply_source_edit",
        {"operation": "create_file", "path": TARGET, "content": source},
        _applied_receipt(),
    )
    assert state.mutation_context is not None

    merged = state.mutation_context.merge(_fresh_context())

    assert merged.is_new_file is False
    assert TARGET not in merged.creatable_paths
    assert merged.evidence_source == "mutation_receipt"
    assert merged.source_body == source


def test_exact_existing_source_overrides_static_fresh_authority() -> None:
    fresh = _fresh_context()
    source = "package dev.mmm.debugfixture;\n\npublic final class DebugToken {}\n"
    exact = TargetMutationContext(
        target_path=TARGET,
        target_symbol="DebugToken",
        source_body=source,
        is_new_file=False,
        evidence_source="host_exact_source",
        writable_paths=(TARGET,),
        target_pinned=True,
    )

    merged = fresh.merge(exact).merge(fresh)

    assert merged.is_new_file is False
    assert TARGET not in merged.creatable_paths
    assert merged.evidence_source == "host_exact_source"
    assert merged.source_body == source
