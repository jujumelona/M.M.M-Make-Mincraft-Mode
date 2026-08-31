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


def test_fresh_without_reuse_evidence_keeps_reserved_creation_target() -> None:
    payload = _fresh_request()

    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.target_symbol == TARGET_SYMBOL
    assert context.is_new_file is True
    assert context.localization_stage == tool_loop.LocalizationStage.READY


def test_exact_reserved_path_in_initial_source_is_not_treated_as_new_file() -> None:
    payload = _fresh_request()
    payload["initial_exact_source_context"] = {
        "records": [
            {
                "path": TARGET_PATH,
                "content": "package generated.example.mmmplan; public final class TaskFeature { void run() {} }",
            }
        ]
    }

    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.target_symbol == TARGET_SYMBOL
    assert context.is_new_file is False
    assert context.source_body is not None
    assert context.localization_stage == tool_loop.LocalizationStage.READY


def test_incidental_initial_source_cannot_replace_reserved_fresh_target() -> None:
    payload = _fresh_request()
    payload["initial_exact_source_context"] = {
        "records": [
            {
                "path": "src/main/java/mod/Existing.java",
                "content": "package mod; public final class Existing {}",
            }
        ]
    }

    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path == TARGET_PATH
    assert context.is_new_file is True


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
    payload = _fresh_request(binding_source_refs=("src/main/java/mod/TradeEngine.java",))

    context = tool_loop._extract_mutation_context_from_payload(payload)

    assert context is not None
    assert context.target_path is None
    assert context.localization_stage == tool_loop.LocalizationStage.NEED_FILE


def test_different_target_replaces_context_without_cross_file_state_leak() -> None:
    fresh = tool_loop.TargetMutationContext(
        target_path=TARGET_PATH,
        target_symbol=TARGET_SYMBOL,
        source_body="public final class TaskFeature {}",
        is_new_file=True,
        evidence_source="evidence_fresh_owned_anchor",
    )
    localized = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/TradeEngine.java",
        target_symbol="TradeEngine#tick",
        source_body="public void tick() { executeTrades(); }",
        start_line=40,
        end_line=42,
        evidence_source="search_code_rag",
    )

    merged = fresh.merge(localized)

    assert merged == localized
    assert merged.is_new_file is False
    assert merged.target_symbol == "TradeEngine#tick"
    assert "TaskFeature" not in (merged.source_body or "")


def test_different_path_only_evidence_resets_old_body_and_symbol() -> None:
    ready = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/A.java",
        target_symbol="A#run",
        source_body="public void run() {}",
        evidence_source="search_code_rag",
    )
    next_file = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/B.java",
        evidence_source="search_code_rag_path_only",
    )

    merged = ready.merge(next_file)

    assert merged.target_path == "src/main/java/mod/B.java"
    assert merged.target_symbol is None
    assert merged.source_body is None
    assert merged.localization_stage == tool_loop.LocalizationStage.NEED_SYMBOL


def test_repository_evidence_downgrades_same_path_from_new_to_existing() -> None:
    prospective = tool_loop.TargetMutationContext(
        target_path=TARGET_PATH,
        target_symbol=TARGET_SYMBOL,
        is_new_file=True,
        evidence_source="evidence_fresh_owned_anchor",
    )
    observed = tool_loop.TargetMutationContext(
        target_path=TARGET_PATH,
        target_symbol=TARGET_SYMBOL,
        source_body="public final class TaskFeature { void run() {} }",
        evidence_source="search_code_rag",
    )

    merged = prospective.merge(observed)

    assert merged.target_path == TARGET_PATH
    assert merged.is_new_file is False
    assert merged.source_body == observed.source_body
    assert merged.localization_stage == tool_loop.LocalizationStage.READY


def test_same_target_incremental_localization_still_accumulates() -> None:
    file_context = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/Item.java",
        evidence_source="search_code_rag_path_only",
    )
    symbol_context = tool_loop.TargetMutationContext(
        target_symbol="Item#use",
        start_line=20,
        end_line=35,
        evidence_source="java_workspace_symbols",
    )
    body_context = tool_loop.TargetMutationContext(
        source_body="public ActionResult use() { return PASS; }",
        evidence_source="search_code_rag",
    )

    merged = file_context.merge(symbol_context).merge(body_context)

    assert merged.target_path == "src/main/java/mod/Item.java"
    assert merged.target_symbol == "Item#use"
    assert merged.start_line == 20
    assert merged.end_line == 35
    assert merged.source_body == body_context.source_body
    assert merged.is_new_file is False
    assert merged.localization_stage == tool_loop.LocalizationStage.READY



def test_target_hardening_has_no_late_monkey_patch_owner() -> None:
    assert not getattr(
        tool_loop.TargetMutationContext.merge,
        "_mmm_target_identity_merge_guard",
        False,
    )
    assert not getattr(
        tool_loop._extract_mutation_context_from_payload,
        "_mmm_fresh_owned_target_grounding",
        False,
    )


def _worker07_tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_mutation_target_guard_accepts_pinned_existing_path() -> None:
    context = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/TradeEngine.java",
        target_symbol="TradeEngine#tick",
        source_body="public void tick() { executeTrades(); }",
        evidence_source="search_code_rag",
    )
    assert tool_loop._mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": "src/main/java/mod/TradeEngine.java",
            "old": "executeTrades();",
            "new": "executeValidatedTrades();",
        },
        context,
    ) is None


def test_mutation_target_guard_normalizes_supported_path_alias() -> None:
    context = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/TradeEngine.java",
        target_symbol="TradeEngine#tick",
        source_body="public void tick() { executeTrades(); }",
        evidence_source="search_code_rag",
    )
    assert tool_loop._mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "target_path": "./src/main/java/mod/TradeEngine.java",
            "old": "executeTrades();",
            "new": "executeValidatedTrades();",
        },
        context,
    ) is None


def test_mutation_target_guard_rejects_cross_file_drift() -> None:
    context = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/TradeEngine.java",
        target_symbol="TradeEngine#tick",
        source_body="public void tick() { executeTrades(); }",
        evidence_source="search_code_rag",
    )
    invented = "src/main/java/generated/generated_mod/mmmplan/Invented.java"
    error = tool_loop._mutation_target_error(
        "apply_source_edit",
        {
            "operation": "create_java_type",
            "path": invented,
            "package_name": "generated.generated_mod.mmmplan",
            "declaration": "public final class Invented",
        },
        context,
    )
    assert error is not None
    assert "MUTATION_TARGET_DRIFT" in error
    assert context.target_path in error
    assert invented in error


def test_mutation_target_guard_rejects_missing_body_existing_context() -> None:
    context = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/TradeEngine.java",
        target_symbol="TradeEngine#tick",
        evidence_source="java_workspace_symbols",
    )
    assert context.localization_stage == tool_loop.LocalizationStage.NEED_BODY
    error = tool_loop._mutation_target_error(
        "apply_source_edit",
        {"operation": "replace_exact", "path": context.target_path, "old": "x", "new": "y"},
        context,
    )
    assert error is not None
    assert "MUTATION_TARGET_UNBOUND" in error


def test_mutation_target_guard_rejects_recreate_of_existing_target() -> None:
    context = tool_loop.TargetMutationContext(
        target_path="src/main/java/mod/TradeEngine.java",
        target_symbol="TradeEngine",
        source_body="public final class TradeEngine {}",
        evidence_source="search_code_rag",
    )
    error = tool_loop._mutation_target_error(
        "apply_source_edit",
        {
            "operation": "create_java_type",
            "path": context.target_path,
            "package_name": "mod",
            "declaration": "public final class TradeEngine",
        },
        context,
    )
    assert error is not None
    assert "MUTATION_TARGET_CREATION_CONFLICT" in error


def test_mutation_target_guard_allows_reserved_new_target_creation() -> None:
    context = tool_loop.TargetMutationContext(
        target_path=TARGET_PATH,
        target_symbol=TARGET_SYMBOL,
        is_new_file=True,
        evidence_source="evidence_fresh_owned_anchor",
    )
    assert tool_loop._mutation_target_error(
        "apply_source_edit",
        {
            "operation": "create_java_type",
            "path": TARGET_PATH,
            "package_name": "generated.example.mmmplan",
            "declaration": "public final class TaskFeature",
        },
        context,
    ) is None


def test_act_prefers_canonical_source_edit_when_multiple_mutators_exposed() -> None:
    schemas = tuple(
        _worker07_tool_schema(name)
        for name in ("apply_source_edit", "apply_source_patch", "apply_java_operations", "repair_project")
    )
    selected = tool_loop._filter_tools_for_phase(
        schemas,
        tool_loop.LoopPhase.ACT,
        "coder",
    )
    assert [schema["function"]["name"] for schema in selected] == ["apply_source_edit"]


def test_act_preserves_legacy_fallback_when_source_edit_is_absent() -> None:
    schemas = tuple(
        _worker07_tool_schema(name)
        for name in ("apply_source_patch", "repair_project")
    )
    selected = tool_loop._filter_tools_for_phase(
        schemas,
        tool_loop.LoopPhase.ACT,
        "coder",
    )
    assert [schema["function"]["name"] for schema in selected] == [
        "apply_source_patch",
        "repair_project",
    ]

