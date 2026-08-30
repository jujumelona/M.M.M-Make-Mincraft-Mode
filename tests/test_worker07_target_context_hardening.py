from __future__ import annotations

from minecraft_mod_ai import implementation_kind_boundary_contract as boundary
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

    assert boundary._fresh_target_has_reuse_evidence(payload) is True
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
