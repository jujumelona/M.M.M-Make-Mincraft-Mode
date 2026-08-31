from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "minecraft_mod_ai" / "progress_aware_tool_loop.py"
TEST = ROOT / "tests" / "test_worker07_target_context_hardening.py"
DOC = ROOT / "docs" / "parallel_progress" / "worker-07.md"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_loop() -> None:
    text = LOOP.read_text(encoding="utf-8")

    helper_marker = "def _mutation_target_error("
    if helper_marker not in text:
        anchor = '''def _normalized_target_path(value: Any) -> str:\n    return str(value or "").strip().replace("\\\\", "/")\n'''
        insertion = anchor + '''\n\n_SOURCE_EDIT_PATH_KEYS = ("path", "file", "target_path", "target_file")\n_SOURCE_CREATE_OPERATIONS = frozenset(\n    {\n        "create",\n        "create_file",\n        "create_java_type",\n        "create_class",\n        "create_type",\n        "write",\n        "write_file",\n    }\n)\n\n\ndef _canonical_mutation_path(value: Any) -> str:\n    clean = _normalized_target_path(value)\n    while clean.startswith("./"):\n        clean = clean[2:]\n    return clean\n\n\ndef _mutation_target_error(\n    tool_name: str,\n    arguments: Mapping[str, Any],\n    context: TargetMutationContext | None,\n) -> str | None:\n    """Reject source edits that escape the repository-localized target."""\n\n    if tool_name != "apply_source_edit":\n        return None\n    if context is None or not context.is_mutation_ready:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "\n            "repository-localized target context."\n        )\n\n    pinned = _canonical_mutation_path(context.target_path)\n    supplied = ""\n    for key in _SOURCE_EDIT_PATH_KEYS:\n        value = arguments.get(key)\n        if isinstance(value, str) and value.strip():\n            supplied = _canonical_mutation_path(value)\n            break\n\n    if not pinned or not supplied:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires the pinned target "\n            "path in its model payload."\n        )\n    if supplied != pinned:\n        return (\n            f"MUTATION_TARGET_DRIFT: pinned target {pinned!r} but "\n            f"apply_source_edit requested {supplied!r}."\n        )\n\n    operation = str(arguments.get("operation", "")).strip().casefold()\n    if not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:\n        return (\n            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "\n            f"{pinned!r} cannot be recreated by {operation!r}."\n        )\n    return None\n'''
        text = replace_once(text, anchor, insertion, label="target guard helper")

    old_act = '''    elif phase == LoopPhase.ACT:\n        selected_names = [name for name in by_name if name in _MUTATION_ACT_TOOLS]\n'''
    new_act = '''    elif phase == LoopPhase.ACT:\n        # apply_source_edit is the canonical model-facing source mutation surface.\n        # Prefer it whenever available so alternate mutators cannot bypass the\n        # repository-localized target binding. Legacy fallbacks remain available\n        # only when the canonical tool is genuinely absent.\n        if "apply_source_edit" in by_name:\n            selected_names = ["apply_source_edit"]\n        else:\n            selected_names = [name for name in by_name if name in _MUTATION_ACT_TOOLS]\n'''
    if new_act not in text:
        text = replace_once(text, old_act, new_act, label="canonical ACT source editor")

    guard = '''            target_error = _mutation_target_error(\n                call.name,\n                call.arguments,\n                state.mutation_context,\n            )\n            if target_error is not None:\n                state.record_failure(call.name, target_error)\n                print(\n                    f"  [!] MUTATION TARGET REJECTED: {call.name} -> {target_error}",\n                    flush=True,\n                )\n                return call, {\n                    "ok": False,\n                    "tool": call.name,\n                    **route_metadata,\n                    "error": target_error,\n                }\n\n'''
    execute_anchor = '''            try:\n                if call.name.startswith("external_mcp_"):\n'''
    if guard not in text:
        text = replace_once(
            text,
            execute_anchor,
            guard + execute_anchor,
            label="pre-runtime source target guard",
        )

    LOOP.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TEST.read_text(encoding="utf-8")
    if "test_mutation_target_guard_rejects_cross_file_drift" in text:
        return

    addition = r'''


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
'''
    TEST.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def patch_doc() -> None:
    text = DOC.read_text(encoding="utf-8")
    marker = "## Latest-main source-edit target re-certification"
    if marker in text:
        return
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    section = f'''\n\n{marker}\n\nRe-certified from latest synchronized `main@{base}` after parallel-worker integration. A final Worker-07 execution-boundary gap was found and closed: repository localization already pinned file/symbol/body identity, but the concrete model-supplied `apply_source_edit` path was not revalidated immediately before runtime execution.\n\nThe canonical loop now enforces all of the following before source mutation:\n\n- `apply_source_edit` requires a `READY` repository-localized `TargetMutationContext`; an existing file with a missing body cannot mutate.\n- the concrete source-edit payload path must equal the pinned localized path after conservative `./` and separator normalization; generated/cross-file drift is rejected before the runtime is called.\n- create operations cannot recreate an already-localized existing file.\n- a legitimate host-reserved new target can still be created.\n- when `apply_source_edit` is available, ACT exposes it as the single canonical model-facing source mutation surface; legacy mutation tools remain fallback-only when it is absent.\n- VERIFY failure returns to ACT without replacing `state.mutation_context`, so every repair edit is checked against the same pinned target by the same pre-runtime guard.\n\nThe one-shot re-certification gate requires syntax/lint, focused Worker-07 plus unified-loop regression suites, repository static audit, high-confidence dead-code audit, and no growth in the runtime monkey-patch mutation count before the final direct-`main` commit is pushed.\n'''
    DOC.write_text(text.rstrip() + section + "\n", encoding="utf-8")


def main() -> None:
    patch_loop()
    patch_tests()
    patch_doc()


if __name__ == "__main__":
    main()
