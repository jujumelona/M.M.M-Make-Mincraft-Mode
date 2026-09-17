from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "minecraft_mod_ai" / "progress_aware_tool_loop.py"
TRANSITION_TEST = ROOT / "tests" / "test_fresh_java_grounding_transition.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_loop() -> None:
    text = LOOP.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '''        if mutation_context and mutation_context.is_new_file and mutation_context.is_mutation_ready:\n            preferred = (\n                "search_code_rag", "external_mcp_call", "java_workspace_symbols"\n            )\n''',
        '''        if mutation_context and mutation_context.is_new_file and mutation_context.is_mutation_ready:\n            preferred = (\n                "search_code_rag", "java_workspace_symbols", "external_mcp_call"\n            )\n''',
        "fresh Java evidence preference",
    )

    text = replace_once(
        text,
        '''        names = [name for name in preferred if name in by_name and name not in attempted]\n        if semantic_retrieval_choice and names:\n            names = [names[0]]\n        elif names:\n            names = [names[0]]\n''',
        '''        names = [name for name in preferred if name in by_name and name not in attempted]\n        if names:\n            fresh_reserved = bool(\n                mutation_context\n                and mutation_context.is_new_file\n                and mutation_context.is_mutation_ready\n            )\n            if (\n                semantic_retrieval_choice\n                and fresh_reserved\n                and "search_code_rag" in names\n                and "search_code_rag" not in attempted\n            ):\n                # Fresh Java starts with current-project code evidence, but once that\n                # route is exhausted the model must be allowed to choose among the\n                # remaining reviewed evidence routes instead of being force-fed one.\n                names = ["search_code_rag"]\n            elif not semantic_retrieval_choice:\n                names = [names[0]]\n''',
        "semantic evidence portfolio",
    )

    text = replace_once(
        text,
        '''    state = HostRunState()\n    unavailable_verifiers: set[str] = set()\n''',
        '''    state = HostRunState()\n    unavailable_verifiers: set[str] = set()\n    required_evidence_choice = False\n''',
        "required evidence choice state",
    )

    text = replace_once(
        text,
        '''        phase_names = frozenset(_tool_name(schema) for schema in phase_tools if _tool_name(schema))\n        tool_choice = request.tool_choice\n        parallel = request.parallel_tool_calls\n\n        if state.phase == LoopPhase.ACT:\n''',
        '''        phase_names = frozenset(_tool_name(schema) for schema in phase_tools if _tool_name(schema))\n        tool_choice = request.tool_choice\n        parallel = request.parallel_tool_calls\n\n        if (\n            required_evidence_choice\n            and require_rag\n            and not baseline_ready\n            and state.phase in {LoopPhase.OBSERVE, LoopPhase.RECOVER}\n            and len(phase_names) > 1\n        ):\n            # The host requires evidence, not a particular semantic route. Force one\n            # visible tool call while leaving route selection to the model.\n            tool_choice = "required"\n            parallel = False\n        elif state.phase == LoopPhase.ACT:\n''',
        "required semantic evidence tool choice",
    )

    text = replace_once(
        text,
        '''            if repeated:\n                raise _fixed_point_error(state)\n            continue\n\n        if not turn.tool_calls:\n''',
        '''            if require_rag and not baseline_ready:\n                required_evidence_choice = True\n            if repeated:\n                raise _fixed_point_error(state)\n            continue\n\n        if not turn.tool_calls:\n''',
        "rejection escalation",
    )

    text = replace_once(
        text,
        '''        if not turn.tool_calls:\n            content = turn.content.strip()\n            if not content:\n                raise ModelConfigurationError("Tool-capable model returned an empty final response.")\n            if implementation and state.phase in {LoopPhase.ACT, LoopPhase.VERIFY, LoopPhase.RECOVER}:\n''',
        '''        if not turn.tool_calls:\n            content = turn.content.strip()\n            if not content:\n                raise ModelConfigurationError("Tool-capable model returned an empty final response.")\n            if require_rag and not baseline_ready:\n                repeated = state.record_no_progress_result({\n                    "phase": state.phase.value,\n                    "validation": state.validation_status,\n                    "verifier": state.latest_verifier_fingerprint,\n                    "missing_required_evidence": True,\n                    "prose": content,\n                })\n                messages.extend([\n                    {"role": "assistant", "content": content},\n                    {\n                        "role": "system",\n                        "content": (\n                            "Reviewed production evidence is still required. Select exactly one "\n                            "currently exposed evidence function that best matches the information "\n                            "need and call it. Do not answer in prose and do not invent tool names."\n                        ),\n                    },\n                ])\n                required_evidence_choice = True\n                if repeated:\n                    raise _fixed_point_error(state)\n                continue\n            if implementation and state.phase in {LoopPhase.ACT, LoopPhase.VERIFY, LoopPhase.RECOVER}:\n''',
        "prose before required evidence",
    )

    text = replace_once(
        text,
        '''        if progress:\n            state.clear_no_progress_result()\n        else:\n            state.record_no_progress_result({\n                "phase_before": phase_before.value,\n                "phase_after": phase_after,\n                "localization_before": loc_before,\n                "localization_after": loc_after,\n                "target": ctx_after,\n                "validation": state.validation_status,\n                "verifier": state.latest_verifier_fingerprint,\n                "calls": call_info,\n                "results": result_info,\n            })\n''',
        '''        if progress:\n            state.clear_no_progress_result()\n            if _target_evidence_ready(\n                state, require_rag=require_rag, fresh_java_target=fresh_java_target\n            ):\n                required_evidence_choice = False\n        else:\n            state.record_no_progress_result({\n                "phase_before": phase_before.value,\n                "phase_after": phase_after,\n                "localization_before": loc_before,\n                "localization_after": loc_after,\n                "target": ctx_after,\n                "validation": state.validation_status,\n                "verifier": state.latest_verifier_fingerprint,\n                "calls": call_info,\n                "results": result_info,\n            })\n            if require_rag and not _target_evidence_ready(\n                state, require_rag=require_rag, fresh_java_target=fresh_java_target\n            ):\n                required_evidence_choice = True\n''',
        "no-progress evidence escalation",
    )

    LOOP.write_text(text, encoding="utf-8")


def patch_transition_test() -> None:
    text = TRANSITION_TEST.read_text(encoding="utf-8")
    anchor = '''def test_generic_fresh_evidence_does_not_unlock_fresh_java():\n'''
    addition = '''def test_fresh_java_after_weak_code_rag_exposes_remaining_semantic_routes():\n    selected = loop._filter_tools_for_phase(\n        (\n            _tool("search_project_rag"),\n            _tool("java_workspace_symbols"),\n            _tool("external_mcp_call"),\n            _tool("search_code_rag"),\n        ),\n        loop.LoopPhase.OBSERVE,\n        "coder",\n        mutation_context=_fresh_context(),\n        attempted_sources={"search_code_rag"},\n        localization_active=True,\n        semantic_retrieval_choice=True,\n    )\n    assert [schema["function"]["name"] for schema in selected] == [\n        "java_workspace_symbols",\n        "external_mcp_call",\n    ]\n\n\n'''
    if "test_fresh_java_after_weak_code_rag_exposes_remaining_semantic_routes" not in text:
        text = replace_once(text, anchor, addition + anchor, "fresh Java fallback regression")
    TRANSITION_TEST.write_text(text, encoding="utf-8")


def main() -> None:
    patch_loop()
    patch_transition_test()


if __name__ == "__main__":
    main()
