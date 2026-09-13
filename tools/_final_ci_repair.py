from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise SystemExit(f"missing repair anchor in {path}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


# Successful work is already durably checkpointed by deadline_executor's on_result.
# The detached-result loop must only update the in-memory assembly mirrors.
path = "minecraft_mod_ai/planning_state_adaptive_implementation.py"
text = read(path)
loop_start = text.index("            for work_item, result_receipt in completed_work:")
loop_end = text.index("        except ParallelExecutionTimeout as exc:", loop_start)
block = text[loop_start:loop_end]
old = '''                    if kind_tag == "artifact":
                        result_receipt = _planning_only_artifact_receipt(result_receipt)
                        job["artifact_fragments"].setdefault(artifact_kind, {})[
                            step_id
                        ] = result_receipt
                        working_state = _store_artifact_progress(
                            working_state,
                            requirement_ref=job["requirement_ref"],
                            artifact_kind=artifact_kind,
                            step_id=step_id,
                            receipt=result_receipt,
                        )
                    else:
                        job["fragments"][idx] = result_receipt
                        working_state = store_criterion_progress(
                            working_state,
                            requirement_ref=job["requirement_ref"],
                            selected_sections=job["selected_sections"],
                            criterion_index=idx,
                            criterion=job["criteria"][idx],
                            fragment=result_receipt,
                        )

'''
new = '''                    # Durable progress was already checkpointed by on_result.
                    # Only mirror detached results into the assembly job here.
                    if kind_tag == "artifact":
                        result_receipt = _planning_only_artifact_receipt(result_receipt)
                        job["artifact_fragments"].setdefault(artifact_kind, {})[
                            step_id
                        ] = result_receipt
                    else:
                        job["fragments"][idx] = result_receipt

'''
if old not in block:
    raise SystemExit("adaptive duplicate-progress anchor missing")
block = block.replace(old, new, 1)
old = "                    working_state = _checkpoint_state(working_state, checkpoint)\n                    emit_root_cause(\n"
if old not in block:
    raise SystemExit("adaptive duplicate-checkpoint anchor missing")
block = block.replace(old, "                    emit_root_cause(\n", 1)
write(path, text[:loop_start] + block + text[loop_end:])


# The separate model-authored coverage decision is retired. Requirement pagination is
# host-owned and terminates when the semantic requirement frontier stops advancing.
path = "minecraft_mod_ai/planning_contract_ssot.py"
text = read(path)
start = text.index("REQUIREMENT_COVERAGE_SCHEMA: dict[str, Any] = {")
end = text.index("RESEARCH_NOTE_SCHEMA: dict[str, Any] = {", start)
text = text[:start] + text[end:]
entry = '        ("REQUIREMENT_COVERAGE_SCHEMA", REQUIREMENT_COVERAGE_SCHEMA),\n'
if entry not in text:
    raise SystemExit("coverage schema registry entry missing")
write(path, text.replace(entry, "", 1))


# Update pagination regressions from the retired coverage-authority protocol to the
# current page/frontier protocol. No arbitrary page-count cap is introduced.
path = "tests/test_planning_requirement_pagination.py"
text = read(path)
replacements = [
    (
        "router = Router([page(BEHAVIORS[:4]), coverage(), page(BEHAVIORS[4:])])",
        "router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[4:])])",
    ),
    (
        '    assert router.messages[2]["uncovered_authored_behavior"] == coverage()\n',
        '    assert "uncovered_authored_behavior" not in router.messages[1]\n',
    ),
    (
        "router = Router([page(BEHAVIORS[:4]), coverage(True)])",
        "router = Router([page(BEHAVIORS[:4]), page([])])",
    ),
    (
        '''def test_repeated_full_page_cannot_be_mistaken_for_complete_coverage():
    router = Router([page(BEHAVIORS[:4]), coverage(), page(BEHAVIORS[:4])])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_PAGINATION_NO_PROGRESS"):
        compile_researched_requirements(router, PROMPT, state())
''',
        '''def test_repeated_full_page_is_semantic_convergence_evidence():
    router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[:4])])
    result = compile_researched_requirements(router, PROMPT, state())
    assert len(result["decisions"]) == 4
    assert len(router.messages) == 2
''',
    ),
    (
        "def test_missing_coverage_decision_cannot_certify_full_page():",
        "def test_invalid_continuation_payload_cannot_certify_full_page():",
    ),
    (
        'match="REQUIREMENT_COVERAGE_INVALID"',
        'match="REQUIREMENT_PAGINATION_FAILED: invalid continuation page"',
    ),
    (
        '''def test_invented_remaining_quote_cannot_trigger_more_generation():
    router = Router([page(BEHAVIORS[:4]), {**coverage(), "remaining_source_quote": "unrequested teleporter"}])
    with pytest.raises(ModelConfigurationError, match="exact task quote"):
        compile_researched_requirements(router, PROMPT, state())
    assert len(router.messages) == 2
''',
        '''def test_continuation_context_is_host_owned_not_model_coverage_authority():
    router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[4:])])
    compile_researched_requirements(router, PROMPT, state())
    continuation = router.messages[1]
    assert continuation["original_prompt"] == PROMPT
    assert continuation["already_compiled_requirements"] == page(BEHAVIORS[:4])["requirements"]
    assert "complete" not in continuation
    assert "remaining_source_quote" not in continuation
    assert "remaining_behavior" not in continuation
    assert "uncovered_authored_behavior" not in continuation
''',
    ),
    (
        '''def test_coverage_schema_is_atomic():
    from minecraft_mod_ai.model_output_atomicity_contract import (
        assert_atomic_model_schema,
    )
    from minecraft_mod_ai.planning_contract_ssot import REQUIREMENT_COVERAGE_SCHEMA
    assert_atomic_model_schema(REQUIREMENT_COVERAGE_SCHEMA, surface="requirement coverage")
''',
        '''def test_requirement_page_schema_is_atomic():
    from minecraft_mod_ai.model_output_atomicity_contract import (
        assert_atomic_model_schema,
    )
    from minecraft_mod_ai.planning_contract_ssot import SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA
    assert_atomic_model_schema(
        SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
        surface="requirement page",
    )
''',
    ),
    (
        "def test_coverage_transport_failure_cannot_publish_partial_requirements():",
        "def test_continuation_transport_failure_cannot_publish_partial_requirements():",
    ),
    (
        'match="REQUIREMENT_COVERAGE_FAILED"',
        'match="REQUIREMENT_PAGINATION_FAILED"',
    ),
    (
        "router = Router([page(behaviors[:4]), coverage(), page(behaviors[4:]), coverage(True)])",
        "router = Router([page(behaviors[:4]), page(behaviors[4:]), page([])])",
    ),
    (
        "    assert len(router.messages) == 4\n",
        "    assert len(router.messages) == 3\n",
    ),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f"pagination stale anchor missing: {old[:100]!r}")
    text = text.replace(old, new, 1)
write(path, text)


# Same-stage generators mutate shared stage state and therefore remain serialized.
replace_once(
    "tests/test_scheduler_parallel_safety_contract.py",
    '''def test_anchor_fenced_runtime_disables_coarse_stage_admission() -> None:
    assert scheduler_contract._SERIAL_CPU_STAGES == ()
''',
    '''def test_shared_stage_mutation_domains_are_serialized_by_admission() -> None:
    assert scheduler_contract._SERIAL_CPU_STAGES == (
        "generate:content",
        "generate:system",
        "generate:entity",
    )
''',
)
