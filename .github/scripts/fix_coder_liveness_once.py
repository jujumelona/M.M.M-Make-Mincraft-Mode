from __future__ import annotations

from pathlib import Path
import re

ROOT = Path('.')


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding='utf-8')
    if old not in text:
        raise SystemExit(f'{label}: anchor missing')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')


# 1) VERIFY is verification only. Do not expose research/RAG tools there.
progress = ROOT / 'minecraft_mod_ai/progress_aware_tool_loop.py'
replace_once(
    progress,
    '''    elif phase in (LoopPhase.VERIFY, LoopPhase.RECOVER):
        selected_names = [
            name for name in by_name
            if name in _VERIFY_TOOLS or name in _READ_OBSERVE_TOOLS
        ]
''',
    '''    elif phase == LoopPhase.VERIFY:
        selected_names = [name for name in by_name if name in _VERIFY_TOOLS]
    elif phase == LoopPhase.RECOVER:
        selected_names = [name for name in by_name if name in _READ_OBSERVE_TOOLS]
''',
    'split VERIFY from RECOVER tools',
)

replace_once(
    progress,
    '''    forced_rag_tool: str | None = None
    forced_rag_attempts = 0
    required_rag_choice = False
''',
    '''    forced_rag_tool: str | None = None
    forced_rag_attempts = 0
    required_rag_choice = False
    unavailable_verifiers: set[str] = set()
''',
    'verifier health state',
)

replace_once(
    progress,
    '''        if required_rag_choice:
            phase_tools = tuple(
                schema for schema in phase_tools
                if _tool_name(schema) in _RAG_EVIDENCE_TOOLS
            )
            if not phase_tools:
                raise ModelConfigurationError(
                    "Required production evidence is unavailable: no reviewed RAG tool remains "
                    "eligible for semantic selection."
                )
        phase_tool_names = frozenset(_tool_name(s) for s in phase_tools if _tool_name(s))
''',
    '''        if required_rag_choice:
            phase_tools = tuple(
                schema for schema in phase_tools
                if _tool_name(schema) in _RAG_EVIDENCE_TOOLS
            )
            if not phase_tools:
                raise ModelConfigurationError(
                    "Required production evidence is unavailable: no reviewed RAG tool remains "
                    "eligible for semantic selection."
                )

        forced_verify_tool: str | None = None
        if state.phase == LoopPhase.VERIFY:
            phase_tools = tuple(
                schema for schema in phase_tools
                if _tool_name(schema) not in unavailable_verifiers
            )
            verifier_preference = (
                "java_diagnostics",
                "jdt_diagnostics",
                "run_gradle_build",
                "gradle_build",
                "run_gametest",
            )
            available_verifier_names = {
                _tool_name(schema) for schema in phase_tools if _tool_name(schema)
            }
            forced_verify_tool = next(
                (name for name in verifier_preference if name in available_verifier_names),
                None,
            )
            if forced_verify_tool is None:
                state.termination_reason = "VERIFIER_UNAVAILABLE"
                raise ModelConfigurationError(
                    "VERIFIER_UNAVAILABLE: no healthy host verifier remains; refusing to send "
                    "the coder back into retrieval or mutation without trustworthy diagnostics."
                )
            phase_tools = tuple(
                schema for schema in phase_tools
                if _tool_name(schema) == forced_verify_tool
            )

        phase_tool_names = frozenset(_tool_name(s) for s in phase_tools if _tool_name(s))
''',
    'deterministic verifier selection',
)

replace_once(
    progress,
    '''        elif state.phase == LoopPhase.ACT:
            mutation_names = [n for n in phase_tool_names if n in _MUTATION_ACT_TOOLS]
            if len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel_tool_calls = False
''',
    '''        elif state.phase == LoopPhase.ACT:
            mutation_names = [n for n in phase_tool_names if n in _MUTATION_ACT_TOOLS]
            if len(mutation_names) == 1:
                tool_choice = {"type": "function", "function": {"name": mutation_names[0]}}
                parallel_tool_calls = False
        elif forced_verify_tool is not None:
            tool_choice = {"type": "function", "function": {"name": forced_verify_tool}}
            parallel_tool_calls = False
''',
    'force verifier tool choice',
)

replace_once(
    progress,
    '''        if forced_rag_tool is not None:
            if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_rag_tool:
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    f"Production coder violated host-forced RAG tool choice {forced_rag_tool!r}; received {called}."
                )
            forced_rag_tool = None
            forced_rag_attempts = 0

        messages.append({
''',
    '''        if forced_rag_tool is not None:
            if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_rag_tool:
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    f"Production coder violated host-forced RAG tool choice {forced_rag_tool!r}; received {called}."
                )
            forced_rag_tool = None
            forced_rag_attempts = 0

        if forced_verify_tool is not None:
            if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != forced_verify_tool:
                called = ", ".join(call.name for call in turn.tool_calls) or "<none>"
                raise ModelConfigurationError(
                    f"VERIFIER_PROTOCOL_VIOLATION: expected exactly {forced_verify_tool!r}; received {called}."
                )

        messages.append({
''',
    'verify protocol gate',
)

replace_once(
    progress,
    '''            if call.name in _VERIFY_TOOLS:
                status = _verification_outcome(call.name, payload)
                if status == "UNAVAILABLE":
                    # Runtime/tool failure is verifier health, not a source defect.
                    state.validation_status = "UNAVAILABLE"
                    state.record_failure(
                        call.name, payload.get("error", "verification unavailable")
                    )
                    continue
                if status != state.validation_status:
                    state.validation_status = status
                    turn_made_progress = True
                if status == "FAIL" and implementation_requires_mutation:
                    # Only trustworthy verifier evidence may request another edit.
                    state.record_failure(
                        call.name, "verification reported source defects"
                    )
                    state.phase = LoopPhase.ACT
                continue
''',
    '''            if call.name in _VERIFY_TOOLS:
                status = _verification_outcome(call.name, payload)
                if status == "UNAVAILABLE":
                    # Runtime/tool failure is verifier health, not a source defect.
                    # Retire this verifier for the current HostRunState instead of
                    # spending another model turn selecting it or unrelated research.
                    unavailable_verifiers.add(call.name)
                    state.validation_status = "UNAVAILABLE"
                    state.record_failure(
                        call.name, payload.get("error", "verification unavailable")
                    )
                    continue
                if status != state.validation_status:
                    state.validation_status = status
                    turn_made_progress = True
                if status == "FAIL" and implementation_requires_mutation:
                    # Only trustworthy verifier evidence may request another edit.
                    state.record_failure(
                        call.name, "verification reported source defects"
                    )
                    state.phase = LoopPhase.ACT
                continue
''',
    'retire unavailable verifier',
)

# 2) Any tool call, not only apply_source_edit, must have a viable output page.
budget = ROOT / 'minecraft_mod_ai/generation_output_budget.py'
replace_once(
    budget,
    '''_MIN_STRUCTURAL_TOOL_OUTPUT_TOKENS = 4096
# This is a liveness/safety floor, not a quality target. Deliberate static profiles may
# choose 512/1024-token scalar edit pages; accidental context collapse to 0/1 (or another
# tiny fragment) must fail before inference. Dynamic profiles still receive the larger
# 4096-token structural floor above.
_MIN_VIABLE_STRUCTURAL_TOOL_OUTPUT_TOKENS = 128
''',
    '''_MIN_STRUCTURAL_TOOL_OUTPUT_TOKENS = 4096
_MIN_GENERAL_TOOL_OUTPUT_TOKENS = 512
# This is a liveness/safety floor, not a quality target. Deliberate static profiles may
# choose 512/1024-token scalar edit pages; accidental context collapse to 0/1 (or another
# tiny fragment) must fail before inference. The rule applies to every model-facing tool
# action because a one-token verifier/retrieval call cannot encode valid function args.
_MIN_VIABLE_STRUCTURAL_TOOL_OUTPUT_TOKENS = 128
''',
    'general tool floor constant',
)

replace_once(
    budget,
    '''def _structural_tool_floor(config: Any, tools: Sequence[Any]) -> int:
    if _structural_tool_call_is_compact(tools):
        return max(
            1,
            min(
                _MIN_STRUCTURAL_TOOL_OUTPUT_TOKENS,
                tool_action_token_budget(config),
            ),
        )
    return 1
''',
    '''def _structural_tool_floor(config: Any, tools: Sequence[Any]) -> int:
    if not tools:
        return 1
    target = (
        _MIN_STRUCTURAL_TOOL_OUTPUT_TOKENS
        if _structural_tool_call_is_compact(tools)
        else _MIN_GENERAL_TOOL_OUTPUT_TOKENS
    )
    return max(1, min(target, tool_action_token_budget(config)))
''',
    'general tool dynamic floor',
)

replace_once(
    budget,
    '''    if not _structural_tool_call_is_compact(tools):
        return
    viable = min(
        _MIN_VIABLE_STRUCTURAL_TOOL_OUTPUT_TOKENS,
        max(1, tool_action_token_budget(config)),
    )
''',
    '''    if not tools:
        return
    viable = min(
        _MIN_VIABLE_STRUCTURAL_TOOL_OUTPUT_TOKENS,
        max(1, tool_action_token_budget(config)),
    )
''',
    'all tools viability check',
)

# 3) llama.cpp /apply-template is a root endpoint, not /v1/apply-template.
prefill = ROOT / 'minecraft_mod_ai/prefill_calibration_strictness_contract.py'
replace_once(
    prefill,
    '''    endpoint = f"{server_url.rstrip('/')}/apply-template"
''',
    '''    base_url = server_url.rstrip('/')
    if base_url.endswith('/v1'):
        base_url = base_url[:-3].rstrip('/')
    endpoint = f"{base_url}/apply-template"
''',
    'root apply-template endpoint',
)

# 4) The canonical tool loop already owns output recovery. Never restart router.generate_text
# outside it, because doing so creates a new HostRunState over a mutated staged workspace.
custom = ROOT / 'minecraft_mod_ai/custom_module_generator.py'
text = custom.read_text(encoding='utf-8')
pattern = re.compile(
    r'''        summary = ""\n        continuation_count = 0\n        seen_output_states: set\[str\] = set\(\)\n        active_messages = initial_messages\n        while True:\n.*?\n\n        operations, touched_paths, discarded_paths = _collect_staged_operations\(''',
    re.S,
)
replacement = '''        summary = ""
        continuation_count = 0
        try:
            with _active_checkpoint_persistence(
                checkpoint_root,
                staged_root,
                checkpoint_identity,
            ):
                summary = self.router.generate_text(
                    "coder",
                    initial_messages,
                    response_format="text",
                    tool_stage="generation",
                    enable_tools=True,
                )
            _persist_generation_checkpoint(
                checkpoint_root,
                staged_root,
                identity_sha256=checkpoint_identity,
            )
        except BaseException as exc:
            try:
                _persist_generation_checkpoint(
                    checkpoint_root,
                    staged_root,
                    identity_sha256=checkpoint_identity,
                )
            except (OSError, ValueError) as checkpoint_exc:
                print(
                    "custom module: checkpoint update failed",
                    f"module={module.module_id}",
                    f"error={type(checkpoint_exc).__name__}",
                    flush=True,
                )

            boundary_kind = completion_boundary_kind(exc)
            if boundary_kind != OUTPUT_EXHAUSTED:
                raise

            progress_operations, _progress_paths, _discarded_paths = (
                _collect_staged_operations(root, staged_root, before)
            )
            if progress_operations:
                self._validate_operations(progress_operations)
                self._validate_total_patch_bytes(progress_operations)
            raise CustomModuleGenerationError(
                "ATOMIC_ACTION_OUTPUT_STALLED: the canonical progress-aware coder loop exhausted "
                "its bounded in-state output recovery; refusing an outer continuation because it "
                "would reset HostRunState over an already-mutated staged workspace."
            ) from exc

        operations, touched_paths, discarded_paths = _collect_staged_operations('''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit(f'custom module continuation replacement count={count}')
custom.write_text(text, encoding='utf-8')

# Focused permanent regressions.
test = ROOT / 'tests/test_coder_liveness_boundaries.py'
test.write_text('''from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import generation_output_budget as budget
from minecraft_mod_ai import prefill_calibration_strictness_contract as prefill
from minecraft_mod_ai.progress_aware_tool_loop import LoopPhase, _filter_tools_for_phase


def _tool(name: str):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def test_verify_exposes_only_verifiers():
    tools = (
        _tool("java_diagnostics"),
        _tool("search_project_rag"),
        _tool("discover_ecosystem_resources"),
        _tool("inspect_github_repository"),
        _tool("apply_source_edit"),
    )
    selected = _filter_tools_for_phase(tools, LoopPhase.VERIFY, "coder")
    assert [item["function"]["name"] for item in selected] == ["java_diagnostics"]


def test_non_structural_tool_never_gets_one_token_static_budget(monkeypatch):
    monkeypatch.delenv("MMM_GENERATION_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TEXT_MAX_TOKENS", raising=False)
    config = SimpleNamespace(adapter="llama_cpp", max_new_tokens=1, extra={})
    with pytest.raises(budget.GenerationOutputBudgetError, match="OUTPUT_BUDGET_UNVIABLE"):
        budget.generation_output_token_budget(config, tools=(_tool("java_diagnostics"),))


def test_apply_template_strips_openai_v1_prefix(monkeypatch):
    seen = {}

    class _Response:
        status_code = 200

    class _Timeout:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _TimeoutException(Exception):
        pass

    def _post(url, *, json, timeout):
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return _Response()

    fake_httpx = SimpleNamespace(
        post=_post,
        Timeout=_Timeout,
        TimeoutException=_TimeoutException,
    )
    fake_module = SimpleNamespace(
        _positive_env_float=lambda _name, default: default,
        _DEFAULT_COMPLETION_TIMEOUT_SECONDS=120.0,
        _DEFAULT_HTTPX_POST=object(),
        httpx=fake_httpx,
    )

    response = prefill._post_apply_template(
        fake_module,
        "http://127.0.0.1:8910/v1",
        {"messages": []},
    )
    assert response.status_code == 200
    assert seen["url"] == "http://127.0.0.1:8910/apply-template"


def test_recover_does_not_expose_verifiers():
    tools = (_tool("java_diagnostics"), _tool("search_project_rag"))
    selected = _filter_tools_for_phase(tools, LoopPhase.RECOVER, "coder")
    assert [item["function"]["name"] for item in selected] == ["search_project_rag"]
''', encoding='utf-8')
