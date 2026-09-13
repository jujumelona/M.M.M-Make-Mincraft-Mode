from __future__ import annotations

import re
from pathlib import Path

CORE = Path('minecraft_mod_ai/progress_aware_tool_loop.py')
TEST = Path('tests/test_progress_aware_semantic_convergence.py')
text = CORE.read_text(encoding='utf-8')


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected exactly one match, found {count}')
    text = text.replace(old, new, 1)


# Remove the global tool-round budget from this state machine. Termination is based on
# semantic state recurrence / route exhaustion instead of a turn count.
replace_once('        _agent_tool_round_limit,\n', '', 'round-limit import')
replace_once('    round_limit = _agent_tool_round_limit()\n', '', 'round-limit initialization')
replace_once('            "round_limit": round_limit,\n', '', 'round-limit trace')

# HostRunState owns semantic convergence memory. Keep the legacy integer observation
# field only for compatibility/telemetry; it is no longer a threshold or counter.
replace_once(
    '    no_progress_streak: int = 0\n'
    '    attempted_queries: set[str] = field(default_factory=set)\n',
    '    no_progress_streak: int = 0\n'
    '    seen_no_progress_digests: set[str] = field(default_factory=set)\n'
    '    semantic_fixed_point: bool = False\n'
    '    attempted_queries: set[str] = field(default_factory=set)\n',
    'semantic convergence fields',
)
replace_once('    last_result_digest: str | None = None\n', '', 'obsolete last-result digest')

pattern = re.compile(
    r'    def record_no_progress_result\(self, value: Any\) -> int:\n.*?'
    r'(?=    def next_untried_internal_tool\()',
    re.DOTALL,
)
new_block = '''    def record_no_progress_result(self, value: Any) -> bool:\n        """Return True only when the same stable semantic state recurs.\n\n        A new action/result frontier is not convergence, regardless of how many turns\n        have elapsed. Any material progress clears this recurrence memory.\n        """\n\n        stable = _stable_value(value, drop_volatile=True)\n        canonical = json.dumps(\n            stable,\n            ensure_ascii=False,\n            sort_keys=True,\n            separators=(",", ":"),\n            default=str,\n        )\n        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()\n        with self._lock:\n            repeated = digest in self.seen_no_progress_digests\n            self.seen_no_progress_digests.add(digest)\n            self.semantic_fixed_point = repeated\n            # Compatibility/telemetry bit only; never used as a numeric threshold.\n            self.no_progress_streak = int(repeated)\n            return repeated\n\n    def clear_no_progress_result(self) -> None:\n        with self._lock:\n            self.seen_no_progress_digests.clear()\n            self.semantic_fixed_point = False\n            self.no_progress_streak = 0\n\n'''
text, count = pattern.subn(lambda _m: new_block, text, count=1)
if count != 1:
    raise SystemExit(f'record_no_progress_result block: expected one match, found {count}')

# Delete the entire numeric round-limit termination branch.
pattern = re.compile(
    r'\n        if state\.step_index > round_limit:\n.*?'
    r'(?=\n        if forced_rag_tool is None and state\.no_progress_streak >= 2:)',
    re.DOTALL,
)
text, count = pattern.subn('', text, count=1)
if count != 1:
    raise SystemExit(f'round-limit branch: expected one match, found {count}')

replace_once(
    '        if forced_rag_tool is None and state.no_progress_streak >= 2:\n',
    '        if forced_rag_tool is None and state.semantic_fixed_point:\n',
    'fixed-point predicate',
)
text = text.replace(
    'f"[HOST NO-PROGRESS BOUNDARY HIT] Step={state.step_index} Streak={state.no_progress_streak}\\n"',
    'f"[HOST SEMANTIC FIXED POINT] Step={state.step_index}\\n"',
)

# A forced tool choice is already a deterministic protocol contract. If the model
# returns prose instead, fail immediately; retry counts are neither necessary nor useful.
text = text.replace('    forced_rag_attempts = 0\n', '')
pattern = re.compile(
    r'            if forced_rag_tool is not None:\n'
    r'                forced_rag_attempts \+= 1\n.*?'
    r'(?=            if require_rag and not state\.has_fresh_evidence:)',
    re.DOTALL,
)
forced_block = '''            if forced_rag_tool is not None:\n                raise ModelConfigurationError(\n                    f"Production coder violated host-forced RAG tool choice {forced_rag_tool!r} "\n                    "by returning prose instead of the required tool call."\n                )\n'''
text, count = pattern.subn(lambda _m: forced_block, text, count=1)
if count != 1:
    raise SystemExit(f'forced-RAG prose block: expected one match, found {count}')

# Remove obsolete counter resets left in semantic route selection / completion.
text = text.replace('                    forced_rag_attempts = 0\n', '')
text = text.replace('                        forced_rag_attempts = 0\n', '')
text = text.replace('            forced_rag_attempts = 0\n', '')

# A verifier FAIL prose turn terminates only if that exact semantic state was already
# seen, not after an arbitrary number of repeats.
replace_once(
    '                repeated = state.record_no_progress_result(\n'
    '                    {\n'
    '                        "phase": LoopPhase.ACT.value,\n'
    '                        "validation_status": "FAIL",\n'
    '                        "verifier_fingerprint": state.latest_verifier_fingerprint,\n'
    '                        "prose": content,\n'
    '                    }\n'
    '                )\n'
    '                if repeated > 1:\n',
    '                repeated = state.record_no_progress_result(\n'
    '                    {\n'
    '                        "phase": LoopPhase.ACT.value,\n'
    '                        "validation_status": "FAIL",\n'
    '                        "verifier_fingerprint": state.latest_verifier_fingerprint,\n'
    '                        "prose": content,\n'
    '                    }\n'
    '                )\n'
    '                if repeated:\n',
    'verifier prose recurrence',
)

# Semantic identity must describe state/action/result, not incidental raw argument text.
# Tool result identity already includes tool name + verifier outcome / failure code.
replace_once(
    '                    "model_tool_calls": model_calls_info,\n'
    '                    "validation_status": state.validation_status,\n',
    '                    "validation_status": state.validation_status,\n',
    'fixed-point raw model arguments',
)

# Trace names remain backward-compatible, but runtime decisions must contain no numeric cap.
for forbidden in (
    '_agent_tool_round_limit',
    'state.step_index > round_limit',
    'state.no_progress_streak >= 2',
    'repeated > 1',
    'forced_rag_attempts',
    'after one bounded forced attempt',
):
    if forbidden in text:
        raise SystemExit(f'forbidden numeric convergence contract remains: {forbidden}')

CORE.write_text(text, encoding='utf-8')

TEST.write_text('''from __future__ import annotations\n\nfrom minecraft_mod_ai.progress_aware_tool_loop import HostRunState\n\n\ndef test_semantic_state_recurs_without_numeric_retry_threshold():\n    state = HostRunState()\n    semantic = {\n        "phase": "ACT",\n        "target": "DebugToken.java",\n        "verifier_fingerprint": "diag-a",\n        "tool_results": [{"name": "apply_source_edit", "failure_code": "MUTATION_UNCHANGED"}],\n    }\n    assert state.record_no_progress_result(semantic) is False\n    assert state.semantic_fixed_point is False\n    assert state.record_no_progress_result(semantic) is True\n    assert state.semantic_fixed_point is True\n\n\ndef test_new_semantic_frontier_is_not_convergence():\n    state = HostRunState()\n    first = {"phase": "ACT", "verifier_fingerprint": "diag-a", "tool_results": [{"name": "apply_source_edit", "failure_code": "A"}]}\n    second = {"phase": "ACT", "verifier_fingerprint": "diag-b", "tool_results": [{"name": "apply_source_edit", "failure_code": "A"}]}\n    assert state.record_no_progress_result(first) is False\n    assert state.record_no_progress_result(second) is False\n    assert state.semantic_fixed_point is False\n\n\ndef test_material_progress_clears_semantic_recurrence_memory():\n    state = HostRunState()\n    semantic = {"phase": "ACT", "verifier_fingerprint": "diag-a", "tool_results": [{"name": "apply_source_edit", "failure_code": "A"}]}\n    assert state.record_no_progress_result(semantic) is False\n    assert state.record_no_progress_result(semantic) is True\n    state.clear_no_progress_result()\n    assert state.semantic_fixed_point is False\n    assert state.record_no_progress_result(semantic) is False\n\n\ndef test_semantic_cycle_detection_catches_nonconsecutive_recurrence():\n    state = HostRunState()\n    a = {"phase": "OBSERVE", "tool_results": [{"name": "search_code_rag", "failure_code": "EMPTY"}]}\n    b = {"phase": "OBSERVE", "tool_results": [{"name": "java_workspace_symbols", "failure_code": "EMPTY"}]}\n    assert state.record_no_progress_result(a) is False\n    assert state.record_no_progress_result(b) is False\n    assert state.record_no_progress_result(a) is True\n''', encoding='utf-8')

print('semantic convergence migration prepared')
