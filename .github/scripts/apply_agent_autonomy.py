from __future__ import annotations

from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def replace_section(text: str, start: str, end: str, new: str, label: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start)) if a >= 0 else -1
    if a < 0 or b < 0:
        raise SystemExit(f"{label}: section markers not found")
    return text[:a] + new + text[b:]


# ---------------------------------------------------------------------------
# 1. Model tool loop: the model owns completion. There is no host-owned round
#    or call count. Only an exact consecutive no-progress fixed point stops it.
# ---------------------------------------------------------------------------
p = Path("minecraft_mod_ai/model_router.py")
t = p.read_text(encoding="utf-8")
if "import hashlib\n" not in t:
    t = replace_once(t, "import json\n", "import hashlib\nimport json\n", "hashlib import")
t = t.replace("_DEFAULT_MAX_TOOL_ROUNDS = 8\n_DEFAULT_MAX_TOOL_CALLS = 24\n", "")

tool_loop = '''    def _generate_with_tools(
        self,
        *,
        adapter: Any,
        request: GenerationRequest,
        runtime: Any,
        stage: str,
    ) -> str:
        """Gather tool evidence until the model itself returns a final answer.

        No host-owned tool-round or tool-call ceiling exists. The only loop guard
        is semantic: two consecutive identical tool-call/result exchanges prove
        an exact no-progress fixed point.
        """
        messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
        previous_exchange_state: str | None = None
        round_index = 0

        while True:
            turn_request = GenerationRequest(
                messages=messages,
                media_paths=request.media_paths if round_index == 0 else (),
                response_format=request.response_format,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
            )
            turn = adapter.generate_turn(turn_request)
            if not turn.tool_calls:
                if not turn.content.strip():
                    raise ModelConfigurationError(
                        "Tool-capable model returned an empty final response."
                    )
                return turn.content.strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.raw_arguments
                                or json.dumps(
                                    dict(call.arguments),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                        for call in turn.tool_calls
                    ],
                }
            )

            observations: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                try:
                    result = runtime.call(stage, call.name, call.arguments)
                    payload: Mapping[str, Any] = {
                        "ok": True,
                        "tool": call.name,
                        "result": result,
                    }
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "tool": call.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                    }
                )
                observations.append(
                    {
                        "name": call.name,
                        "arguments": dict(call.arguments),
                        "observation": payload,
                    }
                )

            exchange_state = hashlib.sha256(
                json.dumps(
                    {
                        "assistant_content": turn.content or "",
                        "tool_exchanges": observations,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if exchange_state == previous_exchange_state:
                raise ModelConfigurationError(
                    "Agent reached an exact no-progress tool fixed point: identical "
                    "tool calls produced identical observations on consecutive turns."
                )
            previous_exchange_state = exchange_state
            round_index += 1

'''
t = replace_section(
    t,
    "    def _generate_with_tools(\n",
    "    def _tool_runtime(",
    tool_loop,
    "agent tool loop",
)
p.write_text(t, encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Planner semantic repair: preserve existing state/output cycle detection,
#    but remove the fixed two-attempt budget.
# ---------------------------------------------------------------------------
p = Path("minecraft_mod_ai/planner_incremental_resume_contract.py")
t = p.read_text(encoding="utf-8")
old_loop = '''        max_attempts = _bounded_env(
            "MMM_PLANNER_BATCH_REPAIR_ATTEMPTS",
            _DEFAULT_BATCH_REPAIR_ATTEMPTS,
            maximum=2,
        )
        seen_states: set[str] = set()
        seen_outputs: set[str] = set()
        last_output_sha256 = ""

        for attempt in range(1, max_attempts + 1):
'''
new_loop = '''        seen_states: set[str] = set()
        seen_outputs: set[str] = set()
        last_output_sha256 = ""
        attempt = 0

        while True:
            attempt += 1
'''
t = replace_once(t, old_loop, new_loop, "batch repair budget")

marker = "        _save_failed_patch(\n"
search_from = t.find("def _install_bounded_batch_repair")
removed_tail = False
while True:
    a = t.find(marker, search_from)
    if a < 0:
        break
    b = t.find("    patch_one_invalid_batch._mmm_bounded_semantic_batch_repair", a)
    if b < 0:
        break
    if 'reason="repair_budget_exhausted"' in t[a:b]:
        t = t[:a] + t[b:]
        removed_tail = True
        break
    search_from = a + len(marker)
if not removed_tail:
    raise SystemExit("batch repair exhaustion tail not found")
if "repair_budget_exhausted" in t or "range(1, max_attempts + 1)" in t:
    raise SystemExit("batch repair numeric ceiling still present")

# Remove now-dead batch-attempt configuration helper/constant.
t = t.replace("_DEFAULT_BATCH_REPAIR_ATTEMPTS = 2\n", "")
t = re.sub(
    r"\n\ndef _bounded_env\(name: str, default: int, \*, maximum: int\) -> int:\n"
    r"(?:    .*\n)+?(?=\n\ndef _batch_schema)",
    "",
    t,
    count=1,
)
if "_bounded_env(" in t:
    raise SystemExit("dead bounded-env planner helper still referenced")
if "import os\n" in t:
    t = t.replace("import os\n", "")

# Replace the >2 outline guard with semantic request+output fixed-point detection.
outline_guard = '''def _install_outline_cycle_guard(incremental_module: Any) -> None:
    from . import complete_planner

    current = complete_planner._generate_json_page_with_repair
    if getattr(current, "_mmm_outline_cycle_guard", False):
        return

    class _GuardedRouter:
        def __init__(self, router: Any) -> None:
            self._router = router
            self._seen_exchanges: set[str] = set()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._router, name)

        def generate_text(
            self,
            role: str,
            messages: Any,
            *,
            media_paths=(),
            response_format="text",
        ) -> str:
            system_content = ""
            user_content = ""
            if isinstance(messages, (list, tuple)) and messages:
                first = messages[0]
                last = messages[-1]
                if isinstance(first, dict):
                    system_content = str(first.get("content", ""))
                if isinstance(last, dict):
                    user_content = str(last.get("content", ""))

            is_batch_repair = (
                "field-level JSON patcher" in system_content
                or "regenerate exactly ONE invalid production batch" in system_content
            )
            output = self._router.generate_text(
                role,
                messages,
                media_paths=media_paths,
                response_format=response_format,
            )
            if not is_batch_repair:
                exchange = incremental_module._fingerprint(
                    {
                        "role": role,
                        "system": system_content,
                        "user": user_content,
                        "response_format": response_format,
                        "media_paths": [str(path) for path in media_paths],
                        "model_output": output,
                    }
                )
                if exchange in self._seen_exchanges:
                    raise complete_planner.SpecValidationError(
                        "Planner reached an identical request/response fixed point "
                        "without semantic progress."
                    )
                self._seen_exchanges.add(exchange)
            return output

    @wraps(current)
    def generate_cycle_safe(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        if not incremental_module._outline_allowed(expected_contracts):
            return current(
                router,
                system_prompt=system_prompt,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )
        return current(
            _GuardedRouter(router),
            system_prompt=system_prompt,
            request=request,
            media_paths=media_paths,
            expected_contracts=expected_contracts,
            stage=stage,
        )

    generate_cycle_safe._mmm_outline_cycle_guard = True  # type: ignore[attr-defined]
    generate_cycle_safe.__wrapped__ = current  # type: ignore[attr-defined]
    complete_planner._generate_json_page_with_repair = generate_cycle_safe


'''
t = replace_section(
    t,
    "def _install_outline_cycle_guard(",
    "def install(",
    outline_guard,
    "outline semantic guard",
)
p.write_text(t, encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. JSON pages and plan width: no total/page-count ceiling. A single response
#    remains finite, but remaining work continues through complete/next_cursor.
# ---------------------------------------------------------------------------
p = Path("minecraft_mod_ai/complete_planner.py")
t = p.read_text(encoding="utf-8")
json_repair = '''def _generate_json_page_with_repair(
    router: ModelRouter,
    *,
    system_prompt: str,
    request: dict[str, Any] | str,
    media_paths: Sequence[str | Path],
    expected_contracts: Sequence[frozenset[str]],
    stage: str,
) -> dict[str, Any]:
    """Repair a page until valid or the exact invalid state repeats."""
    request_text = (
        request
        if isinstance(request, str)
        else json.dumps(request, ensure_ascii=False)
    )
    seen_failures: set[str] = set()
    last_error = ""
    first_attempt = True
    while True:
        prompt = system_prompt
        if last_error:
            prompt += (
                " CRITICAL: The previous response was invalid or truncated. Output ONLY "
                "a JSON object matching the contract. Repair the validator error below. "
                "If this page is too large, emit fewer complete records on THIS page, "
                "set complete=false, and continue remaining work with next_cursor instead "
                "of dropping requirements. Validator error: " + last_error
            )
        text = router.generate_text(
            "planner",
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": request_text},
            ],
            media_paths=media_paths if first_attempt else (),
            response_format="json",
        )
        first_attempt = False
        try:
            return _extract_json(text, expected_contracts=expected_contracts)
        except SpecValidationError as exc:
            last_error = str(exc)
            failure = _canonical_json_sha256(
                {"model_output": text, "validation_error": last_error}
            )
            if failure in seen_failures:
                raise SpecValidationError(
                    f"{stage} reached an identical invalid JSON fixed point: {exc}"
                ) from exc
            seen_failures.add(failure)


'''
t = replace_section(
    t,
    "def _generate_json_page_with_repair(\n",
    "def _clean_json_text(",
    json_repair,
    "JSON page repair",
)

t = t.replace(
    "CRITICAL RULE: Emit EXACTLY ONE (1) production batch per response page. Never emit multiple batches in one go. If more batches or requirements remain, set complete=false and supply a next_cursor.\n",
    "Emit as many COMPLETE production batches as safely fit in this JSON page. There is no fixed batch-count width. If more batches or requirements remain, set complete=false and supply a next_cursor so planning continues across more JSON pages.\n",
)
t = t.replace(
    '                    "Generate up to 4 NEW production batches in one response to complete the plan efficiently. "',
    '                    "Generate as many NEW complete production batches as safely fit in this JSON page. "\n                    "There is no host-owned total or per-page batch-count ceiling; use next_cursor for remaining work. "',
)

width_old = '''            # Batch multiple deliverables per LLM call to minimize total calls
            batch_size = min(len(remaining), 4)
            target_deliverables = remaining[:batch_size]
            target_list_str = ", ".join(f'"{d}"' for d in target_deliverables)
'''
width_new = '''            # The model chooses current page width; host validates progress.
            target_deliverables = list(remaining)
'''
t = replace_once(t, width_old, width_new, "production deliverable width")
t = t.replace("        page_count = 0\n", "")
t = t.replace("            page_count += 1\n", "")

prompt_old = '''                system_prompt=(
                    "Return exactly one production-batch JSON page. "
                    f"Your task is to implement ALL of these deliverables in one response: [{target_list_str}]. "
                    f"Generate one module per deliverable ({batch_size} modules total). "
                    f"Put all completed deliverable names in completed_deliverables. "
                    f"There are {len(remaining)} deliverables total remaining. "
                    f"Set complete={'true' if len(remaining) <= batch_size else 'false'}. "
                    f"{'Set next_cursor to any non-empty string.' if len(remaining) > batch_size else 'Set next_cursor to empty string.'} "
                    "Never repeat an ID or file path."
                ),
'''
prompt_new = '''                system_prompt=(
                    "Return exactly one production-batch JSON page. "
                    "remaining_deliverables is the authoritative unfinished checklist. "
                    "Implement as many WHOLE deliverables as safely fit in this response; "
                    "there is no host-owned deliverable-count ceiling. Record only actually "
                    "completed items in completed_deliverables and include concrete evidence. "
                    "If unfinished work remains, set complete=false and provide next_cursor. "
                    "When everything is complete, set complete=true and next_cursor empty. "
                    "Never repeat an ID or file path."
                ),
'''
t = replace_once(t, prompt_old, prompt_new, "production page prompt")

remaining_old = "            remaining = [v for v in remaining if v not in completed_set]\n"
remaining_new = '''            before_state = _canonical_json_sha256(
                {
                    "remaining": remaining,
                    "modules": module_catalog.receipt(),
                    "assets": asset_catalog.receipt(),
                    "audio": audio_catalog.receipt(),
                    "test_count": len(test_catalog) - len(tests),
                }
            )
            remaining = [v for v in remaining if v not in completed_set]
            next_cursor_value = page.get("next_cursor")
            if isinstance(next_cursor_value, str) and next_cursor_value:
                cursor = next_cursor_value
            elif remaining:
                cursor = "host_resume_" + _canonical_json_sha256(
                    {
                        "batch_id": batch.batch_id,
                        "remaining": remaining,
                        "modules": module_catalog.receipt(),
                        "assets": asset_catalog.receipt(),
                        "audio": audio_catalog.receipt(),
                        "test_count": len(test_catalog),
                    }
                )[:20]
            after_state = _canonical_json_sha256(
                {
                    "remaining": remaining,
                    "modules": module_catalog.receipt(),
                    "assets": asset_catalog.receipt(),
                    "audio": audio_catalog.receipt(),
                    "test_count": len(test_catalog),
                }
            )
            if remaining and after_state == before_state:
                raise SpecValidationError(
                    "Production batch page reached an exact no-progress state."
                )
'''
t = replace_once(t, remaining_old, remaining_new, "production progress state")

cursor_replacement = '''            if next_cursor in seen_cursors:
                host_cursor = "host_resume_" + catalog.receipt()["sha256"][:20]
                if host_cursor in seen_cursors:
                    raise SpecValidationError(
                        "Production outline pagination repeated both cursor and catalog state."
                    )
                next_cursor = host_cursor
            seen_cursors.add(next_cursor)
'''
for old in (
    '''            if next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
''',
    '''            if next_cursor in seen_cursors:
                break  # pagination stalled
            seen_cursors.add(next_cursor)
''',
):
    if old in t:
        t = t.replace(old, cursor_replacement, 1)

if "EXACTLY ONE (1) production batch" in t or "Generate up to 4 NEW production batches" in t:
    raise SystemExit("fixed production-batch width still present")
if "batch_size = min(len(remaining), 4)" in t:
    raise SystemExit("fixed deliverable width still present")
p.write_text(t, encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Regression contracts for the user's intended autonomy semantics.
# ---------------------------------------------------------------------------
p = Path("tests/test_agent_tool_calling.py")
t = p.read_text(encoding="utf-8")
if "import pytest\n" not in t:
    t = t.replace("import json\n", "import json\n\nimport pytest\n", 1)
t = t.replace(
    "from minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall\n",
    "from minecraft_mod_ai.model_adapters import (\n    GenerationResponse,\n    ModelConfigurationError,\n    ToolCall,\n)\n",
)
if "test_agent_can_exceed_eight_tool_rounds" not in t:
    t += '''


def test_agent_can_exceed_eight_tool_rounds(monkeypatch) -> None:
    class LongAdapter:
        def __init__(self) -> None:
            self.count = 0

        def generate_turn(self, request):
            self.count += 1
            if self.count <= 12:
                query = f"evidence_{self.count}"
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id=f"call_{self.count}",
                            name="search_code_rag",
                            arguments={"query": query},
                            raw_arguments=json.dumps({"query": query}),
                        ),
                    )
                )
            return GenerationResponse(content="enough evidence")

    adapter = LongAdapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )
    assert router.generate_text(
        "coder", [{"role": "user", "content": "research deeply"}]
    ) == "enough evidence"
    assert len(runtime.calls) == 12


def test_agent_stops_on_consecutive_exact_tool_fixed_point(monkeypatch) -> None:
    class LoopAdapter:
        def generate_turn(self, request):
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="call",
                        name="search_code_rag",
                        arguments={"query": "same"},
                        raw_arguments='{"query":"same"}',
                    ),
                )
            )

    adapter = LoopAdapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )
    with pytest.raises(ModelConfigurationError, match="no-progress tool fixed point"):
        router.generate_text("coder", [{"role": "user", "content": "research"}])
    assert len(runtime.calls) == 2
'''
p.write_text(t, encoding="utf-8")

p = Path("tests/test_planner_incremental_repair_contract.py")
t = p.read_text(encoding="utf-8")
old = '''def test_broken_outline_envelope_is_cut_off_before_third_identical_request(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    # The host allows one diagnostic regeneration for the same semantic outline.
    router = _Router("{}", "{}")

    with pytest.raises(SpecValidationError, match="cycle detected"):
        _run(router, stage="outline cycle cut off")

    # Initial generation + one diagnostic regeneration. The third identical request is
    # rejected by the host before it reaches the model.
    assert len(router.calls) == 2
'''
new = '''def test_broken_outline_stops_only_at_exact_request_response_fixed_point(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    router = _Router("{}", "{}", "{}")
    with pytest.raises(SpecValidationError, match="fixed point"):
        _run(router, stage="outline fixed point")
    assert len(router.calls) == 3


def test_outline_can_keep_repairing_beyond_two_generations(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    final = _batch("eventual_valid_outline")
    router = _Router(
        '{"diagnostic":1}',
        '{"diagnostic":2}',
        '{"diagnostic":3}',
        '{"diagnostic":4}',
        _outline(final),
    )
    page = _run(router, stage="long progressive outline")
    assert page["production_batches"] == [final]
    assert len(router.calls) == 5
'''
t = replace_once(t, old, new, "outline regression tests")
p.write_text(t, encoding="utf-8")

print("progress-driven autonomy migration prepared")
