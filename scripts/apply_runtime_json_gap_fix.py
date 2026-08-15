from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, got {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


hw = "minecraft_mod_ai/llama_server_hardware_policy.py"
replace_once(
    hw,
    """    Tool-capable turns use the smallest widely compatible function-calling wire
    contract. Structured non-tool turns ask llama.cpp only for a generic JSON object
    and explicitly disable model-internal thinking. Detailed JSON Schema constraints
    stay on the host so llama.cpp never has to translate application schemas into
    fragile GBNF grammars.
""",
    """    Tool-capable turns use the smallest widely compatible function-calling wire
    contract. Structured non-tool turns deliberately do *not* send response_format,
    json_schema, or grammar to llama.cpp: those controls can compile through fragile
    server-side GBNF and fail before the model runs. JSON syntax/schema validation and
    isolated repair are host-owned. We only disable model-internal thinking here.
""",
)
replace_once(
    hw,
    """    if getattr(request, \"response_format\", None) == \"json\":
        payload[\"response_format\"] = {\"type\": \"json_object\"}
        # llama.cpp exposes both controls on /v1/chat/completions.  reasoning_effort
        # is the server-level hard disable while enable_thinking is consumed by Qwen
        # chat templates.  Supplying both makes the transport intent explicit and
        # prevents reasoning tokens from exhausting the visible structured response.
        payload[\"reasoning_effort\"] = \"none\"
        payload[\"chat_template_kwargs\"] = {\"enable_thinking\": False}
""",
    """    if getattr(request, \"response_format\", None) == \"json\":
        # Never ask llama.cpp to compile JSON/JSON-Schema into a sampler grammar.
        # JSON decoding, schema validation, and isolated repair are host-owned.
        payload[\"reasoning_effort\"] = \"none\"
        payload[\"chat_template_kwargs\"] = {\"enable_thinking\": False}
""",
)
replace_once(
    hw,
    '            structured = payload.get("response_format") is not None\n',
    '            structured = getattr(request, "response_format", None) == "json"\n',
)

guard = "minecraft_mod_ai/planning_stall_guard_contract.py"
replace_once(
    guard,
    '    _completed_domains: set[str] = field(default_factory=set, repr=False)\n',
    '    _completed_domains: set[str] = field(default_factory=set, repr=False)\n    _gap_domains: set[str] = field(default_factory=set, repr=False)\n',
)
replace_once(
    guard,
    """        total: Any = _UNSET,
        complete_domain: str | None = None,
    ) -> dict[str, Any]:
""",
    """        total: Any = _UNSET,
        complete_domain: str | None = None,
        gap_domain: str | None = None,
    ) -> dict[str, Any]:
""",
)
replace_once(
    guard,
    """            if complete_domain:
                self._completed_domains.add(_safe_progress_value(complete_domain))
            self.updated_at = time.monotonic()
""",
    """            if complete_domain:
                safe_complete = _safe_progress_value(complete_domain)
                self._completed_domains.add(safe_complete)
                self._gap_domains.discard(safe_complete)
            if gap_domain:
                safe_gap = _safe_progress_value(gap_domain)
                if safe_gap not in self._completed_domains:
                    self._gap_domains.add(safe_gap)
            self.updated_at = time.monotonic()
""",
)
replace_once(
    guard,
    '            "completed": len(self._completed_domains),\n            "total": self.total,\n',
    '            "completed": len(self._completed_domains),\n            "gaps": len(self._gap_domains),\n            "terminal": len(self._completed_domains | self._gap_domains),\n            "total": self.total,\n',
)
replace_once(
    guard,
    '        f" completed={snapshot[\'completed\']}"\n        f" total={total}"\n',
    '        f" completed={snapshot[\'completed\']}"\n        f" gaps={snapshot[\'gaps\']}"\n        f" terminal={snapshot[\'terminal\']}"\n        f" total={total}"\n',
)
replace_once(
    guard,
    """    completed_domain: str | None = None,
    total: int | None = None,
""",
    """    completed_domain: str | None = None,
    gap_domain: str | None = None,
    total: int | None = None,
""",
)
replace_once(
    guard,
    "    progress.record(complete_domain=completed_domain, **kwargs)\n",
    "    progress.record(complete_domain=completed_domain, gap_domain=gap_domain, **kwargs)\n",
)
replace_once(
    guard,
    """    terminal = event in {
        \"domain_checkpoint_complete\",
        \"domain_complete\",
        \"domain_gap_receipt\",
    }
    report_planner_research_progress(
""",
    """    successful_terminal = event in {
        \"domain_checkpoint_complete\",
        \"domain_complete\",
    }
    gap_terminal = event == \"domain_gap_receipt\"
    report_planner_research_progress(
""",
)
replace_once(
    guard,
    '        completed_domain=cursor.get("domain") if terminal else None,\n        total=_progress_int(payload.get("total"), minimum=1),\n',
    '        completed_domain=cursor.get("domain") if successful_terminal else None,\n        gap_domain=cursor.get("domain") if gap_terminal else None,\n        total=_progress_int(payload.get("total"), minimum=1),\n',
)
replace_once(
    guard,
    """        progress.record(stage=\"complete\", checkpoint=\"research-saved\")
        print(
            \"planner research: pre-design complete\",
            _progress_fields(progress),
            f\"elapsed={time.monotonic() - started:.1f}s\",
            flush=True,
        )
""",
    """        final_snapshot = progress.snapshot()
        gap_count = int(final_snapshot.get(\"gaps\", 0) or 0)
        if gap_count:
            progress.record(stage=\"complete-with-gaps\", checkpoint=\"research-saved-with-gaps\")
            final_label = \"planner research: pre-design terminal with gaps\"
        else:
            progress.record(stage=\"complete\", checkpoint=\"research-saved\")
            final_label = \"planner research: pre-design complete\"
        print(
            final_label,
            _progress_fields(progress),
            f\"elapsed={time.monotonic() - started:.1f}s\",
            flush=True,
        )
""",
)

Path("tests/test_runtime_json_gap_regression.py").write_text(
    '''from types import SimpleNamespace\n\nfrom minecraft_mod_ai.llama_server_hardware_policy import _server_payload\nfrom minecraft_mod_ai import planning_stall_guard_contract as guard\n\n\ndef test_json_requests_never_enable_native_llama_grammar():\n    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))\n    request = SimpleNamespace(messages=({"role": "user", "content": "return JSON"},), tools=(), response_format="json")\n    payload = _server_payload(adapter, request)\n    assert "response_format" not in payload\n    assert "json_schema" not in payload\n    assert "grammar" not in payload\n    assert payload["reasoning_effort"] == "none"\n    assert payload["chat_template_kwargs"] == {"enable_thinking": False}\n\n\ndef test_terminal_gap_is_not_counted_as_verified_completion():\n    progress = guard._PlanningProgress(total=2)\n    progress_token = guard._ACTIVE_PROGRESS.set(progress)\n    cursor_token = guard._ACTIVE_PROGRESS_CURSOR.set(None)\n    try:\n        guard._research_progress_hook({"event": "domain_gap_receipt", "domain_id": "broken-domain", "page_index": 2, "page_count": 2})\n        snapshot = progress.snapshot()\n        assert snapshot["completed"] == 0\n        assert snapshot["gaps"] == 1\n        assert snapshot["terminal"] == 1\n        guard._research_progress_hook({"event": "domain_complete", "domain_id": "verified-domain", "page_index": 1, "page_count": 1})\n        snapshot = progress.snapshot()\n        assert snapshot["completed"] == 1\n        assert snapshot["gaps"] == 1\n        assert snapshot["terminal"] == 2\n    finally:\n        guard._ACTIVE_PROGRESS_CURSOR.reset(cursor_token)\n        guard._ACTIVE_PROGRESS.reset(progress_token)\n''',
    encoding="utf-8",
)
