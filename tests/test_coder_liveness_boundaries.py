from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import generation_output_budget as budget
from minecraft_mod_ai import llama_exact_context
from minecraft_mod_ai import prefill_calibration_strictness_contract as prefill
from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    LlamaCompletionBoundaryError,
)
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


def test_exact_context_never_clamps_tool_action_to_one_token(monkeypatch):
    monkeypatch.setattr(
        llama_exact_context,
        "live_context_accounting",
        lambda _server_url, _payload: llama_exact_context.LiveContextAccounting(
            input_tokens=15360,
            context_tokens=15361,
        ),
    )
    payload = {
        "max_tokens": 512,
        "tools": [_tool("java_diagnostics")],
    }

    with pytest.raises(LlamaCompletionBoundaryError) as caught:
        llama_exact_context.capacity_safe_payload(
            "http://127.0.0.1:8910/v1",
            payload,
        )

    assert caught.value.kind == CONTEXT_PRESSURE
    assert caught.value.prompt_tokens == 15360
    assert caught.value.max_tokens == 1
    assert "remaining_tokens=1" in str(caught.value)
    assert "required_output_tokens=128" in str(caught.value)


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


def test_observe_keeps_retrieval_tools_after_verify_isolation():
    tools = (
        _tool("java_diagnostics"),
        _tool("search_project_rag"),
        _tool("inspect_github_repository"),
    )
    selected = _filter_tools_for_phase(tools, LoopPhase.OBSERVE, "coder")
    names = [item["function"]["name"] for item in selected]
    assert "search_project_rag" in names
    assert "inspect_github_repository" in names
    assert "java_diagnostics" not in names
