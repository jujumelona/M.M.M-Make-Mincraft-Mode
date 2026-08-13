from __future__ import annotations

import threading
from types import SimpleNamespace

import minecraft_mod_ai.agentic_pre_design_rag as forced_rag
import minecraft_mod_ai.llama_structured_decode_policy as decode_policy
import minecraft_mod_ai.llama_tuning_pipeline as tuning_pipeline


def test_forced_rag_context_is_isolated_across_concurrent_plans(monkeypatch) -> None:
    barrier = threading.Barrier(2)
    observed: dict[str, str] = {}
    agentic = SimpleNamespace()
    agentic.normalize_research_brief = lambda prompt, _design: {
        "domains": [{"domain_id": "request", "queries": [prompt]}]
    }
    agentic._domain_evidence_slice = lambda domain_id, deterministic: {
        "base": domain_id,
        **dict(deterministic),
    }
    agentic._research_receipt = lambda value: value
    agentic._research_domain_with_agent = lambda *args, **kwargs: {}

    def original_collect(router, prompt, *, trace_metadata=None):
        barrier.wait(timeout=2)
        forced = forced_rag._FORCED_RAG_CONTEXT.get()
        assert forced is not None
        observed[prompt] = str(forced["owner"])
        return {
            "research_brief": {"domains": []},
            "deterministic": {},
            "domain_notes": [],
            "errors": [],
        }

    agentic.collect_pre_design_research = original_collect
    monkeypatch.setattr(
        forced_rag,
        "_forced_rag_bundle",
        lambda _router, brief: {
            "owner": brief["domains"][0]["queries"][0],
            "domains": [
                {"domain_id": "request", "queries": []},
            ],
        },
    )
    forced_rag.harden_pre_design_research(agentic)

    failures: list[BaseException] = []

    def run(prompt: str) -> None:
        try:
            agentic.collect_pre_design_research(object(), prompt)
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=run, args=("alpha",)),
        threading.Thread(target=run, args=("beta",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not failures
    assert observed == {"alpha": "alpha", "beta": "beta"}


def test_bounded_section_disables_thinking_without_touching_research_tools() -> None:
    class _Hardware:
        @staticmethod
        def _server_payload(adapter, request):
            payload = {
                "model": "local",
                "messages": list(request.messages),
                "max_tokens": adapter.config.max_new_tokens,
            }
            if request.tools:
                payload["tools"] = list(request.tools)
            if request.response_format == "json":
                payload["response_format"] = {"type": "json_object"}
            return payload

    decode_policy.bind_structured_decode_policy(_Hardware)
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=8192))

    section_request = SimpleNamespace(
        messages=({"role": "user", "content": "serialize section"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"section": {"type": "object"}},
        },
        tools=(),
    )
    section_payload = _Hardware._server_payload(adapter, section_request)
    assert section_payload["thinking_budget_tokens"] == 0
    assert section_payload["reasoning_effort"] == "none"

    research_request = SimpleNamespace(
        messages=({"role": "user", "content": "research"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"research_note": {"type": "object"}},
        },
        tools=(
            {
                "type": "function",
                "function": {"name": "search_project_rag", "parameters": {}},
            },
        ),
    )
    research_payload = _Hardware._server_payload(adapter, research_request)
    assert "thinking_budget_tokens" not in research_payload
    assert "reasoning_effort" not in research_payload
    assert research_payload["tools"]

    generic_json_request = SimpleNamespace(
        messages=({"role": "user", "content": "other json"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"game_design": {"type": "object"}},
        },
        tools=(),
    )
    generic_payload = _Hardware._server_payload(adapter, generic_json_request)
    assert "thinking_budget_tokens" not in generic_payload
    assert "reasoning_effort" not in generic_payload


def test_native_tuning_pipeline_keeps_single_stage_order_and_is_versioned() -> None:
    assert tuning_pipeline._TUNING_PIPELINE_VERSION >= 2
    pipeline = tuning_pipeline.NativeLlamaTuningPipeline(
        autotune=SimpleNamespace(),
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    assert tuple(stage.name for stage in pipeline.stages()) == (
        "hardware",
        "efficiency",
        "runtime",
        "cache-reuse",
        "decode-speed",
    )
