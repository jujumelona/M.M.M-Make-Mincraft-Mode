from __future__ import annotations

import threading
from types import SimpleNamespace

import minecraft_mod_ai.agentic_research_game_design as agentic
import minecraft_mod_ai.pre_design_grounded_rag as forced_rag
import minecraft_mod_ai.llama_structured_decode_policy as decode_policy
import minecraft_mod_ai.llama_tuning_pipeline as tuning_pipeline


def test_forced_rag_context_is_isolated_across_concurrent_plans() -> None:
    """Concurrent plans carry bounded evidence through arguments, never global context."""

    barrier = threading.Barrier(2)
    observed: dict[str, str] = {}
    failures: list[BaseException] = []
    lock = threading.Lock()

    def run(owner: str) -> None:
        deterministic = {
            "forced_project_rag": {
                "schema_version": "mmm/forced-pre-design-rag-v2",
                "research_sha256": f"sha256:{owner}",
                "domain_count": 1,
                "query_count": 1,
                "project_source_count": 1,
                "domains": [
                    {
                        "domain_id": "request",
                        "queries": [{"query": owner, "raw": owner * 20_000}],
                    }
                ],
            }
        }
        try:
            barrier.wait(timeout=2)
            result = agentic._domain_evidence_slice("request", deterministic)
            receipt = result["forced_project_rag"]
            assert receipt["research_sha256"] == f"sha256:{owner}"
            assert owner * 20_000 not in str(result)
            with lock:
                observed[owner] = str(receipt["research_sha256"])
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
    assert observed == {
        "alpha": "sha256:alpha",
        "beta": "sha256:beta",
    }
    assert not hasattr(forced_rag, "_FORCED_RAG_CONTEXT")
    assert not hasattr(forced_rag, "harden_pre_design_research")


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


def test_native_tuning_pipeline_keeps_single_graph_owned_stage_order() -> None:
    pipeline = tuning_pipeline.NativeLlamaTuningPipeline(
        autotune=SimpleNamespace(),
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    assert tuple(stage.name for stage in pipeline.stages()) == (
        "runtime-types",
        "hardware",
        "efficiency",
        "runtime",
        "cache-reuse",
        "decode-speed",
        "kernel-autotune",
        "qwen-transport",
        "multimodal",
    )
