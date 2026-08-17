from types import SimpleNamespace

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


def test_json_requests_never_enable_native_llama_grammar():
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))
    request = SimpleNamespace(
        messages=({"role": "user", "content": "return JSON"},),
        tools=(),
        response_format="json",
    )
    payload = _server_payload(adapter, request)
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_llama_tool_turn_uses_native_qwen_tool_metadata(monkeypatch):
    from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract
    from minecraft_mod_ai.model_adapters import llama_cpp_adapter as llama_adapter_module

    tool = {
        "type": "function",
        "function": {
            "name": "search_project_rag",
            "description": "search",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    request = GenerationRequest(
        messages=({"role": "user", "content": "use the tool"},),
        tools=(tool,),
        tool_choice="auto",
        parallel_tool_calls=True,
        response_format="json",
    )
    adapter = LlamaCppAdapter(
        AdapterConfig(
            role="planner",
            adapter="llama_cpp",
            model_id="local-test",
            max_new_tokens=512,
        )
    )
    monkeypatch.setattr(adapter, "_server_url", lambda _request: "http://unit.test/v1")
    monkeypatch.setattr(stream_contract, "_report_server_connection", lambda _url: None)

    sent_payloads = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {
                                        "name": "search_project_rag",
                                        "arguments": '{"query":"x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }

    def fake_post(_url, *, json, timeout=None):
        del timeout
        sent_payloads.append(dict(json))
        return Response()

    monkeypatch.setattr(llama_adapter_module.httpx, "post", fake_post)
    turn = adapter.generate_turn(request)

    assert [call.name for call in turn.tool_calls] == ["search_project_rag"]
    assert turn.tool_calls[0].arguments == {"query": "x"}
    assert len(sent_payloads) == 1
    payload = sent_payloads[0]
    assert payload["tools"] == [tool]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    rendered = "\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "mmm/host-tool-envelope" not in rendered


def test_runtime_trajectory_retrieval_keeps_execution_context_contract():
    import inspect

    from minecraft_mod_ai import temporary_skill_contract, trajectory_memory

    current = trajectory_memory.relevant_trajectories
    depth = 0
    while current is not None:
        assert "current_context" in inspect.signature(
            current,
            follow_wrapped=False,
        ).parameters
        depth += 1
        current = getattr(current, "__wrapped__", None)
    assert depth >= 2
    assert temporary_skill_contract._trajectory_memory is trajectory_memory
    assert "relevant_trajectories" not in temporary_skill_contract.__dict__
