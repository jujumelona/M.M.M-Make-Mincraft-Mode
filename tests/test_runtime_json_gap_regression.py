from types import SimpleNamespace

import pytest

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


def _source_edit_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one narrow semantic source edit.",
            "parameters": SOURCE_EDIT_SCHEMA,
        },
    }


def _tool_request(tool: dict) -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "use the tool"},),
        tools=(tool,),
        tool_choice="auto",
        parallel_tool_calls=False,
        response_format="json",
    )


def test_qwen35_json_requests_keep_validation_host_side_without_grammar():
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            max_new_tokens=512,
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        )
    )
    request = SimpleNamespace(
        messages=({"role": "user", "content": "return JSON"},),
        tools=(),
        response_format="json",
    )
    payload = _server_payload(adapter, request)
    for forbidden in ("response_format", "json_schema", "grammar"):
        assert forbidden not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_llama_tool_turn_host_parses_qwen_markup_without_server_peg(monkeypatch):
    from minecraft_mod_ai import llama_exact_context
    from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract
    from minecraft_mod_ai.model_adapters import (
        llama_cpp_adapter as llama_adapter_module,
    )

    tool = {
        "type": "function",
        "function": {
            "name": "search_project_rag",
            "description": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
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
            model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
            max_new_tokens=512,
        )
    )
    monkeypatch.setattr(adapter, "_server_url", lambda _request: "http://unit.test/v1")
    monkeypatch.setattr(
        llama_exact_context,
        "capacity_safe_payload",
        lambda _url, payload: dict(payload),
    )
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
                            "content": (
                                "<tool_call><function=search_project_rag>"
                                "<parameter=query>x</parameter>"
                                "</function></tool_call>"
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

    def fake_post(_url, *, json, timeout=None):
        del timeout
        sent_payloads.append(dict(json))
        return Response()

    monkeypatch.setattr(llama_adapter_module.httpx, "post", fake_post)
    turn = adapter.generate_turn(request)

    assert request.tool_choice == "auto"
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


def test_qwen_canonical_permission_name_maps_back_to_exposed_source_edit():
    from minecraft_mod_ai.model_adapters import (
        llama_cpp_adapter as llama_adapter_module,
    )

    request = _tool_request(_source_edit_tool())
    turn = llama_adapter_module._qwen_tool_generation_response(
        {
            "content": (
                "<tool_call><function=apply_source_patch>"
                "<parameter=action>delete_file</parameter>"
                "<parameter=file>src/main/java/example/Old.java</parameter>"
                "</function></tool_call>"
            )
        },
        request,
    )

    assert [call.name for call in turn.tool_calls] == ["apply_source_edit"]
    assert turn.tool_calls[0].arguments == {
        "operation": "delete_file",
        "path": "src/main/java/example/Old.java",
    }


def test_qwen_canonical_tool_name_does_not_revive_removed_whole_file_operation():
    from minecraft_mod_ai.model_adapters import (
        llama_cpp_adapter as llama_adapter_module,
    )

    request = _tool_request(_source_edit_tool())
    with pytest.raises(RuntimeError, match="value outside enum"):
        llama_adapter_module._qwen_tool_generation_response(
            {
                "content": (
                    "<tool_call><function=apply_source_patch>"
                    "<parameter=operation>update_file</parameter>"
                    "<parameter=path>src/main/java/example/Old.java</parameter>"
                    "</function></tool_call>"
                )
            },
            request,
        )


def test_qwen_canonical_tool_name_still_rejects_broad_patch_payload():
    from minecraft_mod_ai.model_adapters import (
        llama_cpp_adapter as llama_adapter_module,
    )

    request = _tool_request(_source_edit_tool())
    with pytest.raises(RuntimeError, match="unknown parameter 'patch'"):
        llama_adapter_module._qwen_tool_generation_response(
            {
                "content": (
                    "<tool_call><function=apply_source_patch>"
                    "<parameter=operation>replace_exact</parameter>"
                    "<parameter=path>src/main/java/example/Old.java</parameter>"
                    "<parameter=patch>whole file</parameter>"
                    "</function></tool_call>"
                )
            },
            request,
        )


def test_qwen_canonical_tool_name_is_preserved_when_alias_is_not_exposed():
    from minecraft_mod_ai.model_adapters import (
        llama_cpp_adapter as llama_adapter_module,
    )

    search_tool = {
        "type": "function",
        "function": {
            "name": "search_code_rag",
            "description": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
    request = _tool_request(search_tool)
    turn = llama_adapter_module._qwen_tool_generation_response(
        {
            "content": (
                "<tool_call><function=apply_source_patch>"
                "<parameter=operation>delete_file</parameter>"
                "<parameter=path>src/main/java/example/Old.java</parameter>"
                "</function></tool_call>"
            )
        },
        request,
    )
    assert [call.name for call in turn.tool_calls] == ["apply_source_patch"]
    assert turn.tool_calls[0].arguments == {
        "operation": "delete_file",
        "path": "src/main/java/example/Old.java",
    }


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
