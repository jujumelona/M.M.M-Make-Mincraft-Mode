from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import runtime_stability_contract as contract


def test_tool_schema_projection_resolves_refs_and_drops_grammar_hazards():
    tool = {
        "type": "function",
        "function": {
            "name": "external_mcp_call",
            "description": "Call MCP",
            "strict": True,
            "parameters": {
                "$defs": {
                    "Request": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "pattern": "(?s).+",
                                "minLength": 1,
                            },
                            "limit": {
                                "anyOf": [
                                    {"type": "integer", "minimum": 1},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    }
                },
                "$ref": "#/$defs/Request",
            },
        },
    }
    safe = contract._grammar_safe_tool(tool)
    params = safe["function"]["parameters"]
    assert params == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }
    assert "strict" not in safe["function"]
    assert tool["function"]["parameters"]["$ref"] == "#/$defs/Request"


def test_tool_schema_is_projected_before_first_request_without_retry_layer():
    def raw_payload(adapter, request):
        del adapter, request
        return {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_project_rag",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string", "pattern": ".+"}},
                            "required": ["q"],
                        },
                    },
                }
            ]
        }

    policy = SimpleNamespace(_server_payload=raw_payload)
    contract._install_llama_tool_schema_projection(policy)
    payload = policy._server_payload(None, None)

    assert getattr(policy._server_payload, "_mmm_grammar_safe_tools_v1", False)
    assert payload["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }


def test_http_400_transport_is_not_wrapped_or_retried_by_stability_contract():
    class Response:
        status_code = 400
        text = '{"error":{"message":"Failed to initialize samplers: failed to parse grammar"}}'

    calls = 0

    def raw_post(server_url, payload):
        nonlocal calls
        del server_url, payload
        calls += 1
        return Response()

    adapter = SimpleNamespace(_post_completion=raw_post)
    original_post = adapter._post_completion
    policy = SimpleNamespace(_server_payload=lambda adapter, request: {})

    contract._install_llama_tool_schema_projection(policy)
    assert adapter._post_completion is original_post

    response = adapter._post_completion("http://localhost", {})
    assert response.status_code == 400
    assert calls == 1
