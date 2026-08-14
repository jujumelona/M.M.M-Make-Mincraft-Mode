from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import runtime_stability_contract as contract


def _rich_note(domain_id: str = "mk_platform") -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "claims": [
            {
                "claim": "한글 근거가 매우 긴 합성 결과입니다. " * 40,
                "evidence_refs": ["ledger://page/" + ("가" * 100)],
            }
            for _ in range(6)
        ],
        "gaps": ["남은 공백 " * 100 for _ in range(4)],
        "next_queries": ["후속 검색 " * 100 for _ in range(4)],
        "sufficient": False,
    }


def test_synthesis_nodes_are_utf8_bounded_and_pairwise_contracting():
    compact = contract._compact_synthesis_note(_rich_note())
    assert contract._json_bytes(compact) <= contract._SYNTHESIS_NODE_BYTES

    module = SimpleNamespace(
        _SYNTHESIS_PROTOCOL_SCHEMA="mmm/research-hierarchical-synthesis-v2",
        _SYNTHESIS_INPUT_BYTES=3600,
        _synthesize_group_with_recovery=lambda *args, **kwargs: [],
        _emit_research_progress=lambda *args, **kwargs: None,
    )
    contract._install_synthesis_convergence(module)
    groups = module._group_synthesis_notes([_rich_note() for _ in range(5)])
    assert len(groups) == 3
    assert all(len(group) <= 2 for group in groups)
    assert all(contract._json_bytes(group) <= 3600 for group in groups)
    assert module._SYNTHESIS_PROTOCOL_SCHEMA.endswith("-v3")


def test_valid_large_synthesis_outputs_terminate_without_failure_signal():
    calls: list[tuple[int, int]] = []

    def synthesize(*args, **kwargs):
        group = kwargs["group"]
        level = kwargs["level"]
        calls.append((level, len(group)))
        return [_rich_note()]

    module = SimpleNamespace(
        _SYNTHESIS_PROTOCOL_SCHEMA="mmm/research-hierarchical-synthesis-v2",
        _SYNTHESIS_INPUT_BYTES=3600,
        _synthesize_group_with_recovery=synthesize,
        _emit_research_progress=lambda *args, **kwargs: None,
    )
    contract._install_synthesis_convergence(module)

    result = module._hierarchical_synthesis(
        None,
        None,
        prompt="make platform",
        domain={"domain_id": "mk_platform"},
        page_notes=[_rich_note() for _ in range(4)],
        domain_key="unit-test",
        failures=[],
    )

    assert result["domain_id"] == "mk_platform"
    assert calls == [(0, 2), (0, 2), (1, 2)]
    assert contract._json_bytes(result) <= contract._SYNTHESIS_NODE_BYTES


def test_recovery_expansion_is_collapsed_on_host_before_next_model_level():
    frontier_sizes: list[int] = []

    def synthesize(*args, **kwargs):
        group = kwargs["group"]
        frontier_sizes.append(len(group))
        # Simulate validated recovery children that would otherwise keep/expand work.
        return [_rich_note(), _rich_note()]

    module = SimpleNamespace(
        _SYNTHESIS_PROTOCOL_SCHEMA="v2",
        _SYNTHESIS_INPUT_BYTES=3600,
        _synthesize_group_with_recovery=synthesize,
        _emit_research_progress=lambda *args, **kwargs: None,
    )
    contract._install_synthesis_convergence(module)
    result = module._hierarchical_synthesis(
        None,
        None,
        prompt="x",
        domain={"domain_id": "mk_platform"},
        page_notes=[_rich_note() for _ in range(4)],
        domain_key="unit-test",
        failures=[],
    )
    assert result["domain_id"] == "mk_platform"
    assert len(frontier_sizes) < 10


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


def test_exact_llama_grammar_400_retries_once_with_minimal_host_validated_schema():
    class Response:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

    calls: list[dict[str, object]] = []

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

    responses = [
        Response(400, '{"error":{"message":"Failed to initialize samplers: failed to parse grammar"}}'),
        Response(200, "ok"),
    ]

    def raw_post(server_url, payload):
        del server_url
        calls.append(payload)
        return responses.pop(0)

    policy = SimpleNamespace(_server_payload=raw_payload)
    adapter = SimpleNamespace(_post_completion=raw_post)
    contract._install_llama_grammar_safety(policy, adapter)
    payload = policy._server_payload(None, None)
    response = adapter._post_completion("http://127.0.0.1:8910/v1", payload)

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0]["tools"][0]["function"]["parameters"]["properties"]["q"] == {"type": "string"}
    assert calls[1]["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_unrelated_http_400_is_never_retried():
    class Response:
        status_code = 400
        text = "bad request"

    calls = 0

    def raw_post(server_url, payload):
        nonlocal calls
        del server_url, payload
        calls += 1
        return Response()

    policy = SimpleNamespace(_server_payload=lambda adapter, request: {})
    adapter = SimpleNamespace(_post_completion=raw_post)
    contract._install_llama_grammar_safety(policy, adapter)
    response = adapter._post_completion("http://localhost", {})
    assert response.status_code == 400
    assert calls == 1
