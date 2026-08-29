from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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


def _research_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "research_note": {
                "type": "object",
                "properties": {
                    "domain_id": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "evidence_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["claim", "evidence_refs"],
                            "additionalProperties": False,
                        },
                    },
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "next_queries": {"type": "array", "items": {"type": "string"}},
                    "sufficient": {"type": "boolean"},
                },
                "required": ["domain_id", "claims", "gaps", "next_queries", "sufficient"],
                "additionalProperties": False,
            }
        },
        "required": ["research_note"],
        "additionalProperties": False,
    }


def _bounded_module():
    class BoundedError(RuntimeError):
        pass

    return SimpleNamespace(
        _SYNTHESIS_INPUT_BYTES=3600,
        _MIN_CONTINUATION_PROGRESS_CHARS=512,
        _BoundedResearchOutputError=BoundedError,
        _emit_research_progress=lambda *args, **kwargs: None,
    )


def test_bounded_research_has_one_outer_generation_and_host_bound_schema():
    class SpecValidationError(RuntimeError):
        pass

    calls: list[dict[str, object]] = []

    class Router:
        def generate_text(self, role, messages, **kwargs):
            calls.append({"role": role, "messages": messages, **kwargs})
            return json.dumps(
                {
                    "research_note": {
                        "domain_id": "request",
                        "claims": [
                            {"claim": "supported", "evidence_refs": ["page:1"]}
                        ],
                        "gaps": [],
                        "next_queries": [],
                        "sufficient": True,
                    }
                }
            )

    module = _bounded_module()
    contract._install_bounded_research_efficiency(module)
    agentic = SimpleNamespace(SpecValidationError=SpecValidationError)
    messages = [
        {"role": "system", "content": "Return compact research JSON."},
        {
            "role": "user",
            "content": json.dumps({"domain": {"domain_id": "request"}}),
        },
    ]
    result = module._generate_bounded(
        agentic,
        Router(),
        messages=messages,
        response_schema=_research_schema(),
        parser=lambda raw: json.loads(raw)["research_note"],
        progress_label="domain request synthesis 0:0",
    )

    assert result["domain_id"] == "request"
    assert len(calls) == 1
    call = calls[0]
    assert call["response_format"] == "json"
    assert call["enable_tools"] is False
    bound = call["response_schema"]
    note_schema = bound["properties"]["research_note"]
    assert note_schema["properties"]["domain_id"]["const"] == "request"
    assert note_schema["allOf"]
    assert "do not return a bare research_note body" in call["messages"][0]["content"]
    assert module._SYNTHESIS_INPUT_BYTES >= contract._MIN_SYNTHESIS_INPUT_BYTES


def test_parser_validation_failure_is_not_replayed_by_bounded_layer():
    class SpecValidationError(RuntimeError):
        pass

    calls = 0

    class Router:
        def generate_text(self, role, messages, **kwargs):
            nonlocal calls
            del role, messages, kwargs
            calls += 1
            return '{"research_note":{}}'

    module = _bounded_module()
    contract._install_bounded_research_efficiency(module)
    agentic = SimpleNamespace(SpecValidationError=SpecValidationError)

    def invalid_parser(raw):
        del raw
        raise SpecValidationError("semantic mismatch")

    with pytest.raises(module._BoundedResearchOutputError, match="after host repair"):
        module._generate_bounded(
            agentic,
            Router(),
            messages=[
                {"role": "system", "content": "research"},
                {
                    "role": "user",
                    "content": json.dumps({"domain": {"domain_id": "request"}}),
                },
            ],
            response_schema=_research_schema(),
            parser=invalid_parser,
            progress_label="domain request synthesis 0:0",
        )

    assert calls == 1


def test_page_schema_removes_model_owned_continuation_before_generation():
    module = _bounded_module()
    schema = _research_schema()
    schema["properties"]["continuation"] = {
        "type": "object",
        "properties": {
            "complete": {"type": "boolean"},
            "next_offset": {"type": "integer"},
            "tail_sha256": {"type": "string"},
        },
        "required": ["complete", "next_offset", "tail_sha256"],
        "additionalProperties": False,
    }
    schema["required"] = ["research_note", "continuation"]
    messages = [
        {"role": "system", "content": "read page"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "domain": {"domain_id": "mk_platform"},
                    "continuation_contract": {"current_offset": 100},
                    "evidence_page": {
                        "content_total_chars": 1800,
                        "tail_sha256": "sha256:tail",
                    },
                }
            ),
        },
    ]

    bound, domain_id = contract._bound_research_schema(module, schema, messages)

    assert domain_id == "mk_platform"
    assert "continuation" not in bound["properties"]
    assert "required" not in bound
    rendered = json.dumps(bound, sort_keys=True)
    assert "next_offset" not in rendered
    assert "tail_sha256" not in rendered


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
    assert all(contract._json_bytes(group) <= module._SYNTHESIS_INPUT_BYTES for group in groups)
    assert module._SYNTHESIS_INPUT_BYTES >= contract._MIN_SYNTHESIS_INPUT_BYTES
    assert module._SYNTHESIS_PROTOCOL_SCHEMA.endswith("-v4")


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
    failures: list[dict[str, str]] = []

    result = module._hierarchical_synthesis(
        None,
        None,
        prompt="make platform",
        domain={"domain_id": "mk_platform"},
        page_notes=[_rich_note() for _ in range(4)],
        domain_key="unit-test",
        failures=failures,
    )

    assert result["domain_id"] == "mk_platform"
    assert calls == [(0, 2), (0, 2), (1, 2)]
    assert contract._json_bytes(result) <= contract._SYNTHESIS_NODE_BYTES
    assert failures == []


def test_recovery_expansion_is_collapsed_on_host_before_next_model_level():
    frontier_sizes: list[int] = []

    def synthesize(*args, **kwargs):
        group = kwargs["group"]
        frontier_sizes.append(len(group))
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


def test_raw_evidence_leaves_reach_first_synthesis_before_host_compaction():
    delivered: list[str] = []

    def base_group(notes):
        return [[note] for note in notes]

    def synthesize(*args, **kwargs):
        group = kwargs["group"]
        fragment = group[0].get("evidence_fragment")
        if isinstance(fragment, dict):
            delivered.append(str(fragment.get("content", "")))
        return [
            {
                "domain_id": "mk_platform",
                "claims": [],
                "gaps": [],
                "next_queries": [],
                "sufficient": True,
            }
        ]

    module = SimpleNamespace(
        _SYNTHESIS_PROTOCOL_SCHEMA="v2",
        _SYNTHESIS_INPUT_BYTES=3600,
        _group_synthesis_notes=base_group,
        _synthesize_group_with_recovery=synthesize,
        _emit_research_progress=lambda *args, **kwargs: None,
    )
    contract._install_synthesis_convergence(module)
    page_notes = [
        {
            "domain_id": "mk_platform",
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "sufficient": True,
            "evidence_fragment": {
                "page_ref": f"page-{index}",
                "content_sha256": f"sha-{index}",
                "content": content,
            },
        }
        for index, content in enumerate(("RAW-A", "RAW-B"))
    ]
    failures: list[dict[str, str]] = []

    result = module._hierarchical_synthesis(
        None,
        None,
        prompt="x",
        domain={"domain_id": "mk_platform"},
        page_notes=page_notes,
        domain_key="unit-test",
        failures=failures,
    )

    assert result["domain_id"] == "mk_platform"
    assert result["sufficient"] is False
    assert delivered == ["RAW-A", "RAW-B"]
    assert failures and failures[-1]["unit"] == "synthesis:final"


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
