from __future__ import annotations

import json

from minecraft_mod_ai.causal_tool_graph import (
    shortest_causal_path,
    verified_state_from_messages,
)


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool(name: str, result: dict) -> dict:
    return {
        "role": "tool",
        "name": name,
        "content": json.dumps({"ok": True, "tool": name, "result": result}),
    }


def _good_rag_result() -> dict:
    return {
        "structured_content": {
            "hits": [{"path": "src/main/java/example/Mod.java"}],
            "receipt": {
                "result_count": 1,
                "coverage_score": 0.75,
                "relevance_score": 0.8,
            },
        },
        "text": "{}",
    }


def test_nested_runtime_rag_receipt_unlocks_workspace_source_route() -> None:
    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    state = verified_state_from_messages(
        (_tool("search_code_rag", _good_rag_result()),),
        schemas,
        require_fresh_evidence=True,
    )
    assert "project_observed" in state
    assert "code_evidence" in state
    assert "evidence_ready" in state


def test_weak_rag_observes_project_but_does_not_certify_evidence() -> None:
    schemas = (_schema("search_code_rag"),)
    weak = {
        "parsed_text": {
            "receipt": {
                "result_count": 1,
                "coverage_score": 0.1,
                "relevance_score": 0.8,
            }
        }
    }
    state = verified_state_from_messages(
        (_tool("search_code_rag", weak),),
        schemas,
        require_fresh_evidence=True,
    )
    assert "project_observed" in state
    assert "code_evidence" in state
    assert "evidence_ready" not in state


def test_external_partial_transport_success_is_not_causal_evidence() -> None:
    schemas = (_schema("external_mcp_call"),)
    partial = {
        "structured_content": {
            "schema_version": "mmm/external-mcp-evidence-bundle-v1",
            "required_corroboration": 2,
            "status": "PARTIAL",
            "evidence": [{"status": "PASS"}],
        }
    }
    state = verified_state_from_messages((_tool("external_mcp_call", partial),), schemas)
    assert "external_observation" not in state
    assert "evidence_ready" not in state


def test_external_pass_requires_requested_corroboration() -> None:
    schemas = (_schema("external_mcp_call"),)
    bundle = {
        "parsed_text": {
            "schema_version": "mmm/external-mcp-evidence-bundle-v1",
            "required_corroboration": 2,
            "status": "PASS",
            "evidence": [{"status": "PASS"}, {"status": "PASS"}],
        }
    }
    state = verified_state_from_messages((_tool("external_mcp_call", bundle),), schemas)
    assert "external_observation" in state
    assert "evidence_ready" in state


def test_failed_gradle_result_never_becomes_verified() -> None:
    schemas = (_schema("search_code_rag"), _schema("run_gradle_build"))
    messages = (
        _tool("search_code_rag", _good_rag_result()),
        _tool(
            "run_gradle_build",
            {
                "structured_content": {
                    "status": "FAIL",
                    "command": ["./gradlew", "build"],
                    "returncode": 1,
                }
            },
        ),
    )
    state = verified_state_from_messages(messages, schemas)
    assert "build_verified" not in state
    assert "verified" not in state


def test_passing_gradle_result_becomes_verified() -> None:
    schemas = (_schema("search_code_rag"), _schema("run_gradle_build"))
    messages = (
        _tool("search_code_rag", _good_rag_result()),
        _tool(
            "run_gradle_build",
            {
                "structured_content": {
                    "status": "PASS",
                    "command": ["./gradlew", "build"],
                    "returncode": 0,
                }
            },
        ),
    )
    state = verified_state_from_messages(messages, schemas)
    assert "build_verified" in state
    assert "verified" in state


def test_workspace_rag_is_the_direct_precondition_for_source_mutation() -> None:
    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    assert shortest_causal_path(
        schemas,
        state=frozenset({"workspace_bound"}),
        goals=("repair",),
    ) == ("search_code_rag", "apply_source_patch")
