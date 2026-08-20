from __future__ import annotations

import json

from minecraft_mod_ai.causal_tool_graph import verified_state_from_messages


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


def _rag() -> dict:
    return {
        "structured_content": {
            "receipt": {
                "result_count": 1,
                "coverage_score": 0.8,
                "relevance_score": 0.8,
            }
        }
    }


def test_static_validation_fail_never_certifies_verified() -> None:
    schemas = (_schema("search_code_rag"), _schema("run_static_validation"))
    state = verified_state_from_messages(
        (
            _tool("search_code_rag", _rag()),
            _tool(
                "run_static_validation",
                {"structured_content": {"status": "FAIL", "findings": [{"severity": "error"}]}},
            ),
        ),
        schemas,
    )
    assert "static_verified" not in state
    assert "verified" not in state


def test_gametest_fail_never_certifies_verified() -> None:
    schemas = (_schema("search_code_rag"), _schema("run_gametest"))
    state = verified_state_from_messages(
        (
            _tool("search_code_rag", _rag()),
            _tool(
                "run_gametest",
                {
                    "structured_content": {
                        "status": "FAIL",
                        "commands": [
                            {
                                "status": "FAIL",
                                "command": ["./gradlew", "runGametest"],
                                "returncode": 1,
                            }
                        ],
                    }
                },
            ),
        ),
        schemas,
    )
    assert "gametest_verified" not in state
    assert "verified" not in state


def test_jdt_errors_are_observations_not_verification() -> None:
    schemas = (_schema("search_code_rag"), _schema("java_diagnostics"))
    state = verified_state_from_messages(
        (
            _tool("search_code_rag", _rag()),
            _tool(
                "java_diagnostics",
                {
                    "structured_content": {
                        "schema_version": "mmm/java-diagnostics-v2",
                        "error_count": 2,
                        "warning_count": 0,
                        "diagnostics": {},
                    }
                },
            ),
        ),
        schemas,
    )
    assert "static_verified" not in state
    assert "verified" not in state


def test_clean_jdt_can_certify_static_verification() -> None:
    schemas = (_schema("search_code_rag"), _schema("java_diagnostics"))
    state = verified_state_from_messages(
        (
            _tool("search_code_rag", _rag()),
            _tool(
                "java_diagnostics",
                {
                    "structured_content": {
                        "schema_version": "mmm/java-diagnostics-v2",
                        "error_count": 0,
                        "warning_count": 3,
                        "diagnostics": {},
                    }
                },
            ),
        ),
        schemas,
    )
    assert "static_verified" in state
    assert "verified" in state


def test_pass_external_bundle_may_contain_failed_provider_attempt() -> None:
    schemas = (_schema("external_mcp_call"),)
    state = verified_state_from_messages(
        (
            _tool(
                "external_mcp_call",
                {
                    "structured_content": {
                        "schema_version": "mmm/external-mcp-evidence-bundle-v1",
                        "required_corroboration": 1,
                        "status": "PASS",
                        "evidence": [{"status": "PASS"}],
                        "attempts": [
                            {"server": "first", "status": "ERROR"},
                            {"server": "second", "status": "PASS"},
                        ],
                    }
                },
            ),
        ),
        schemas,
    )
    assert "external_observation" in state
    assert "evidence_ready" in state
