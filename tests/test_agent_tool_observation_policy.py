from __future__ import annotations

import json

from minecraft_mod_ai.agent_tool_runtime import _bounded_result, _redact_text


def test_agent_observation_redacts_secrets_and_marks_untrusted() -> None:
    result = _bounded_result(
        {
            "receipt": {
                "result_count": 2,
                "coverage_score": 0.9,
                "relevance_score": 0.8,
            },
            "headers": {"Authorization": "Bearer super-secret-token"},
            "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
            "nested": {"client_secret": "do-not-leak"},
            "text": "password=hunter2 safe evidence text",
        }
    )

    encoded = json.dumps(result, ensure_ascii=False)
    assert result["_mmm_observation"] == {
        "trust": "untrusted_data_only",
        "sanitized": True,
        "truncated": False,
    }
    assert result["headers"]["Authorization"] == "[REDACTED]"
    assert result["api_key"] == "[REDACTED]"
    assert result["nested"]["client_secret"] == "[REDACTED]"
    assert "super-secret-token" not in encoded
    assert "abcdefghijklmnopqrstuvwxyz" not in encoded
    assert "hunter2" not in encoded


def test_large_observation_preserves_rag_receipt(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_OBSERVATION_BYTES", "8192")
    receipt = {
        "result_count": 4,
        "coverage_score": 0.75,
        "relevance_score": 0.88,
    }
    result = _bounded_result(
        {
            "receipt": receipt,
            "hits": [{"text": "x" * 12000}],
            "next_cursor": "page-2",
        }
    )

    assert result["truncated"] is True
    assert result["_mmm_observation"]["truncated"] is True
    assert {"receipt": receipt} in result["preserved_evidence"]
    assert {"next_cursor": "page-2"} in result["preserved_evidence"]
    assert len(result["preview"].encode("utf-8")) <= 4096


def test_free_text_secret_redaction_covers_auth_and_private_keys() -> None:
    private_key = (
        "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----"
    )
    redacted = _redact_text(
        "Authorization: Bearer abcdefghijklmnop "
        "access_token=qrstuvwxyz012345 "
        + private_key
    )

    assert "abcdefghijklmnop" not in redacted
    assert "qrstuvwxyz012345" not in redacted
    assert "abc123" not in redacted
    assert "[REDACTED]" in redacted
    assert "[REDACTED_PRIVATE_KEY]" in redacted
