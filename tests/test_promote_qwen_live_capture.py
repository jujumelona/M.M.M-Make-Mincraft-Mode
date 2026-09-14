from __future__ import annotations

import pytest

from tools.promote_qwen_live_capture import render_replay_test


def test_render_replay_accepts_only_real_capture_and_preserves_raw_text() -> None:
    capture = {
        "provenance": "real_capture",
        "conditions": {
            "seed": 17,
            "temperature": 0.8,
            "max_tokens": 16,
            "stop": None,
        },
        "raw_assistant_message": {"role": "assistant", "content": '{"ans'},
    }
    rendered = render_replay_test(capture, "timeout-partial-key")
    assert "test_qwen35_real_timeout_partial_key_remains_rejected" in rendered
    assert repr('{"ans') in rendered
    assert "pytest.raises(Exception)" in rendered


def test_render_replay_rejects_synthetic_capture_provenance() -> None:
    capture = {
        "provenance": "synthetic_qwen35_failure_shape",
        "conditions": {},
        "raw_assistant_message": {"role": "assistant", "content": "bad"},
    }
    with pytest.raises(ValueError, match="provenance=real_capture"):
        render_replay_test(capture, "must-not-promote")
