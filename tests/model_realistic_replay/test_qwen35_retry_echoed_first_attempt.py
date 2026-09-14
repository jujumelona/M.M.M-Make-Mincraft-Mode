import pytest

from ._support import replay_text


def test_qwen35_retry_echoed_first_attempt_is_rejected():
    # Synthetic replay of a retry failure shape where the model echoes the
    # previous malformed candidate before emitting the corrected candidate.
    raw = '{"answer":}\n{"answer":"ok"}'
    with pytest.raises(Exception):
        replay_text(raw)
