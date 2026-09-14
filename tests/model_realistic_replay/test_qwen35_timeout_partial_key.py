import pytest

from ._support import replay_text


def test_qwen35_timeout_partial_key_is_rejected():
    # Synthetic replay of timeout/token-cutoff morphology: transport returns a
    # partial JSON key before the completion can close the object.
    with pytest.raises(Exception):
        replay_text('{"ans')
