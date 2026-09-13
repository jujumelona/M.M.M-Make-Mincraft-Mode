import pytest
from ._support import replay_text

def test_qwen35_helpful_preamble_is_rejected():
    with pytest.raises(Exception):
        replay_text('Sure, here is the result:\n{"answer":"ok"}')
