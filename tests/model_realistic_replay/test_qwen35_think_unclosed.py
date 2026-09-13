import pytest
from ._support import MODEL, replay_text

def test_qwen35_unclosed_thinking_is_rejected():
    with pytest.raises(Exception):
        replay_text('<think>I should fill answer, then {"answer":"ok"}')
