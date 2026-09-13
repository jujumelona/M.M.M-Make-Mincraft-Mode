import pytest
from ._support import replay_text

def test_qwen35_truncated_string_is_rejected():
    with pytest.raises(Exception):
        replay_text('{"answer":"part')
