import pytest
from ._support import replay_text

def test_qwen35_helpful_extra_field_is_rejected():
    with pytest.raises(Exception):
        replay_text('{"answer":"ok","explanation":"because"}')
