import pytest
from ._support import MODEL, replay_text

def test_qwen35_epilogue_is_rejected():
    with pytest.raises(Exception):
        replay_text('{"answer":"ok"}\nI hope this helps.')
