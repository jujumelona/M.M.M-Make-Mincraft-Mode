import pytest
from ._support import replay_text

def test_qwen35_think_closed_before_json_is_not_silently_accepted():
    raw = '<think>I should satisfy the schema exactly.</think>\n{"answer":"ok"}'
    with pytest.raises(Exception):
        replay_text(raw)
