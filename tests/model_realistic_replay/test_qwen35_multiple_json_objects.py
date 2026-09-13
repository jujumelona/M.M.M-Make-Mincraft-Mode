import pytest
from ._support import replay_text

def test_qwen35_multiple_candidate_objects_are_rejected():
    with pytest.raises(Exception):
        replay_text('{"answer":"first"}\n{"answer":"second"}')
