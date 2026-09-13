import pytest
from ._support import MODEL, replay_text

def test_qwen35_type_drift_is_rejected():
    with pytest.raises(Exception):
        replay_text('{"answer":["ok"]}')
