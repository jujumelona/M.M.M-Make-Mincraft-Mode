import pytest
from ._support import MODEL, replay_text

def test_qwen35_semantic_key_drift_is_rejected():
    with pytest.raises(Exception):
        replay_text('{"response":"ok"}')
