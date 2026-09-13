import pytest
from ._support import MODEL, replay_text

def test_qwen35_markdown_fenced_json_is_rejected_by_strict_transport():
    with pytest.raises(Exception):
        replay_text('```json\n{"answer":"ok"}\n```')
