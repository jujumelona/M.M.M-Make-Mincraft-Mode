import pytest
from ._support import replay_tool

def test_qwen35_nearby_tool_name_hallucination_is_rejected():
    raw = '<tool_call><function=submit_template><parameter=answer>ok</parameter></function></tool_call>'
    with pytest.raises(Exception):
        replay_tool(raw)
