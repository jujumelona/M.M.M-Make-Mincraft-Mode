import pytest
from ._support import MODEL, replay_tool

def test_qwen35_multiple_tool_calls_are_rejected_for_atomic_template():
    raw = '<tool_call>{"name":"submit_fixed_template","arguments":{"answer":"one"}}</tool_call>\n<tool_call>{"name":"submit_fixed_template","arguments":{"answer":"two"}}</tool_call>'
    with pytest.raises(Exception):
        replay_tool(raw)
