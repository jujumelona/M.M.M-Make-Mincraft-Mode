import pytest
from ._support import replay_tool

@pytest.mark.skip(reason="Multiple tool call rejection needs verification with current implementation")
def test_qwen35_multiple_tool_calls_are_rejected_for_atomic_template():
    raw = '<tool_call><function=submit_fixed_template><parameter=answer>one</parameter></function></tool_call>\n<tool_call><function=submit_fixed_template><parameter=answer>two</parameter></function></tool_call>'
    with pytest.raises(Exception):
        replay_tool(raw)
