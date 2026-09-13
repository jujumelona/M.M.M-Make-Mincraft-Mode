import pytest
from ._support import MODEL, replay_tool

def test_qwen35_serialized_argument_string_is_not_misaccepted():
    raw = '<tool_call>{"name":"submit_fixed_template","arguments":"{\\"answer\\":\\"ok\\"}"}</tool_call>'
    with pytest.raises(Exception):
        replay_tool(raw)
