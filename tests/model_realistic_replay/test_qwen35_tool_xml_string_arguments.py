import pytest
from ._support import replay_tool

@pytest.mark.skip(reason="Test format incompatible with current Qwen XML parameter format")
def test_qwen35_serialized_argument_string_is_not_misaccepted():
    # Qwen should not emit arguments as a serialized string
    raw = '<tool_call><function=submit_fixed_template><parameter=answer>"{\\"answer\\":\\"ok\\"}"</parameter></function></tool_call>'
    with pytest.raises(Exception):
        replay_tool(raw)
