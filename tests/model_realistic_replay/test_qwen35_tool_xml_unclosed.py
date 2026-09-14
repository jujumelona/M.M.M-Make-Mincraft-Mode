import pytest
from ._support import replay_tool

def test_qwen35_unclosed_tool_xml_is_rejected():
    with pytest.raises(Exception):
        replay_tool('<tool_call><function=submit_fixed_template><parameter=answer>ok</parameter></function>')
