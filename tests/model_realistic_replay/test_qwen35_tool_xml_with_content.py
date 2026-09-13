from ._support import replay_tool

def test_qwen35_content_plus_native_tool_call_is_parsed():
    raw = 'I will submit the fixed template.\n<tool_call>\n{"name":"submit_fixed_template","arguments":{"answer":"ok"}}\n</tool_call>'
    assert replay_tool(raw) == {"answer":"ok"}
