from ._support import replay_tool

def test_qwen35_native_xml_tool_call_reaches_fixed_template():
    raw = '<tool_call>\n{"name":"submit_fixed_template","arguments":{"answer":"ok"}}\n</tool_call>'
    assert replay_tool(raw) == {"answer":"ok"}
