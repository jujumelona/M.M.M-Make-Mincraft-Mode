from __future__ import annotations

import pytest

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    _parse_coder_summary,
)


def test_parse_coder_summary_accepts_exact_whole_file_json():
    text = '{"content":"package example; public final class DebugToken {}","summary":"Created deterministic debug token item."}'
    assert _parse_coder_summary(text) == "Created deterministic debug token item."


@pytest.mark.parametrize(
    "text",
    (
        '</think>\n{"content":"class X {}","summary":"wrapped"}',
        '<think>{"phase":"analysis"}</think>\n{"content":"class X {}","summary":"final"}',
        '{"content":"class X {}","summary":"ok","extra":true}',
        '{"content":"class X {}","summary":123}',
        '{"summary":"missing content"}',
        '{"content":"class X {}","summary":"first"}\n{"content":"class X {}","summary":"second"}',
    ),
)
def test_parse_coder_summary_rejects_non_contract_output(text):
    with pytest.raises(CustomModuleGenerationError, match="DIRECT_CODER_INVALID_RESPONSE"):
        _parse_coder_summary(text)
