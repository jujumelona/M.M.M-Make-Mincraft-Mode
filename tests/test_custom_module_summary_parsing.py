from __future__ import annotations

import pytest

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    _parse_coder_summary,
)


def test_parse_coder_summary_accepts_exact_fixed_json():
    assert _parse_coder_summary('{"summary":"Created deterministic debug token item."}') == (
        "Created deterministic debug token item."
    )


@pytest.mark.parametrize(
    "text",
    (
        '</think>\n{"summary":"wrapped"}',
        '<think>{"phase":"analysis"}</think>\n{"summary":"final"}',
        '{"summary":"ok","extra":true}',
        '{"summary":123}',
        '{"summary":"first"}\n{"summary":"second"}',
    ),
)
def test_parse_coder_summary_rejects_non_contract_output(text):
    with pytest.raises(CustomModuleGenerationError, match="RESPONSE_TEMPLATE"):
        _parse_coder_summary(text)
