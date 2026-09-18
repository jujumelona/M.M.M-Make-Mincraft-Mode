from __future__ import annotations

import pytest

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    _parse_coder_summary,
)


def test_parse_coder_summary_accepts_reasoning_wrapper_before_fixed_json():
    value = _parse_coder_summary(
        '</think>\n\n{"summary":"Created deterministic debug token item."}'
    )
    assert value == "Created deterministic debug token item."


def test_parse_coder_summary_requires_exact_fixed_field():
    with pytest.raises(CustomModuleGenerationError, match="exactly"):
        _parse_coder_summary('{"summary":"ok","extra":true}')


def test_parse_coder_summary_requires_string_value():
    with pytest.raises(CustomModuleGenerationError, match="must be a string"):
        _parse_coder_summary('{"summary":123}')


def test_parse_coder_summary_ignores_unrelated_reasoning_json():
    value = _parse_coder_summary(
        '<think>{"phase":"analysis"}</think>\n{"summary":"final"}'
    )
    assert value == "final"


def test_parse_coder_summary_rejects_multiple_summary_objects():
    with pytest.raises(CustomModuleGenerationError, match="multiple"):
        _parse_coder_summary(
            '{"summary":"first"}\n{"summary":"second"}'
        )
