from __future__ import annotations

import pytest

from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.custom_module_generator import _plain_coder_output


def test_plain_coder_output_accepts_complete_source_text() -> None:
    source = "package example;\n\npublic final class DebugToken {}"
    assert _plain_coder_output(source) == source + "\n"


def test_plain_coder_output_normalizes_newlines_and_outer_whitespace() -> None:
    source = "\r\n  package example;\r\npublic final class DebugToken {}  \r\n"
    assert _plain_coder_output(source) == (
        "package example;\npublic final class DebugToken {}\n"
    )


@pytest.mark.parametrize("text", ("", "   ", "\r\n\t\r\n"))
def test_plain_coder_output_rejects_empty_response(text: str) -> None:
    with pytest.raises(
        CustomModuleGenerationError,
        match="DIRECT_CODER_EMPTY_RESPONSE",
    ):
        _plain_coder_output(text)


def test_plain_coder_output_does_not_restore_obsolete_json_summary_contract() -> None:
    payload = (
        '{"content":"package example; public final class DebugToken {}",'
        '"summary":"obsolete"}'
    )
    assert _plain_coder_output(payload) == payload + "\n"
