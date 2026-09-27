from __future__ import annotations

import pytest

from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    _plain_coder_output,
)


def test_plain_coder_output_accepts_source_and_normalizes_terminal_newline() -> None:
    text = "package example;\r\npublic final class DebugToken {}"

    assert _plain_coder_output(text) == (
        "package example;\npublic final class DebugToken {}\n"
    )


@pytest.mark.parametrize("text", ("", "   ", "\r\n\t"))
def test_plain_coder_output_rejects_empty_source(text: str) -> None:
    with pytest.raises(
        CustomModuleGenerationError,
        match="DIRECT_CODER_EMPTY_RESPONSE",
    ):
        _plain_coder_output(text)
