from __future__ import annotations

import pytest

from minecraft_mod_ai.model_runtime_performance import (
    _length_prefixed_digest,
    _text_digest,
)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "minecraft",
        "한글과 emoji 🚀",
        "x" * 100_000,
    ],
)
def test_text_digest_matches_sequence_digest_contract(text: str) -> None:
    assert _text_digest(text) == _length_prefixed_digest((text,))
