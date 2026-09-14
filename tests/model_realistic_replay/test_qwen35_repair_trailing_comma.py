import pytest

from ._support import replay_text


def test_qwen35_repair_trailing_comma_is_not_silently_repaired():
    # Synthetic replay of a repair-candidate failure shape. Production parsing
    # must reject malformed JSON rather than inventing a corrected completion.
    with pytest.raises(Exception):
        replay_text('{"answer":"ok",}')
