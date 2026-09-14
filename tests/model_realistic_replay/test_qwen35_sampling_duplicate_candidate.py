import pytest

from ._support import replay_text


def test_qwen35_sampling_duplicate_candidate_is_rejected():
    # Synthetic replay of a high-entropy sampling failure shape: the model emits
    # two independently valid candidates instead of one schema instance.
    raw = '{"answer":"alpha"}\n{"answer":"beta"}'
    with pytest.raises(Exception):
        replay_text(raw)
