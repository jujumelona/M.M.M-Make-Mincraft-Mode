from ._support import replay_text


def test_qwen35_preamble_recovers_only_the_unique_schema_valid_record():
    assert replay_text('Sure, here is the result:\n{"answer":"ok"}') == {"answer": "ok"}
