from ._support import replay_text


def test_qwen35_epilogue_recovers_only_the_unique_schema_valid_record():
    assert replay_text('{"answer":"ok"}\nI hope this helps.') == {"answer": "ok"}
