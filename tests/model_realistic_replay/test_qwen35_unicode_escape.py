from ._support import MODEL, replay_text

def test_qwen35_unicode_and_escape_payload_survives_validation():
    assert replay_text('{"answer":"한글 \\u2713"}') == {"answer":"한글 ✓"}
