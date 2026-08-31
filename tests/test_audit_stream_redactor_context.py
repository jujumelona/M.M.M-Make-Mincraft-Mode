from __future__ import annotations

import pytest

from tools.audit_stream_redactor import StreamingRedactor


def test_context_safe_replacement_survives_every_small_chunk_boundary() -> None:
    exact_secret = "exact-secret-value"
    source = f"prefix/{exact_secret} token=label-secret; suffix"
    expected = "prefix/&lt;redacted&gt; token=&lt;redacted&gt;; suffix"

    for width in range(1, 17):
        redactor = StreamingRedactor(
            (exact_secret,),
            replacement="&lt;redacted&gt;",
        )
        rendered = "".join(
            redactor.feed(source[index : index + width])
            for index in range(0, len(source), width)
        ) + redactor.finish()
        assert rendered == expected, f"chunk width {width} broke custom replacement"
        assert exact_secret not in rendered
        assert "label-secret" not in rendered
        assert "<redacted>" not in rendered


def test_unquoted_multiword_sensitive_value_is_fully_redacted() -> None:
    source = "token=first secret word with spaces; visible=ok"
    expected = "token=<redacted>; visible=ok"

    for width in range(1, 21):
        redactor = StreamingRedactor()
        rendered = "".join(
            redactor.feed(source[index : index + width])
            for index in range(0, len(source), width)
        ) + redactor.finish()
        assert rendered == expected, f"chunk width {width} leaked a multiword value"
        assert "first" not in rendered
        assert "secret word" not in rendered
        assert "spaces" not in rendered


def test_empty_replacement_is_rejected() -> None:
    with pytest.raises(ValueError, match="replacement must be a non-empty string"):
        StreamingRedactor(replacement="")
