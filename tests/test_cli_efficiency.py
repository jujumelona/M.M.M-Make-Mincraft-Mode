import hashlib
import json
from pathlib import Path

import pytest

from minecraft_mod_ai.cli import _read_playtest_actions, _sha256_file


def test_sha256_file_streams_without_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (b"mmm-streaming-hash\0" * 140_000) + b"tail"
    source = tmp_path / "existing-project.zip"
    source.write_bytes(payload)
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()

    def fail_read_bytes(_self: Path) -> bytes:
        raise AssertionError("streaming hash must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    assert _sha256_file(source) == expected


def test_read_playtest_actions_accepts_object_list(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    expected = [{"action": "move"}, {"action": "interact", "target": "door"}]
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert _read_playtest_actions(path) == expected


def test_read_playtest_actions_rejects_non_object_member(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    path.write_text('[{"action":"move"}, "invalid"]', encoding="utf-8")

    with pytest.raises(ValueError, match="Every playtest action"):
        _read_playtest_actions(path)
