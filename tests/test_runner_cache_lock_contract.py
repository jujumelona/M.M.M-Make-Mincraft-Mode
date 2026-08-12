from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import runner
from minecraft_mod_ai.runner_lock_contract import install


def test_preexisting_lock_file_is_not_a_stale_lock_poison(tmp_path: Path) -> None:
    install(runner)
    lock_path = tmp_path / ".minecraft-mod-ai-cache.lock"
    lock_path.write_text("pid=999999\ncreated=0\n", encoding="utf-8")

    with runner._exclusive_cache_lock(tmp_path, timeout_seconds=1):
        assert lock_path.is_file()
        text = lock_path.read_text(encoding="ascii")
        assert "pid=" in text
        assert "acquired=" in text

    # The pathname intentionally remains. Advisory lock ownership is in the kernel,
    # so deleting the file would allow a second inode to bypass a still-held lock.
    assert lock_path.is_file()


def test_cache_lock_rejects_invalid_timeout(tmp_path: Path) -> None:
    install(runner)
    try:
        with runner._exclusive_cache_lock(tmp_path, timeout_seconds=0):
            raise AssertionError("unreachable")
    except runner.BuildRunnerError as exc:
        assert "positive integer" in str(exc)
