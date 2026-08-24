from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


def _acquire(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0x7FFFFFFF, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0x7FFFFFFF, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def install(runner_module: Any) -> None:
    """Replace race-prone stale-file deletion with kernel-released advisory locks.

    The old O_EXCL lock reclaimed an old file by unlinking it. Two contenders could
    both observe the same stale inode; after one acquired a fresh lock, the other
    could unlink that fresh path and enter the Gradle cache concurrently. Advisory
    locks are released by the kernel when a process exits, so no stale-lock deletion
    is required and a crash cannot permanently poison the cache lock.
    """

    current = runner_module._exclusive_cache_lock
    if getattr(current, "_mmm_os_advisory_cache_lock", False):
        return

    @contextmanager
    def exclusive_cache_lock(
        cache_dir: Path,
        *,
        timeout_seconds: int,
    ) -> Iterable[None]:
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise runner_module.BuildRunnerError(
                "Gradle cache lock timeout must be a positive integer."
            )
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = cache_dir / ".minecraft-mod-ai-cache.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        deadline = time.monotonic() + timeout_seconds
        try:
            while not acquired:
                try:
                    _acquire(fd)
                    acquired = True
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise runner_module.BuildRunnerError(
                            f"Timed out waiting for the Gradle cache lock: {lock_path}"
                        )
                    time.sleep(0.2)

            # Metadata is diagnostic only. The lock file deliberately remains on
            # disk: unlinking a locked pathname would let another process create a
            # different inode and acquire an unrelated lock while this one is active.
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(
                fd,
                f"pid={os.getpid()}\nacquired={time.time()}\n".encode("ascii"),
            )
            os.fsync(fd)
            yield
        finally:
            if acquired:
                try:
                    _release(fd)
                except OSError:
                    pass
            os.close(fd)

    exclusive_cache_lock._mmm_os_advisory_cache_lock = True  # type: ignore[attr-defined]
    exclusive_cache_lock.__wrapped__ = current  # type: ignore[attr-defined]
    runner_module._exclusive_cache_lock = exclusive_cache_lock


__all__ = ["install"]
