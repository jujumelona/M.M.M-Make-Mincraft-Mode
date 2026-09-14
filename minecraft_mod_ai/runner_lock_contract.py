from __future__ import annotations

import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any


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


def _install_cache_lock(runner_module: Any) -> None:
    current_lock = runner_module._exclusive_cache_lock
    if getattr(current_lock, "_mmm_os_advisory_cache_lock", False):
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
    exclusive_cache_lock.__wrapped__ = current_lock  # type: ignore[attr-defined]
    runner_module._exclusive_cache_lock = exclusive_cache_lock


def _install_gradle_lock(runner_module: Any) -> None:
    cls = runner_module.GradleRunner
    current_ensure = cls._ensure_gradle
    if getattr(current_ensure, "_mmm_distribution_lock_owner", False):
        return

    @wraps(current_ensure)
    def ensure_gradle(self: Any, gradle_version: str, gradle_sha256: str) -> Path:
        with runner_module._exclusive_cache_lock(
            self.cache_dir,
            timeout_seconds=max(300, self.download_timeout_seconds * 3),
        ):
            return current_ensure(self, gradle_version, gradle_sha256)

    ensure_gradle._mmm_distribution_lock_owner = True  # type: ignore[attr-defined]
    cls._ensure_gradle = ensure_gradle


def _install_build_contract(runner_module: Any) -> None:
    cls = runner_module.GradleRunner
    current_build = cls.build
    if getattr(current_build, "_mmm_narrow_gradle_cache_lock", False):
        return

    @wraps(current_build)
    def build(self: Any, project_root: Path, *, run_gametest: bool = True) -> Any:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self._build_locked(project_root, run_gametest=run_gametest)

    build._mmm_narrow_gradle_cache_lock = True  # type: ignore[attr-defined]
    cls.build = build


def install(runner_module: Any) -> None:
    """Keep the interprocess lock only around Gradle distribution mutation.

    A Gradle build uses Gradle's own user-home/cache coordination. Holding MMM's
    distribution-install lock across the complete build serializes unrelated projects
    and permits accidental nested acquisition. The OS advisory lock therefore owns
    only `_ensure_gradle`, where MMM itself mutates the shared distribution cache.
    """

    _install_cache_lock(runner_module)
    _install_gradle_lock(runner_module)
    _install_build_contract(runner_module)


__all__ = ["install"]
