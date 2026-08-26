from __future__ import annotations

"""Persistent Gradle validation runtime with configuration-cache fallback.

Repeated repair validation keeps the Gradle daemon alive, retains build-cache reuse,
and opportunistically uses the configuration cache. A detected compatibility failure
is recorded per project and retried immediately without that optional optimization.
The shared cache lock protects distribution installation; independent project builds
keep the canonical exact-input cache while avoiding one process-wide outer build lock.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

from .research_perf_common import env_bool

_MARKER = "_mmm_research_gradle_performance_v1"
_LOCK_MARKER = "_mmm_gradle_distribution_lock_scope_v2"
_BYPASS_OUTER_CACHE_LOCK: ContextVar[bool] = ContextVar(
    "mmm_gradle_bypass_outer_cache_lock",
    default=False,
)

_CONFIG_CACHE_FAILURE_MARKERS = (
    "configuration cache problems found",
    "configuration cache state could not be cached",
    "configuration cache entry discarded",
    "not supported with the configuration cache",
)


def _optimized_gradle_arguments(
    arguments: Sequence[str],
    *,
    enable_configuration_cache: bool,
) -> tuple[str, ...]:
    values = [str(value) for value in arguments if str(value) != "--no-daemon"]
    if "--daemon" not in values:
        values.insert(0, "--daemon")
    if "--build-cache" not in values:
        values.append("--build-cache")
    if enable_configuration_cache and "--configuration-cache" not in values:
        values.append("--configuration-cache")
    return tuple(values)


def _configuration_cache_failure(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace").casefold()
    except OSError:
        return False
    return any(marker in text for marker in _CONFIG_CACHE_FAILURE_MARKERS)


def _install_gradle_lock_scope(runner: Any) -> None:
    """Fence distribution mutation without bypassing higher-level build caching."""

    cls = runner.GradleRunner
    current_build = cls.build
    current_ensure = cls._ensure_gradle
    current_lock = runner._exclusive_cache_lock
    if getattr(current_build, _LOCK_MARKER, False):
        return

    def scoped_cache_lock(cache_dir: Path, *, timeout_seconds: int):
        if _BYPASS_OUTER_CACHE_LOCK.get():
            return nullcontext()
        return current_lock(cache_dir, timeout_seconds=timeout_seconds)

    scoped_cache_lock._mmm_gradle_outer_build_bypass = True  # type: ignore[attr-defined]
    scoped_cache_lock.__wrapped__ = current_lock  # type: ignore[attr-defined]

    @wraps(current_ensure)
    def ensure_gradle(self: Any, *args: Any, **kwargs: Any) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Distribution download/extraction mutates shared files and must always use
        # the real advisory lock, even when called inside a build whose outer lock is
        # intentionally bypassed.
        token = _BYPASS_OUTER_CACHE_LOCK.set(False)
        try:
            with current_lock(
                self.cache_dir,
                timeout_seconds=max(300, self.command_timeout_seconds * 3),
            ):
                return current_ensure(self, *args, **kwargs)
        finally:
            _BYPASS_OUTER_CACHE_LOCK.reset(token)

    @wraps(current_build)
    def build(self: Any, project_root: Path, *, run_gametest: bool = True) -> Any:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Keep the complete wrapper chain (notably exact-input successful-build
        # reuse). Only the historical outer shared-cache lock in the base runner is
        # suppressed; _ensure_gradle above still fences shared distribution mutation.
        token = _BYPASS_OUTER_CACHE_LOCK.set(True)
        try:
            return current_build(self, project_root, run_gametest=run_gametest)
        finally:
            _BYPASS_OUTER_CACHE_LOCK.reset(token)

    setattr(ensure_gradle, _LOCK_MARKER, True)
    ensure_gradle.__wrapped__ = current_ensure  # type: ignore[attr-defined]
    setattr(build, _LOCK_MARKER, True)
    build.__wrapped__ = current_build  # type: ignore[attr-defined]
    runner._exclusive_cache_lock = scoped_cache_lock
    cls._ensure_gradle = ensure_gradle
    cls.build = build


def _install_gradle_reuse(runner: Any) -> None:
    cls = runner.GradleRunner
    current = cls._run
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def run(
        self: Any,
        *,
        name: str,
        executable: Path,
        arguments: tuple[str, ...],
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> Any:
        if name not in {"build", "clean_build", "gametest"} or not env_bool(
            "MMM_GRADLE_DAEMON",
            True,
        ):
            return current(
                self,
                name=name,
                executable=executable,
                arguments=arguments,
                cwd=cwd,
                env=env,
                log_path=log_path,
            )
        state_dir = cwd / ".minecraft_ai"
        state_dir.mkdir(parents=True, exist_ok=True)
        incompatible = state_dir / "gradle-configuration-cache-incompatible"
        config_cache = (
            env_bool("MMM_GRADLE_CONFIGURATION_CACHE", True)
            and not incompatible.exists()
        )
        optimized = _optimized_gradle_arguments(
            arguments,
            enable_configuration_cache=config_cache,
        )
        result = current(
            self,
            name=name,
            executable=executable,
            arguments=optimized,
            cwd=cwd,
            env=env,
            log_path=log_path,
        )
        if (
            config_cache
            and int(getattr(result, "exit_code", 1)) != 0
            and _configuration_cache_failure(log_path)
        ):
            incompatible.write_text(
                "auto-disabled after Gradle incompatibility\n",
                encoding="utf-8",
            )
            fallback = tuple(
                value for value in optimized if value != "--configuration-cache"
            )
            return current(
                self,
                name=name,
                executable=executable,
                arguments=fallback,
                cwd=cwd,
                env=env,
                log_path=log_path,
            )
        return result

    setattr(run, _MARKER, True)
    run.__wrapped__ = current  # type: ignore[attr-defined]
    cls._run = run


def harden(runner_module: Any) -> None:
    _install_gradle_lock_scope(runner_module)
    _install_gradle_reuse(runner_module)


__all__ = ["harden"]
