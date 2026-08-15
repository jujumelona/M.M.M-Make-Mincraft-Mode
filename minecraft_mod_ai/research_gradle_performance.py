from __future__ import annotations

"""Persistent Gradle validation runtime with configuration-cache fallback.

Repeated repair validation keeps the Gradle daemon alive, retains build-cache reuse,
and opportunistically uses the configuration cache. A detected compatibility failure
is recorded per project and retried immediately without that optional optimization.
The shared cache lock protects only distribution installation; independent project
builds rely on Gradle's own concurrent cache locking instead of serializing for their
entire build and GameTest duration.
"""

from functools import wraps
from pathlib import Path
from typing import Any, Sequence

from .research_perf_common import env_bool

_MARKER = "_mmm_research_gradle_performance_v1"
_LOCK_MARKER = "_mmm_gradle_distribution_lock_scope_v1"

_CONFIG_CACHE_FAILURE_MARKERS = (
    "configuration cache problems found",
    "configuration cache state could not be cached",
    "configuration cache entry discarded",
    "not supported with the configuration cache",
)


def _optimized_gradle_arguments(arguments: Sequence[str], *, enable_configuration_cache: bool) -> tuple[str, ...]:
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
    """Keep the shared lock around install/extract only, never project execution."""

    cls = runner.GradleRunner
    current_build = cls.build
    current_ensure = cls._ensure_gradle
    if getattr(current_build, _LOCK_MARKER, False):
        return

    @wraps(current_ensure)
    def ensure_gradle(self: Any, gradle_version: str, gradle_sha256: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with runner._exclusive_cache_lock(
            self.cache_dir,
            timeout_seconds=max(300, self.command_timeout_seconds * 3),
        ):
            return current_ensure(self, gradle_version, gradle_sha256)

    @wraps(current_build)
    def build(self: Any, project_root: Path, *, run_gametest: bool = True) -> Any:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # _build_locked owns the canonical build semantics. Its historical name
        # reflects the old outer lock; distribution setup is now fenced by the
        # wrapped _ensure_gradle above, so project execution need not hold it.
        return self._build_locked(project_root, run_gametest=run_gametest)

    ensure_gradle._mmm_gradle_distribution_lock_scope_v1 = True  # type: ignore[attr-defined]
    ensure_gradle.__wrapped__ = current_ensure  # type: ignore[attr-defined]
    build._mmm_gradle_distribution_lock_scope_v1 = True  # type: ignore[attr-defined]
    build.__wrapped__ = current_build  # type: ignore[attr-defined]
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
        if name not in {"clean_build", "gametest"} or not env_bool("MMM_GRADLE_DAEMON", True):
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
        config_cache = env_bool("MMM_GRADLE_CONFIGURATION_CACHE", True) and not incompatible.exists()
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
            incompatible.write_text("auto-disabled after Gradle incompatibility\n", encoding="utf-8")
            fallback = tuple(value for value in optimized if value != "--configuration-cache")
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
