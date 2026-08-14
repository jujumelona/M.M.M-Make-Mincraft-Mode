from __future__ import annotations

"""Persistent Gradle validation runtime with configuration-cache fallback.

Repeated repair validation keeps the Gradle daemon alive, retains build-cache reuse,
and opportunistically uses the configuration cache. A detected compatibility failure
is recorded per project and retried immediately without that optional optimization.
"""

from functools import wraps
from pathlib import Path
from typing import Any, Sequence

from .research_perf_common import env_bool

_MARKER = "_mmm_research_gradle_performance_v1"

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
        if name not in {"build", "gametest"} or not env_bool("MMM_GRADLE_DAEMON", True):
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
    _install_gradle_reuse(runner_module)


__all__ = ["harden"]
