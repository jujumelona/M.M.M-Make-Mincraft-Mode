from __future__ import annotations

import inspect

from minecraft_mod_ai import platform_live_discovery as platform


def test_platform_parallel_io_uses_shared_deadline_executor() -> None:
    source = inspect.getsource(platform)
    assert "ThreadPoolExecutor" not in source
    assert "pool.map" not in source
    assert "future.result()" not in source
    assert "iter_completed_with_deadlines" in source
    for stage in (
        "platform-stable-java-probes",
        "platform-common-metadata",
        "platform-fabric-target",
    ):
        assert stage in source


def test_game_version_prefetch_wait_is_transport_derived_and_bounded() -> None:
    source = inspect.getsource(platform.discover_game_versions)
    assert "future.result(" in source
    assert "timeout=_request_budget_seconds(" in source
    assert "FutureTimeoutError" in source
    assert "_GAME_VERSION_FUTURE = None" in source


def test_request_budget_is_exactly_transport_attempts_plus_backoff(monkeypatch) -> None:
    monkeypatch.setattr(platform, "_RETRY_DELAYS", (0.25, 0.75, 1.5))
    assert platform._request_budget_seconds(timeout=20, retries=1) == 20.0
    assert platform._request_budget_seconds(timeout=20, retries=4) == 82.5


def test_transport_timeouts_remain_finite() -> None:
    source = inspect.getsource(platform)
    assert "timeout: int = 20" in source
    assert "jar = _fetch(jar_url, timeout=90, retries=2)" in source
