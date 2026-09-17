from __future__ import annotations

import ast
from concurrent.futures import Future
import importlib.util
from pathlib import Path

import pytest

from minecraft_mod_ai import deadline_executor


class _TrackingExecutor:
    instances: list["_TrackingExecutor"] = []

    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.shutdown_calls: list[tuple[bool, bool]] = []
        type(self).instances.append(self)

    def submit(self, function, /, *args, **kwargs):
        future: Future[object] = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def _install_tracking_executor(monkeypatch) -> None:
    _TrackingExecutor.instances.clear()
    monkeypatch.setattr(deadline_executor, "ThreadPoolExecutor", _TrackingExecutor)


def test_iterator_owns_executor_until_stream_exhaustion(monkeypatch) -> None:
    _install_tracking_executor(monkeypatch)

    results = deadline_executor.iter_completed_with_deadlines(
        [1, 2, 3],
        lambda value: value * 10,
        max_workers=2,
        stage="lifecycle-test",
    )

    assert _TrackingExecutor.instances == []
    assert next(results) == (1, 10)
    assert len(_TrackingExecutor.instances) == 1
    executor = _TrackingExecutor.instances[0]
    assert executor.shutdown_calls == []
    assert list(results) == [(2, 20), (3, 30)]
    assert executor.shutdown_calls == [(False, True)]


def test_worker_failure_still_closes_executor(monkeypatch) -> None:
    _install_tracking_executor(monkeypatch)

    def fail(value: int) -> int:
        raise ValueError(f"bad-{value}")

    with pytest.raises(deadline_executor.ParallelTaskError) as exc_info:
        deadline_executor.collect_completed_with_deadlines(
            [7],
            fail,
            max_workers=1,
            stage="failure-test",
        )

    assert exc_info.value.item == 7
    assert isinstance(exc_info.value.cause, ValueError)
    assert _TrackingExecutor.instances[0].shutdown_calls == [(False, True)]


def _load_concurrency_audit():
    root = Path(__file__).resolve().parents[1]
    path = root / ".github" / "scripts" / "audit_runtime_concurrency.py"
    spec = importlib.util.spec_from_file_location("audit_runtime_concurrency_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concurrency_audit_rejects_direct_executor_ownership(tmp_path, monkeypatch) -> None:
    audit = _load_concurrency_audit()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "minecraft_mod_ai" / "rogue.py"
    path.parent.mkdir(parents=True)
    source = (
        "from concurrent.futures import ThreadPoolExecutor as Pool\n"
        "pool = Pool(max_workers=2)\n"
    )
    path.write_text(source, encoding="utf-8")

    findings = audit.executor_ownership_violations(path, ast.parse(source))

    assert findings
    assert any("ThreadPoolExecutor" in detail for _, detail in findings)


def test_concurrency_audit_allows_only_canonical_executor_owner(tmp_path, monkeypatch) -> None:
    audit = _load_concurrency_audit()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "minecraft_mod_ai" / "deadline_executor.py"
    path.parent.mkdir(parents=True)
    source = (
        "from concurrent.futures import ThreadPoolExecutor\n"
        "pool = ThreadPoolExecutor(max_workers=1)\n"
    )
    path.write_text(source, encoding="utf-8")

    assert audit.executor_ownership_violations(path, ast.parse(source)) == []
