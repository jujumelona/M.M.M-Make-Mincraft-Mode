from __future__ import annotations

from contextlib import contextmanager
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



def test_ensure_gradle_honors_explicit_shorter_lock_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner_instance = runner.GradleRunner(
        tmp_path / "cache",
        download_timeout_seconds=300,
    )
    seen: list[int] = []
    sentinel = tmp_path / "gradle"

    @contextmanager
    def fake_lock(_cache_dir, *, timeout_seconds):
        seen.append(timeout_seconds)
        yield

    monkeypatch.setattr(runner, "_exclusive_cache_lock", fake_lock)
    monkeypatch.setattr(
        runner_instance,
        "_ensure_gradle_locked",
        lambda _version, _sha256: sentinel,
    )

    assert (
        runner_instance.ensure_gradle(
            "test",
            "0" * 64,
            lock_timeout_seconds=47,
        )
        == sentinel
    )
    assert seen == [47]

def test_first_gradle_command_serializes_shared_user_home_bootstrap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner_instance = runner.GradleRunner(tmp_path / "cache")
    lock_waits: list[int] = []
    inner_calls: list[str] = []

    @contextmanager
    def fake_lock(_cache_dir, *, timeout_seconds):
        lock_waits.append(timeout_seconds)
        yield

    def fake_run_unlocked(**kwargs):
        inner_calls.append(kwargs["name"])
        return runner.CommandResult(
            name=kwargs["name"],
            command=("gradle",),
            exit_code=0,
            duration_seconds=0.01,
            log_path=str(kwargs["log_path"]),
            timed_out=False,
        )

    monkeypatch.setattr(runner, "_exclusive_cache_lock", fake_lock)
    monkeypatch.setattr(runner_instance, "_run_unlocked", fake_run_unlocked)

    call = dict(
        name="compile_java",
        executable=tmp_path / "gradle",
        arguments=("--no-daemon", "compileJava"),
        cwd=tmp_path,
        env={},
        log_path=tmp_path / "compile.log",
    )
    first = runner_instance._run(**call)
    second = runner_instance._run(**call)

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert inner_calls == ["compile_java", "compile_java"]
    assert len(lock_waits) == 1
    assert (
        runner_instance.cache_dir
        / ".minecraft-mod-ai-gradle-user-home-bootstrap-v1.ready"
    ).is_file()


def test_failed_gradle_command_does_not_publish_bootstrap_ready_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner_instance = runner.GradleRunner(tmp_path / "cache")
    lock_waits: list[int] = []

    @contextmanager
    def fake_lock(_cache_dir, *, timeout_seconds):
        lock_waits.append(timeout_seconds)
        yield

    def fake_run_unlocked(**kwargs):
        return runner.CommandResult(
            name=kwargs["name"],
            command=("gradle",),
            exit_code=1,
            duration_seconds=0.01,
            log_path=str(kwargs["log_path"]),
            timed_out=False,
        )

    monkeypatch.setattr(runner, "_exclusive_cache_lock", fake_lock)
    monkeypatch.setattr(runner_instance, "_run_unlocked", fake_run_unlocked)

    call = dict(
        name="compile_java",
        executable=tmp_path / "gradle",
        arguments=("--no-daemon", "compileJava"),
        cwd=tmp_path,
        env={},
        log_path=tmp_path / "compile.log",
    )
    runner_instance._run(**call)
    runner_instance._run(**call)

    assert len(lock_waits) == 2
    assert not (
        runner_instance.cache_dir
        / ".minecraft-mod-ai-gradle-user-home-bootstrap-v1.ready"
    ).exists()

