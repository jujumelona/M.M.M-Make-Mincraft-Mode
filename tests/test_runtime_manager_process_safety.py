from __future__ import annotations

import subprocess

from minecraft_mod_ai.runtime_manager import MinecraftRuntimeManager


class _HungProcess:
    def __init__(self) -> None:
        self.stdin = None
        self.terminated = False
        self.killed = False
        self.wait_calls: list[int | float | None] = []

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if not self.killed:
            raise subprocess.TimeoutExpired("runtime", timeout)
        return -9


def test_stop_process_forces_kill_and_reaps_child() -> None:
    manager = MinecraftRuntimeManager.__new__(MinecraftRuntimeManager)
    process = _HungProcess()

    manager._stop_process(process, graceful_server=False)

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == [15, 5]


def test_process_running_uses_poll_state() -> None:
    class _Exited:
        def poll(self):
            return 0

    assert MinecraftRuntimeManager._process_running(None) is False
    assert MinecraftRuntimeManager._process_running(_Exited()) is False
