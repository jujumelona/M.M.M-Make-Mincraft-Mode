from __future__ import annotations

import io
import queue

import pytest

from minecraft_mod_ai.mineflayer_bridge import MineflayerBridge, MineflayerBridgeError


class _HungProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.killed = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode


def test_call_times_out_instead_of_blocking_forever(monkeypatch, tmp_path) -> None:
    bridge_file = tmp_path / "bridge.mjs"
    bridge_file.write_text("", encoding="utf-8")
    bridge = MineflayerBridge(bridge_file)
    process = _HungProcess()

    def fake_start() -> None:
        bridge.process = process
        bridge._response_lines = queue.Queue()

    monkeypatch.setattr(bridge, "start", fake_start)
    with pytest.raises(MineflayerBridgeError, match="timed out"):
        bridge.call("status", timeout_seconds=0.02)

    assert process.killed is True
    assert bridge.process is None


def test_invalid_timeout_is_rejected_before_start(monkeypatch, tmp_path) -> None:
    bridge_file = tmp_path / "bridge.mjs"
    bridge_file.write_text("", encoding="utf-8")
    bridge = MineflayerBridge(bridge_file)
    called = False

    def fake_start() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(bridge, "start", fake_start)
    with pytest.raises(MineflayerBridgeError, match="finite positive"):
        bridge.call("status", timeout_seconds=0)
    assert called is False
